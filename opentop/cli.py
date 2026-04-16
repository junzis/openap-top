"""Command-line interface for opentop.

Two subcommands:

    opentop optimize ORIG DEST [...]   # run trajectory optimization
    opentop gengrid   --in IN --out OUT [...]   # build and cache a bspline interpolant

See ``opentop --help`` for the full list.
"""

from __future__ import annotations

import re
import sys
import time
from pathlib import Path
from typing import Any, Callable, Union, cast

import click

import opentop as top
import pandas as pd
from opentop import tools

# ============================================================
# Objective parsing
# ============================================================

_TERM_NAMES = {
    "fuel",
    "grid",
    "time",
    "gwp20",
    "gwp50",
    "gwp100",
    "gtp20",
    "gtp50",
    "gtp100",
}
_TERM_RE = re.compile(r"^\s*(?:([-+.\deE]+)\s*\*\s*)?([a-z0-9]+(?::[-+.\deE]+)?)\s*$")


def parse_objective(spec: str) -> list[tuple[float, str, str | None]]:
    """Parse an objective expression.

    Accepts a sum of weighted terms separated by ``+``. Each term is either
    a bare name (``fuel``, ``grid``, ``time``, ``gwp100``, ...) or a name
    with a scalar parameter (``ci:30``). The weight prefix is optional;
    if omitted, weight defaults to 1.0.

    Examples:
        "fuel"                       -> [(1.0, "fuel", None)]
        "0.3*fuel+0.7*grid"          -> [(0.3, "fuel", None), (0.7, "grid", None)]
        "ci:30"                      -> [(1.0, "ci", "30")]
        "0.5*fuel+0.5*gwp100"        -> [(0.5, "fuel", None), (0.5, "gwp100", None)]
    """
    terms: list[tuple[float, str, str | None]] = []
    for chunk in spec.split("+"):
        m = _TERM_RE.match(chunk)
        if not m:
            raise click.UsageError(f"cannot parse objective term: {chunk!r}")
        weight = float(m.group(1)) if m.group(1) else 1.0
        term = m.group(2)
        if ":" in term:
            name, param = term.split(":", 1)
        else:
            name, param = term, None
        if name not in _TERM_NAMES and name != "ci":
            raise click.UsageError(
                f"unknown objective term {name!r}. Valid: "
                f"{sorted(_TERM_NAMES | {'ci:N'})}"
            )
        if name == "ci" and param is None:
            raise click.UsageError("'ci' requires a parameter, e.g. 'ci:30'")
        terms.append((weight, name, param))
    if not terms:
        raise click.UsageError(f"empty objective spec: {spec!r}")
    return terms


def build_objective_callable(
    optimizer: Any,
    terms: list[tuple[float, str, str | None]],
    interpolant: Any = None,
) -> Union[Callable[..., Any], str]:
    """Build an objective callable (or a built-in string) from parsed terms.

    If the spec is a single bare term with weight 1.0 and no grid cost, we
    return the string form so ``trajectory(objective=...)`` can route it
    through its built-in dispatch (the Python API accepts "fuel", "time",
    "ci:30", "gwp100", ... directly).

    Otherwise we return a callable that sums the weighted terms.
    """
    if len(terms) == 1 and terms[0][0] == 1.0 and terms[0][1] != "grid":
        _, name, param = terms[0]
        return name if param is None else f"{name}:{param}"

    # Composite or grid-containing objective — return a callable.
    uses_grid = any(t[1] == "grid" for t in terms)
    if uses_grid and interpolant is None:
        raise click.UsageError(
            "objective contains 'grid' but --grid FILE was not given"
        )

    # Auto-detect grid dimensionality
    if interpolant is not None:
        n_dim = int(interpolant.numel_in(0))
        time_dependent = n_dim == 4
    else:
        n_dim = 3
        time_dependent = False

    def objective(x, u, dt, **kwargs):
        kw = {
            k: v
            for k, v in kwargs.items()
            if k not in ("time_dependent", "n_dim", "interpolant")
        }
        total = 0
        for weight, name, param in terms:
            if name == "fuel":
                term = optimizer.obj_fuel(x, u, dt, **kw)
            elif name == "grid":
                term = optimizer.obj_grid_cost(
                    x,
                    u,
                    dt,
                    interpolant=interpolant,
                    time_dependent=time_dependent,
                    n_dim=n_dim,
                    **kw,
                )
            elif name == "time":
                term = optimizer.obj_time(x, u, dt, **kw)
            elif name == "ci":
                assert param is not None, "ci term must have a parameter"
                term = optimizer.obj_ci(x, u, dt, ci=float(param), **kw)
            elif name in {"gwp20", "gwp50", "gwp100", "gtp20", "gtp50", "gtp100"}:
                term = optimizer._obj_climate(x, u, dt, name, **kw)
            else:
                raise RuntimeError(f"unreachable term: {name}")
            total = total + weight * term
        return total

    return objective


# ============================================================
# Grid file handling
# ============================================================


def load_grid_file(path: Path) -> Any:
    """Load a grid file.

    ``.casadi`` files are loaded directly as pre-built interpolants.
    Other formats are read as parquet and built into a bspline interpolant
    on the fly (slow — we warn the user).
    """
    if path.suffix == ".casadi":
        return tools.load_interpolant(path)

    click.echo(
        f"WARN: {path} is not a .casadi cache. Building bspline from parquet "
        f"(this may take several minutes on a large grid).\n"
        f"      To build once and reuse, run:\n"
        f"      opentop gengrid --in {path} --out {path.with_suffix('.casadi')}",
        err=True,
    )
    df = pd.read_parquet(path)
    return tools.interpolant_from_dataframe(df, shape="bspline")


# ============================================================
# gengrid: bbox + time slicing + altitude padding
# ============================================================


def _parse_bbox(bbox: str) -> tuple[float, float, float, float]:
    """Parse a '--bbox LAT_MIN:LAT_MAX,LON_MIN:LON_MAX' argument."""
    try:
        lat_part, lon_part = bbox.split(",")
        lat_min, lat_max = (float(x) for x in lat_part.split(":"))
        lon_min, lon_max = (float(x) for x in lon_part.split(":"))
    except ValueError as e:
        raise click.UsageError(
            f"invalid --bbox {bbox!r}, expected LAT_MIN:LAT_MAX,LON_MIN:LON_MAX"
        ) from e
    return lat_min, lat_max, lon_min, lon_max


def _parse_time_window(time_arg: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Parse a '--time START,STOP' argument into tz-aware timestamps."""
    try:
        t0_str, t1_str = time_arg.split(",")
    except ValueError as e:
        raise click.UsageError(
            f"invalid --time {time_arg!r}, expected START,STOP"
        ) from e
    t0 = pd.Timestamp(t0_str.strip())
    t1 = pd.Timestamp(t1_str.strip())
    if t0.tzinfo is None:
        t0 = t0.tz_localize("UTC")
    if t1.tzinfo is None:
        t1 = t1.tz_localize("UTC")
    return t0, t1  # type: ignore[return-value]  # pd.Timestamp() stubs widen to NaT


# Target altitudes (ft) used by _pad_altitudes to extend grid coverage from
# 0 to FL480 in 2000 ft steps.  Private to this module.
_PAD_ALTITUDES_FT: list[float] = [float(a) for a in range(0, 49_000, 2000)]


def _pad_altitudes(df: pd.DataFrame) -> pd.DataFrame:
    """Add zero-cost rows at altitudes outside the existing grid band.

    Extends coverage from 0 to 48000 ft in 2000 ft steps. Original data in
    the contrail band (20000-44000 ft typically) is preserved; rows outside
    are created with ``cost=0``. Physically correct: persistent contrails
    cannot form below ~FL200 or above ~FL440.
    """
    from openap.aero import ft

    existing = set(df.altitude.unique())
    missing = [a for a in _PAD_ALTITUDES_FT if a not in existing]
    if not missing:
        return df

    grouping_cols = [
        c for c in ("timestamp", "ts", "latitude", "longitude") if c in df.columns
    ]
    base = df[grouping_cols].drop_duplicates()

    pads = []
    for alt_ft in missing:
        p = base.copy()
        p["altitude"] = float(alt_ft)
        p["height"] = float(alt_ft) * ft
        p["cost"] = 0.0
        for col in df.columns:
            if col not in p.columns:
                p[col] = 0
        pads.append(p[df.columns])

    return pd.concat([df, *pads], ignore_index=True)


# ============================================================
# Click commands
# ============================================================


@click.group()
@click.version_option(package_name="opentop", prog_name="opentop")
def main() -> None:
    """opentop: aircraft trajectory optimization CLI."""


@main.command()
@click.argument("origin")
@click.argument("destination")
@click.option("-a", "--aircraft", required=True, help="Aircraft type, e.g. A320.")
@click.option(
    "--m0",
    type=float,
    default=0.85,
    show_default=True,
    help="Initial mass ratio (fraction of MTOW).",
)
@click.option(
    "--phase",
    type=click.Choice(["all", "cruise", "climb", "descent"]),
    default="all",
    show_default=True,
    help="Flight phase (all = CompleteFlight).",
)
@click.option(
    "--obj",
    "objective",
    default="fuel",
    show_default=True,
    help='Objective expression, e.g. "fuel", "ci:30", "0.3*fuel+0.7*grid".',
)
@click.option(
    "--grid",
    "grid_path",
    type=click.Path(exists=True, path_type=Path),
    help="Grid cost file (.casadi preferred, .parquet also accepted).",
)
@click.option(
    "--max-iter",
    type=int,
    default=1500,
    show_default=True,
    help="Maximum IPOPT iterations.",
)
@click.option(
    "-o",
    "--output",
    type=click.Path(path_type=Path),
    help="Save trajectory to this parquet file (stdout summary only if omitted).",
)
@click.option("-v", "--debug", is_flag=True, help="Verbose IPOPT output.")
def optimize(
    origin: str,
    destination: str,
    aircraft: str,
    m0: float,
    phase: str,
    objective: str,
    grid_path: Path | None,
    max_iter: int,
    output: Path | None,
    debug: bool,
) -> None:
    """Run trajectory optimization for ORIGIN → DESTINATION."""
    phase_map = {
        "all": top.CompleteFlight,
        "cruise": top.Cruise,
        "climb": top.Climb,
        "descent": top.Descent,
    }
    phase_cls = phase_map[phase]

    interpolant = load_grid_file(grid_path) if grid_path else None
    terms = parse_objective(objective)

    click.echo(f"  aircraft:  {aircraft}")
    click.echo(f"  route:     {origin} → {destination}")
    click.echo(f"  phase:     {phase}")
    click.echo(f"  objective: {objective}")
    click.echo(f"  m0:        {m0}")
    click.echo(f"  max_iter:  {max_iter}")
    if grid_path:
        click.echo(f"  grid file: {grid_path}")

    opt = phase_cls(aircraft, origin, destination, m0)
    opt.setup(debug=debug, max_iter=max_iter)

    obj_arg = build_objective_callable(opt, terms, interpolant=interpolant)

    traj_kwargs: dict = {
        "objective": obj_arg,
        "return_failed": True,
    }
    if interpolant is not None:
        traj_kwargs["interpolant"] = interpolant

    t0 = time.time()
    df = opt.trajectory(**traj_kwargs)
    wall = time.time() - t0

    stats = opt.stats
    click.echo("")
    click.echo(f"  success:       {stats.get('success')}")
    click.echo(f"  return_status: {stats.get('return_status')}")
    click.echo(f"  iter_count:    {stats.get('iter_count')}")
    click.echo(f"  wall time:     {wall:.1f} s")
    click.echo(f"  objective:     {opt.objective_value:.4e}")

    if df is not None and not df.empty:
        fuel = df.mass.iloc[0] - df.mass.iloc[-1]
        click.echo(f"  fuel burn:     {fuel:.1f} kg")
        click.echo(f"  max altitude:  {df.altitude.max():.0f} ft")
        click.echo(f"  flight time:   {df.ts.iloc[-1] / 60:.1f} min")

    if output is not None and df is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output)
        click.echo(f"\n  trajectory saved to {output}")

    if not stats.get("success", False):
        sys.exit(1)


@main.command()
@click.option(
    "--in",
    "input_path",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Input grid file (parquet).",
)
@click.option(
    "--out",
    "output_path",
    required=True,
    type=click.Path(path_type=Path),
    help="Output .casadi cache file.",
)
@click.option("--bbox", help="Spatial bbox as LAT_MIN:LAT_MAX,LON_MIN:LON_MAX.")
@click.option(
    "--time",
    "time_window",
    help="Time window as START,STOP (ISO-format timestamps, UTC assumed).",
)
@click.option(
    "--pad-altitudes/--no-pad-altitudes",
    default=True,
    show_default=True,
    help="Extend altitude coverage with zero-cost rows from 0 to FL480.",
)
@click.option(
    "--shape",
    type=click.Choice(["linear", "bspline"]),
    default="bspline",
    show_default=True,
    help="Interpolant type. 'bspline' has smooth gradients and is strongly "
    "preferred for CompleteFlight solves.",
)
def gengrid(
    input_path: Path,
    output_path: Path,
    bbox: str | None,
    time_window: str | None,
    pad_altitudes: bool,
    shape: str,
) -> None:
    """Build and cache a CasADi interpolant from a raw cost grid parquet."""
    click.echo(f"Loading {input_path}")
    df = pd.read_parquet(input_path)
    click.echo(f"  raw shape: {df.shape}")

    if time_window is not None:
        t0, t1 = _parse_time_window(time_window)
        df = cast(pd.DataFrame, df[(df.timestamp >= t0) & (df.timestamp <= t1)].copy())
        if "ts" in df.columns:
            df["ts"] = df.ts - df.ts.min()
        click.echo(f"  after time slice: {df.shape}")

    if bbox is not None:
        lat_min, lat_max, lon_min, lon_max = _parse_bbox(bbox)
        df = cast(
            pd.DataFrame,
            df[
                (df.latitude >= lat_min)
                & (df.latitude <= lat_max)
                & (df.longitude >= lon_min)
                & (df.longitude <= lon_max)
            ].copy(),
        )
        click.echo(f"  after bbox slice: {df.shape}")

    if pad_altitudes:
        before = df.shape[0]
        df = _pad_altitudes(df)
        click.echo(
            f"  after altitude padding: {df.shape} (+{df.shape[0] - before} rows)"
        )

    click.echo(f"\nBuilding {shape} interpolant ({df.shape[0]:,} points)...")
    t0 = time.time()
    interpolant = tools.interpolant_from_dataframe(df, shape=shape)
    click.echo(f"  build time: {time.time() - t0:.1f} s")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tools.save_interpolant(interpolant, output_path)
    size_mb = output_path.stat().st_size / 1024 / 1024
    click.echo(f"\n  saved to {output_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
