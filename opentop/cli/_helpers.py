"""Shared helpers for opentop CLI subcommands."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Callable, Union

import click

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
