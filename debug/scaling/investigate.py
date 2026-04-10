"""NLP scaling investigation driver.

See docs/superpowers/specs/2026-04-10-nlp-scaling-investigation-design.md
for full context.

Runs EHAM-LGAV A320 contrail+CO2 pilot case across three IPOPT scaling
configs, for both top.Cruise and top.CompleteFlight. Captures solver stats
+ parsed IPOPT log + solution quality cross-check. Writes a markdown report
to debug/scaling/investigation/.
"""

from __future__ import annotations

import re
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import openap  # noqa: E402

_openap_local = REPO_ROOT / "openap"
if hasattr(openap, "__path__"):
    openap.__path__.insert(0, str(_openap_local))

from openap import top  # noqa: E402

# --- Test matrix (see spec §"Pilot case") ---
AIRCRAFT = "A320"
M0 = 0.85
COEF = 0.5

# Routes: short, medium, long. Same as tests/compare_nlp_scaling.py.
ROUTES = [
    ("EHAM", "EDDF"),  # Amsterdam - Frankfurt (~370 km, short/easy)
    ("EGLL", "LEMD"),  # London - Madrid (~1260 km, medium)
    ("EHAM", "LGAV"),  # Amsterdam - Athens (~2100 km, long/hard)
]

# Grid slice window (see spec §"Data")
GRID_T0 = "2022-02-20 10:00:00+00:00"
GRID_T1 = "2022-02-20 14:00:00+00:00"
BBOX_PADDING_DEG = 5.0

# Data source
DATA_PATH = Path(
    "/home/junzi/arc/code/1-public/contrail-or-not/data/grid_era5_smoothed.parquet.gz"
)
FIGSHARE_URL = "https://ndownloader.figshare.com/files/55632059"

# Output
OUTPUT_DIR = Path(__file__).parent / "investigation"

# Solver configs
#   default           : IPOPT gradient-based scaling (default behaviour)
#   none              : no IPOPT scaling
#   obj_rescaled      : gradient-based + Python-side objective / f0
#   var_scaled        : gradient-based + opti.set_linear_scale per variable
#   var+obj_rescaled  : combine var_scaled and obj_rescaled
CONFIGS = ("default", "none", "obj_rescaled", "var_scaled", "var+obj_rescaled")


# ============================================================
# Grid loading
# ============================================================


def load_grid_parquet(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the ERA5 smoothed cost grid from parquet.

    If the file is missing, print the figshare download URL and exit.
    """
    if not path.exists():
        print(f"ERROR: grid file not found at {path}", file=sys.stderr)
        print(f"Download from: {FIGSHARE_URL}", file=sys.stderr)
        print(
            "This is the pre-smoothed ERA5 cost grid from the contrail-or-not "
            "paper (Sun et al., figshare 10.6084/m9.figshare.29400650).",
            file=sys.stderr,
        )
        sys.exit(1)
    return pd.read_parquet(path)


def slice_grid(
    df: pd.DataFrame,
    t0: str,
    t1: str,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
) -> pd.DataFrame:
    """Slice a cost grid dataframe to a bounding box + time window.

    Rebases the ``ts`` column so ``ts=0`` corresponds to ``t0``, matching
    the optimizer's internal time axis (which starts at 0 at the origin).

    Args:
        df: Full grid dataframe with columns at least:
            ``timestamp`` (tz-aware), ``ts``, ``height``, ``latitude``,
            ``longitude``, ``cost``.
        t0, t1: ISO-format UTC timestamps bounding the time window.
        lat_min, lat_max: Latitude bounds (inclusive).
        lon_min, lon_max: Longitude bounds (inclusive).

    Returns:
        A filtered dataframe with ``ts`` rebased to start at 0.
    """
    t0_ts = pd.Timestamp(t0)
    t1_ts = pd.Timestamp(t1)
    mask = (
        (df["timestamp"] >= t0_ts)
        & (df["timestamp"] <= t1_ts)
        & (df["latitude"] >= lat_min)
        & (df["latitude"] <= lat_max)
        & (df["longitude"] >= lon_min)
        & (df["longitude"] <= lon_max)
    )
    sliced = df.loc[mask].copy()
    if sliced.empty:
        raise ValueError(
            f"Grid slice is empty for window {t0}..{t1}, "
            f"bbox lat=[{lat_min},{lat_max}] lon=[{lon_min},{lon_max}]"
        )
    start_seconds = sliced["ts"].min()
    sliced["ts"] = sliced["ts"] - start_seconds
    return sliced.sort_values(["ts", "height", "latitude", "longitude"])


# ============================================================
# IPOPT log parser
# ============================================================

_RE_SCALING_ENTRY = re.compile(
    r"^\s*(?P<name>x|c) scaling vector\[\s*\d+\s*\]=\s*(?P<val>[-+\d\.eE]+)\s*$"
)
_RE_RESTORATION = re.compile(
    r"Restoration phase is called at iteration\s+(?P<iter>\d+)"
)
_RE_OBJ_LINE = re.compile(
    r"^Objective\.+:\s+(?P<scaled>[-+\d\.eE]+)\s+(?P<unscaled>[-+\d\.eE]+)"
)
_RE_NLP_ERR_LINE = re.compile(
    r"^Overall NLP error\.+:\s+(?P<scaled>[-+\d\.eE]+)\s+(?P<unscaled>[-+\d\.eE]+)"
)
_RE_CONSTR_LINE = re.compile(
    r"^Constraint violation\.+:\s+(?P<scaled>[-+\d\.eE]+)\s+(?P<unscaled>[-+\d\.eE]+)"
)
_RE_EXIT = re.compile(r"^EXIT:\s*(?P<msg>.+?)\s*$")


def parse_ipopt_log(path: Path) -> dict:
    """Parse an IPOPT output_file dump.

    Returns a dict with the following keys (missing values are omitted):
        x_scaling_count, x_scaling_min, x_scaling_max, x_scaling_median
        c_scaling_count, c_scaling_min, c_scaling_max, c_scaling_median
        restoration_count, restoration_iterations (list[int])
        final_scaled_nlp_error, final_unscaled_nlp_error
        final_constraint_violation
        final_scaled_objective, final_unscaled_objective
        exit_status
    """
    if not path.exists():
        return {}

    x_vals: list[float] = []
    c_vals: list[float] = []
    restoration_iters: list[int] = []
    out: dict = {}

    with path.open() as f:
        for line in f:
            m = _RE_SCALING_ENTRY.match(line)
            if m:
                val = float(m.group("val"))
                if m.group("name") == "x":
                    x_vals.append(val)
                else:
                    c_vals.append(val)
                continue

            m = _RE_RESTORATION.search(line)
            if m:
                restoration_iters.append(int(m.group("iter")))
                continue

            m = _RE_OBJ_LINE.match(line)
            if m:
                out["final_scaled_objective"] = float(m.group("scaled"))
                out["final_unscaled_objective"] = float(m.group("unscaled"))
                continue

            m = _RE_NLP_ERR_LINE.match(line)
            if m:
                out["final_scaled_nlp_error"] = float(m.group("scaled"))
                out["final_unscaled_nlp_error"] = float(m.group("unscaled"))
                continue

            m = _RE_CONSTR_LINE.match(line)
            if m:
                out["final_constraint_violation"] = float(m.group("unscaled"))
                continue

            m = _RE_EXIT.match(line)
            if m:
                out["exit_status"] = m.group("msg")
                continue

    if x_vals:
        arr = np.array(x_vals)
        out["x_scaling_count"] = len(arr)
        out["x_scaling_min"] = float(arr.min())
        out["x_scaling_max"] = float(arr.max())
        out["x_scaling_median"] = float(np.median(arr))

    if c_vals:
        arr = np.array(c_vals)
        out["c_scaling_count"] = len(arr)
        out["c_scaling_min"] = float(arr.min())
        out["c_scaling_max"] = float(arr.max())
        out["c_scaling_median"] = float(np.median(arr))

    out["restoration_count"] = len(restoration_iters)
    out["restoration_iterations"] = restoration_iters

    return out


# ============================================================
# Student objective wrapper (see spec §"Configs to compare")
# ============================================================


def make_student_objective(optimizer, interpolant, coef: float, rescale: float = 1.0):
    """Build the student's blended contrail+CO2 objective.

    Form: ``(grid_cost * coef + fuel_cost * (1 - coef)) / rescale``
    with ``n_dim=4, time_dependent=True``.

    Args:
        optimizer: A top.Cruise / top.CompleteFlight instance. Must have
            ``obj_grid_cost`` and ``obj_fuel`` methods.
        interpolant: CasADi interpolant returned by
            ``top.tools.interpolant_from_dataframe``.
        coef: Blend coefficient in [0, 1]. ``coef=0`` is pure fuel,
            ``coef=1`` is pure grid cost.
        rescale: Divisor applied to the whole blended objective. Used by
            the ``obj_rescaled`` config to normalize so ``f(x0) ≈ 1``.

    Returns:
        A callable ``(x, u, dt, **kwargs) -> expr`` suitable for
        ``optimizer.trajectory(objective=...)``.
    """

    def objective(x, u, dt, **kwargs):
        kw = {
            k: v
            for k, v in kwargs.items()
            if k not in ("time_dependent", "n_dim", "interpolant")
        }
        grid_cost = optimizer.obj_grid_cost(
            x, u, dt, interpolant=interpolant, time_dependent=True, n_dim=4, **kw
        )
        fuel_cost = optimizer.obj_fuel(x, u, dt, **kw)
        return (grid_cost * coef + fuel_cost * (1 - coef)) / rescale

    return objective


def evaluate_blended_on_trajectory(df_traj: pd.DataFrame, coef: float) -> float:
    """Evaluate the student's blend numerically on a returned trajectory.

    Uses the ``grid_cost`` and ``fuel_cost`` columns that ``to_trajectory``
    writes post-solve. Matches the symbolic objective to 4-5 decimal
    places in practice.
    """
    grid_sum = float(df_traj["grid_cost"].sum())
    fuel_sum = float(df_traj["fuel_cost"].sum())
    return grid_sum * coef + fuel_sum * (1 - coef)


# ============================================================
# Solve runner
# ============================================================


@dataclass
class SolveResult:
    route: str  # e.g. "EHAM-LGAV"
    phase: str
    config: str
    success: bool
    return_status: str
    iter_count: int
    wall_time_s: float
    final_objective: float  # from optimizer.objective_value (scaled by rescale)
    blended_physical: float  # re-evaluated from trajectory columns
    grid_sum: float
    fuel_sum: float
    traj: pd.DataFrame
    iterations: dict  # per-iter arrays from solver.stats()
    parsed_log: dict  # from parse_ipopt_log
    log_path: Path


def _configure_solver(optimizer, config: str, log_path: Path) -> None:
    """Apply the given scaling config via ipopt_kwargs."""
    ipopt_kwargs = {
        "output_file": str(log_path),
        "file_print_level": 5,
        "print_info_string": "yes",
    }
    if config == "none":
        ipopt_kwargs["nlp_scaling_method"] = "none"
    else:
        ipopt_kwargs["nlp_scaling_method"] = "gradient-based"

    optimizer.setup(
        debug=False,
        max_iter=3000,
        tol=1e-6,
        ipopt_kwargs=ipopt_kwargs,
    )


def _extract_iterations(stats: dict) -> dict:
    """Safely extract per-iteration arrays from solver.stats()."""
    it = stats.get("iterations", {}) or {}
    keys = ("obj", "inf_pr", "inf_du", "mu", "d_norm", "alpha_pr", "alpha_du")
    out: dict = {}
    for k in keys:
        v = it.get(k)
        if v is not None:
            out[k] = list(v)
    return out


def run_one(
    phase_cls,
    phase_name: str,
    origin: str,
    destination: str,
    config: str,
    interpolant,
    warmstart_df: pd.DataFrame,
    f0: float,
    log_path: Path,
) -> SolveResult:
    """Run one (phase, config) solve and capture diagnostics.

    Args:
        phase_cls: top.Cruise or top.CompleteFlight
        phase_name: "Cruise" or "CompleteFlight"
        origin: ICAO airport code
        destination: ICAO airport code
        config: one of CONFIGS
        interpolant: CasADi interpolant for grid cost
        warmstart_df: fuel-only optimal trajectory used as initial guess
        f0: blended objective evaluated on the warmstart, for obj_rescaled
        log_path: where to write the IPOPT output file
    """
    optimizer = phase_cls(AIRCRAFT, origin, destination, M0)
    _configure_solver(optimizer, config, log_path)

    rescale = f0 if config in ("obj_rescaled", "var+obj_rescaled") else 1.0
    scale_variables = config in ("var_scaled", "var+obj_rescaled")
    objective = make_student_objective(
        optimizer, interpolant, coef=COEF, rescale=rescale
    )

    t0 = time.time()
    try:
        df_traj = optimizer.trajectory(
            objective=objective,
            interpolant=interpolant,
            initial_guess=warmstart_df,
            time_dependent=True,
            n_dim=4,
            return_failed=True,
            scale_variables=scale_variables,
        )
    except Exception as exc:  # pragma: no cover — defensive
        print(f"  [{phase_name} / {config}] solve raised: {exc}", file=sys.stderr)
        raise
    wall = time.time() - t0

    stats = optimizer.solver.stats()
    parsed = parse_ipopt_log(log_path)
    blended_physical = evaluate_blended_on_trajectory(df_traj, coef=COEF)

    return SolveResult(
        route=f"{origin}-{destination}",
        phase=phase_name,
        config=config,
        success=bool(stats.get("success", False)),
        return_status=str(stats.get("return_status", "UNKNOWN")),
        iter_count=int(stats.get("iter_count", -1)),
        wall_time_s=wall,
        final_objective=float(optimizer.objective_value),
        blended_physical=blended_physical,
        grid_sum=float(df_traj["grid_cost"].sum()),
        fuel_sum=float(df_traj["fuel_cost"].sum()),
        traj=df_traj,
        iterations=_extract_iterations(stats),
        parsed_log=parsed,
        log_path=log_path,
    )


def run_fuel_warmstart(
    phase_cls, phase_name: str, origin: str, destination: str
) -> pd.DataFrame:
    """Run the fuel-only solve used as warmstart for the contrail+CO2 configs.

    Uses IPOPT defaults (gradient-based), no file logging.
    """
    optimizer = phase_cls(AIRCRAFT, origin, destination, M0)
    optimizer.setup(debug=False, max_iter=3000)
    df = optimizer.trajectory(objective="fuel")
    if df is None or df.empty:
        raise RuntimeError(f"Fuel-only warmstart failed for {phase_name}")
    return df


# ============================================================
# Report writer
# ============================================================


@dataclass
class InvestigationRun:
    timestamp: datetime
    git_commit: str
    hostname: str
    data_path: Path
    data_size_bytes: int
    data_mtime: datetime
    grid_t0: str
    grid_t1: str
    routes: list[tuple[str, str]]
    coef: float
    results: list[SolveResult] = field(default_factory=list)

    @property
    def output_dir(self) -> Path:
        stamp = self.timestamp.strftime("%Y-%m-%d-%H-%M")
        d = OUTPUT_DIR / stamp
        d.mkdir(parents=True, exist_ok=True)
        return d

    def log_path(self, route: str, phase: str, config: str) -> Path:
        safe_config = config.replace("+", "_plus_")
        return self.output_dir / f"ipopt_{route}_{phase}_{safe_config}.log"

    def csv_path(self, route: str, phase: str, config: str) -> Path:
        safe_config = config.replace("+", "_plus_")
        return self.output_dir / f"{route}_{phase}_{safe_config}_iters.csv"

    def report_path(self) -> Path:
        return self.output_dir / "report.md"


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def _write_iter_csv(result: SolveResult, path: Path) -> None:
    if not result.iterations:
        return
    df = pd.DataFrame(result.iterations)
    df.index.name = "iter"
    df.to_csv(path)


def write_report(run: InvestigationRun) -> Path:
    """Write the markdown report + sidecar iter CSVs. Returns the report path."""
    lines: list[str] = []

    lines.append(f"# NLP scaling investigation — {run.timestamp.isoformat()}")
    lines.append("")
    lines.append("## Run metadata")
    lines.append("")
    lines.append(f"- Git commit: `{run.git_commit}`")
    lines.append(f"- Host: `{run.hostname}`")
    lines.append(f"- Aircraft: {AIRCRAFT}, M0={M0}, coef={run.coef}")
    routes_str = ", ".join(f"{o}-{d}" for o, d in run.routes)
    lines.append(f"- Routes: {routes_str}")
    lines.append(
        f"- Data: `{run.data_path}` "
        f"({run.data_size_bytes / 1024 / 1024:.1f} MB, mtime {run.data_mtime.isoformat()})"
    )
    lines.append(f"- Grid window: {run.grid_t0} .. {run.grid_t1}")
    lines.append("")

    phases = ["Cruise", "CompleteFlight"]
    for phase in phases:
        phase_results = [r for r in run.results if r.phase == phase]
        if not phase_results:
            continue
        lines.append(f"## Phase: {phase}")
        lines.append("")

        # One section per route
        routes_in_phase = []
        for r in phase_results:
            if r.route not in routes_in_phase:
                routes_in_phase.append(r.route)

        for route in routes_in_phase:
            route_results = [r for r in phase_results if r.route == route]
            lines.append(f"### Route: {route}")
            lines.append("")

            # --- Summary table ---
            lines.append(
                "| config | success | iter | wall (s) | "
                "blended (phys) | grid sum | fuel sum |"
            )
            lines.append("|---|---|---|---|---|---|---|")
            for r in route_results:
                lines.append(
                    f"| {r.config} | {r.success} | {r.iter_count} "
                    f"| {r.wall_time_s:.1f} | {r.blended_physical:.4e} "
                    f"| {r.grid_sum:.4e} | {r.fuel_sum:.4e} |"
                )
            lines.append("")

            # --- Iteration traces (CSV sidecar pointers) ---
            for r in route_results:
                _write_iter_csv(r, run.csv_path(r.route, r.phase, r.config))
                lines.append(
                    f"- `{r.config}`: {r.iter_count} iters, "
                    f"final obj = {r.final_objective:.4e}, "
                    f"return_status = `{r.return_status}`, "
                    f"CSV → `{run.csv_path(r.route, r.phase, r.config).name}`"
                )
            lines.append("")

            # --- Solution quality cross-check (only vs 'default' as anchor) ---
            default_result = next(
                (r for r in route_results if r.config == "default"), None
            )
            if default_result is not None and default_result.success:
                lines.append("Cross-check vs `default`:")
                lines.append("")
                lines.append(
                    "| config | max alt diff (ft) | max mass diff (kg) | "
                    "max lat diff | max lon diff | blended gap |"
                )
                lines.append("|---|---|---|---|---|---|")
                for r in route_results:
                    if r.config == "default" or not r.success:
                        continue
                    cmp = _compare_trajectories(default_result.traj, r.traj)
                    a, b = default_result.blended_physical, r.blended_physical
                    gap = abs(a - b) / max(abs(a), abs(b), 1e-15)
                    label = "CLEAN" if gap < 0.01 else "DIVERGENT"
                    lines.append(
                        f"| {r.config} "
                        f"| {cmp['alt']:.0f} | {cmp['mass']:.1f} "
                        f"| {cmp['lat']:.4f} | {cmp['lon']:.4f} "
                        f"| {gap * 100:.2f}% ({label}) |"
                    )
                lines.append("")

    # --- Findings placeholder for human to fill in ---
    lines.append("## Findings")
    lines.append("")
    lines.append("_(write a 5-10 line summary after reading the tables above)_")
    lines.append("")

    report_path = run.report_path()
    report_path.write_text("\n".join(lines))
    return report_path


def _compare_trajectories(a: pd.DataFrame, b: pd.DataFrame) -> dict:
    """Compute max pointwise diffs between two trajectories.

    If lengths differ, resamples both onto a shared time axis (linear interp
    against ``ts``). Columns compared: altitude, mass, latitude, longitude.
    """
    if len(a) != len(b):
        ts_common = np.linspace(
            max(a["ts"].iloc[0], b["ts"].iloc[0]),
            min(a["ts"].iloc[-1], b["ts"].iloc[-1]),
            num=min(len(a), len(b)),
        )

        def _interp(df, col):
            return np.interp(ts_common, df["ts"].values, df[col].values)

        return {
            "alt": float(np.abs(_interp(a, "altitude") - _interp(b, "altitude")).max()),
            "mass": float(np.abs(_interp(a, "mass") - _interp(b, "mass")).max()),
            "lat": float(np.abs(_interp(a, "latitude") - _interp(b, "latitude")).max()),
            "lon": float(np.abs(_interp(a, "longitude") - _interp(b, "longitude")).max()),
        }
    return {
        "alt": float(np.abs(a["altitude"].values - b["altitude"].values).max()),
        "mass": float(np.abs(a["mass"].values - b["mass"].values).max()),
        "lat": float(np.abs(a["latitude"].values - b["latitude"].values).max()),
        "lon": float(np.abs(a["longitude"].values - b["longitude"].values).max()),
    }


def _route_bbox(origin: str, destination: str) -> tuple[float, float, float, float]:
    """Return (lat_min, lat_max, lon_min, lon_max) bbox padded around the route."""
    tmp = top.Cruise(AIRCRAFT, origin, destination, M0)
    lat1, lon1 = tmp.lat1, tmp.lon1
    lat2, lon2 = tmp.lat2, tmp.lon2
    del tmp
    return (
        min(lat1, lat2) - BBOX_PADDING_DEG,
        max(lat1, lat2) + BBOX_PADDING_DEG,
        min(lon1, lon2) - BBOX_PADDING_DEG,
        max(lon1, lon2) + BBOX_PADDING_DEG,
    )


def main() -> int:
    print("=" * 70)
    print(f"NLP scaling investigation — {AIRCRAFT}, {len(ROUTES)} routes")
    print("=" * 70)

    # --- Load grid once ---
    print(f"\nLoading grid from {DATA_PATH}")
    df_full = load_grid_parquet(DATA_PATH)
    print(f"  full shape: {df_full.shape}")

    # --- Set up run metadata ---
    stat = DATA_PATH.stat()
    run = InvestigationRun(
        timestamp=datetime.now(),
        git_commit=_git_commit(),
        hostname=socket.gethostname(),
        data_path=DATA_PATH,
        data_size_bytes=stat.st_size,
        data_mtime=datetime.fromtimestamp(stat.st_mtime),
        grid_t0=GRID_T0,
        grid_t1=GRID_T1,
        routes=list(ROUTES),
        coef=COEF,
    )
    print(f"\nOutput dir: {run.output_dir}")

    # --- Loop over routes ---
    for origin, destination in ROUTES:
        route_label = f"{origin}-{destination}"
        print(f"\n{'=' * 70}")
        print(f"  Route: {route_label}")
        print(f"{'=' * 70}")

        lat_min, lat_max, lon_min, lon_max = _route_bbox(origin, destination)
        print(
            f"  bbox: lat=[{lat_min:.2f}, {lat_max:.2f}], "
            f"lon=[{lon_min:.2f}, {lon_max:.2f}]"
        )
        df_slice = slice_grid(
            df_full,
            t0=GRID_T0,
            t1=GRID_T1,
            lat_min=lat_min,
            lat_max=lat_max,
            lon_min=lon_min,
            lon_max=lon_max,
        )
        print(f"  slice shape: {df_slice.shape}")
        interpolant = top.tools.interpolant_from_dataframe(df_slice)

        # --- 2 phases x 5 configs per route ---
        for phase_cls, phase_name in [
            (top.Cruise, "Cruise"),
            (top.CompleteFlight, "CompleteFlight"),
        ]:
            print(f"\n--- {route_label} / {phase_name} ---")
            print(f"  [{phase_name}] fuel warmstart ...", flush=True)
            warmstart = run_fuel_warmstart(
                phase_cls, phase_name, origin, destination
            )
            f0 = evaluate_blended_on_trajectory(warmstart, coef=COEF)
            print(f"  [{phase_name}] f0 = {f0:.4e}")

            for config in CONFIGS:
                log_path = run.log_path(route_label, phase_name, config)
                print(f"  [{phase_name} / {config}] solving ...", flush=True)
                result = run_one(
                    phase_cls=phase_cls,
                    phase_name=phase_name,
                    origin=origin,
                    destination=destination,
                    config=config,
                    interpolant=interpolant,
                    warmstart_df=warmstart,
                    f0=f0,
                    log_path=log_path,
                )
                print(
                    f"    done: success={result.success} "
                    f"iter={result.iter_count} "
                    f"wall={result.wall_time_s:.1f}s "
                    f"blended={result.blended_physical:.4e}"
                )
                run.results.append(result)

    # --- Write report ---
    report_path = write_report(run)
    print(f"\n✓ Report written to {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
