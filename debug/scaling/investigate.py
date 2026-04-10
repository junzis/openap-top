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

# --- Pilot case (see spec §"Pilot case") ---
AIRCRAFT = "A320"
ORIGIN = "EHAM"
DESTINATION = "LGAV"
M0 = 0.85
COEF = 0.5

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
CONFIGS = ("default", "none", "obj_rescaled")


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


def main() -> int:
    raise NotImplementedError("filled in by Task 7")


if __name__ == "__main__":
    sys.exit(main())
