"""Structured options for trajectory optimization and result packaging.

These dataclasses replace the **kwargs dict plumbing in Base.trajectory.
Task 22 (next) rewrites Base.trajectory to consume these.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import casadi as ca
import pandas as pd


@dataclass(frozen=True, slots=True)
class SolveOptions:
    """Options consumed by Base._run / Base._build_opti / Base._solve."""

    max_iter: int = 1500
    max_fuel: Optional[float] = None
    auto_rescale_objective: bool = True
    return_failed: bool = False
    initial_guess: Optional[pd.DataFrame] = None
    alt_start: Optional[float] = None
    alt_stop: Optional[float] = None
    remove_cruise: bool = False
    exact_hessian: bool = False


@dataclass(frozen=True, slots=True)
class GridOptions:
    """Options for grid-cost objectives."""

    interpolant: Optional[ca.Function] = None
    n_dim: int = 3
    time_dependent: bool = False


@dataclass(frozen=True, slots=True)
class TrajectoryResult:
    """Rich result of a trajectory optimization. Returned when trajectory(result_object=True)."""

    df: pd.DataFrame
    success: bool
    status: str
    objective: float
    iters: int
    fuel: float
    grid_cost: float  # NaN if no interpolant
    stats: dict[str, Any]


def build_result(df, stats: dict, objective: float) -> TrajectoryResult:
    """Package a DataFrame + solver stats + objective into a TrajectoryResult.

    Used by Base._make_result after a trajectory() call. ``df`` may be None
    (from a rejected solve); it is coerced to an empty DataFrame in the result.
    """
    has_df = df is not None and len(df) > 0
    return TrajectoryResult(
        df=df if df is not None else pd.DataFrame(),
        success=bool(stats.get("success", False)),
        status=str(stats.get("return_status", "")),
        objective=objective,
        iters=int(stats.get("iter_count", 0)),
        fuel=(
            float(df["mass"].iloc[0] - df["mass"].iloc[-1])
            if has_df and "mass" in df.columns
            else float("nan")
        ),
        grid_cost=(
            float(df["grid_cost"].sum(skipna=True))
            if has_df and "grid_cost" in df.columns and df["grid_cost"].notna().any()
            else float("nan")
        ),
        stats=stats,
    )
