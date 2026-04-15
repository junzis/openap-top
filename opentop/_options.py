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
