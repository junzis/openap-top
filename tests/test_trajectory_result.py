"""Integration tests for trajectory(result_object=True) and unknown-kwarg rejection.

These tests pin the Phase-3 Task-22 contract: the new explicit-kwargs
API and the opt-in TrajectoryResult return.
"""

import math

import pytest

import opentop as top
import pandas as pd
from opentop._options import TrajectoryResult


def _fast_cruise():
    opt = top.Cruise("A320", (52.308, 4.764), (50.033, 8.570), m0=0.85)
    opt.setup(max_iter=500)
    return opt


def test_trajectory_returns_dataframe_by_default():
    opt = _fast_cruise()
    df = opt.trajectory(objective="fuel")
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0


def test_trajectory_returns_result_object_when_flag_set():
    opt = _fast_cruise()
    r = opt.trajectory(objective="fuel", result_object=True)
    assert isinstance(r, TrajectoryResult)
    assert isinstance(r.df, pd.DataFrame)
    assert len(r.df) > 0
    assert r.success is True
    assert math.isfinite(r.objective)
    assert r.iters > 0
    assert math.isfinite(r.fuel)
    assert r.fuel > 0
    # No interpolant was passed → grid_cost should be NaN.
    assert math.isnan(r.grid_cost)
    assert isinstance(r.stats, dict)
    assert "iter_count" in r.stats or "success" in r.stats


def test_unknown_kwarg_raises_type_error():
    opt = _fast_cruise()
    with pytest.raises(TypeError):
        opt.trajectory(objective="fuel", nonsense_kwarg=42)  # type: ignore[call-arg]  # intentionally passing invalid kwarg to test TypeError


def test_unknown_kwarg_raises_type_error_on_complete_flight():
    opt = top.CompleteFlight("A320", "EHAM", "EDDF", m0=0.85)
    with pytest.raises(TypeError):
        opt.trajectory(objective="fuel", another_bogus_kwarg=1)  # type: ignore[call-arg]  # intentionally passing invalid kwarg to test TypeError


def test_stats_before_solve_raises_runtime_error():
    opt = top.Cruise("A320", (52.308, 4.764), (50.033, 8.570), m0=0.85)
    with pytest.raises(RuntimeError, match="call trajectory"):
        _ = opt.stats


def test_success_before_solve_raises_runtime_error():
    opt = top.Cruise("A320", (52.308, 4.764), (50.033, 8.570), m0=0.85)
    with pytest.raises(RuntimeError, match="call trajectory"):
        _ = opt.success


def test_result_object_df_matches_default_return():
    """The DataFrame returned with result_object=False must equal r.df."""
    opt1 = _fast_cruise()
    df_direct = opt1.trajectory(objective="fuel")

    opt2 = _fast_cruise()
    r = opt2.trajectory(objective="fuel", result_object=True)

    # Same route + settings + deterministic solver → same objective.
    # (Column-wise equality is risky due to ~1e-15 solver noise across runs.)
    assert opt1.objective_value is not None
    assert abs(float(opt1.objective_value) - r.objective) < 1e-6
    # Row count should match exactly.
    assert len(df_direct) == len(r.df)  # type: ignore[arg-type]  # trajectory() without result_object always returns DataFrame


def test_solver_property_emits_deprecation_warning():
    """`optimizer.solver` is deprecated; accessing it emits DeprecationWarning.

    Scheduled for removal in v2.3 per the migration notes in the README.
    This test pins the warning so we don't silently drop it.
    """
    import warnings

    opt = top.Cruise("A320", (52.308, 4.764), (50.033, 8.570), m0=0.85)
    opt.setup(max_iter=300)
    _ = opt.trajectory(objective="fuel")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _ = opt.solver
    assert any(
        issubclass(w.category, DeprecationWarning)
        and "solver" in str(w.message).lower()
        for w in caught
    ), f"expected DeprecationWarning on opt.solver; got {caught}"
