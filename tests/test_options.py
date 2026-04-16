"""Unit tests for the _options dataclasses."""

import dataclasses

import pytest

import pandas as pd
from opentop._options import GridOptions, SolveOptions, TrajectoryResult


def test_solve_options_defaults():
    opts = SolveOptions()
    assert opts.max_iter == 1500
    assert opts.max_fuel is None
    assert opts.auto_rescale_objective is True
    assert opts.return_failed is False
    assert opts.initial_guess is None
    assert opts.alt_start is None
    assert opts.alt_stop is None
    assert opts.remove_cruise is False
    assert opts.exact_hessian is False


def test_solve_options_is_frozen():
    opts = SolveOptions()
    with pytest.raises(dataclasses.FrozenInstanceError):
        opts.max_iter = 999  # type: ignore[misc]


def test_grid_options_defaults():
    g = GridOptions()
    assert g.interpolant is None
    assert g.n_dim == 3
    assert g.time_dependent is False


def test_grid_options_is_frozen():
    g = GridOptions()
    with pytest.raises(dataclasses.FrozenInstanceError):
        g.n_dim = 4  # type: ignore[misc]


def test_trajectory_result_construction():
    r = TrajectoryResult(
        df=pd.DataFrame({"x": [1, 2]}),
        success=True,
        status="Solve_Succeeded",
        objective=1.0,
        iters=10,
        fuel=100.0,
        grid_cost=float("nan"),
        stats={"iter_count": 10},
    )
    assert r.success is True
    assert r.status == "Solve_Succeeded"
    assert r.objective == 1.0
    assert r.iters == 10
    assert r.fuel == 100.0
    assert r.grid_cost != r.grid_cost  # NaN check
    assert len(r.df) == 2
    assert r.stats == {"iter_count": 10}


def test_trajectory_result_is_frozen():
    r = TrajectoryResult(
        df=pd.DataFrame(),
        success=True,
        status="ok",
        objective=0.0,
        iters=0,
        fuel=0.0,
        grid_cost=0.0,
        stats={},
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.success = False  # type: ignore[misc]
