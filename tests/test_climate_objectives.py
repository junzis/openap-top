"""Parametrized climate-objective smoke.

Pins that all 6 metrics (gwp20/50/100, gtp20/50/100) produce finite,
non-zero objective values and converge on a short cruise. Before Phase 3
restructures objective dispatch, this is the safety net.

Note on sign: GWP metrics always sum to a positive value (all coefficients
positive). GTP metrics can be negative — in particular gtp20 and gtp50
have large negative NOx/SOx coefficients that dominate at cruise, making
the physical objective negative. Finiteness and non-zero are the invariants
we can assert uniformly; positivity is metric-dependent.
"""
import math

import pytest

import opentop as top

CLIMATE_METRICS = ["gwp20", "gwp50", "gwp100", "gtp20", "gtp50", "gtp100"]

# GWP metrics: all species coefficients positive → objective always positive.
# GTP metrics: NOx / SOx coefficients negative at short timescales → can be negative.
_POSITIVE_METRICS = {"gwp20", "gwp50", "gwp100"}


@pytest.mark.parametrize("metric", CLIMATE_METRICS)
def test_climate_objective_finite_and_positive(metric):
    opt = top.Cruise("A320", (52.308, 4.764), (50.033, 8.570), m0=0.85)
    opt.setup(max_iter=500)
    df = opt.trajectory(objective=metric)

    assert df is not None, f"{metric}: trajectory returned None"
    assert opt.success, \
        f"{metric}: solver failed: {opt.stats}"
    assert math.isfinite(opt.objective_value), \
        f"{metric}: objective is not finite: {opt.objective_value}"
    assert opt.objective_value != 0, \
        f"{metric}: objective is zero: {opt.objective_value}"
    if metric in _POSITIVE_METRICS:
        assert opt.objective_value > 0, \
            f"{metric}: GWP objective non-positive: {opt.objective_value}"
