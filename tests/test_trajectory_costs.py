"""Tests for fuel_cost / grid_cost semantics not covered elsewhere.

Most presence/sign assertions live in the per-phase test modules. This
module only covers properties that genuinely require their own solve or
are specific to the trajectory-cost semantics (sum vs mass difference,
keyword-argument forwarding).
"""

import pytest

import opentop as top


@pytest.fixture(scope="module")
def cruise_small_nodes_df(aircraft_type, short_flight):
    optimizer = top.Cruise(
        aircraft_type,
        short_flight["origin"],
        short_flight["destination"],
        short_flight["m0"],
    )
    optimizer.setup(nodes=20)
    return optimizer, optimizer.trajectory(objective="fuel")


def test_fuel_cost_sum_matches_mass_difference(cruise_small_nodes_df):
    _, flight = cruise_small_nodes_df
    mass_diff = flight["mass"].iloc[0] - flight["mass"].iloc[-1]
    fuel_cost_sum = flight["fuel_cost"].sum()
    assert abs(fuel_cost_sum - mass_diff) / mass_diff < 0.1


def test_kwargs_passed_through_cruise(aircraft_type, short_flight):
    """Passing an extra kwarg (interpolant=None) must not raise."""
    optimizer = top.Cruise(
        aircraft_type,
        short_flight["origin"],
        short_flight["destination"],
        short_flight["m0"],
    )
    optimizer.setup(nodes=20)
    flight = optimizer.trajectory(objective="fuel", interpolant=None)
    assert flight is not None
    assert "fuel_cost" in flight.columns
    assert "grid_cost" in flight.columns
