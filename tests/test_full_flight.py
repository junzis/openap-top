"""Tests for CompleteFlight trajectory optimizer."""

import pytest

import opentop as top


@pytest.fixture(scope="module")
def complete_flight_df(aircraft_type, short_flight):
    optimizer = top.CompleteFlight(
        aircraft_type,
        short_flight["origin"],
        short_flight["destination"],
        short_flight["m0"],
    )
    return optimizer.trajectory(objective="fuel")


@pytest.fixture(scope="module")
def complete_flight_medium_df(aircraft_type, medium_flight):
    optimizer = top.CompleteFlight(
        aircraft_type,
        medium_flight["origin"],
        medium_flight["destination"],
        medium_flight["m0"],
    )
    return optimizer.trajectory(objective="fuel")


class TestCompleteFlight:
    def test_valid_trajectory(self, complete_flight_df):
        df = complete_flight_df
        assert df is not None
        assert len(df) > 0
        assert "altitude" in df.columns
        assert "heading" in df.columns

    def test_starts_and_ends_low(self, complete_flight_df):
        df = complete_flight_df
        assert df.altitude.iloc[0] < 1000
        assert df.altitude.iloc[-1] < 1000

    def test_climbs_to_cruise(self, complete_flight_df):
        assert complete_flight_df.altitude.max() > 20000

    def test_mass_decreases(self, complete_flight_df):
        df = complete_flight_df
        assert df.mass.iloc[-1] < df.mass.iloc[0]

    def test_heading_reasonable(self, complete_flight_df):
        df = complete_flight_df
        assert df.heading.max() - df.heading.min() < 30

    def test_fuel_cost_column(self, complete_flight_df):
        df = complete_flight_df
        assert "fuel_cost" in df.columns
        assert (df["fuel_cost"].dropna() >= 0).all()

    def test_medium_route(self, complete_flight_medium_df):
        df = complete_flight_medium_df
        assert df is not None
        assert len(df) > 0


def test_complete_flight_callable_objective():
    """Verify objective=callable end-to-end, pinning the `(x, u, dt, **kwargs) -> ca.MX`
    contract. Before Phase 3 changes objective dispatch, this ensures user-supplied
    callables keep working."""
    import opentop as top

    opt = top.CompleteFlight("A320", "EHAM", "EDDF", m0=0.85)
    opt.setup(max_iter=1200)

    def fuel_twice(x, u, dt, **kwargs):
        # Trivial callable: 2x fuel. Optimum path should match pure-fuel; scale differs.
        return 2.0 * opt.obj_fuel(x, u, dt)

    df = opt.trajectory(objective=fuel_twice)
    assert df is not None
    assert opt.solver.stats()["success"]


def test_complete_flight_return_failed_returns_df_on_tight_fuel_budget():
    """When max_fuel is impossibly tight, the mass-violation or infeasibility path
    would normally return None. With return_failed=True, the function must return
    the partial DataFrame instead."""
    import opentop as top

    opt = top.CompleteFlight("A320", "EHAM", "EDDF", m0=0.85)
    opt.setup(max_iter=200)

    df = opt.trajectory(
        objective="fuel",
        max_fuel=100.0,          # physically impossible for EHAM→EDDF
        return_failed=True,
    )
    # Regardless of whether the solver fails outright or returns a degenerate
    # trajectory that violates mass constraints, return_failed=True must hand
    # back a DataFrame (not None).
    assert df is not None

