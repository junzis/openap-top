"""Tests for the Climb trajectory optimizer."""

import pytest

import opentop as top


@pytest.fixture(scope="module")
def climb_optimizer(aircraft_type, medium_flight):
    return top.Climb(
        aircraft_type,
        medium_flight["origin"],
        medium_flight["destination"],
        medium_flight["m0"],
    )


@pytest.fixture(scope="module")
def climb_clipped_df(climb_optimizer):
    return climb_optimizer.trajectory(objective="fuel")


@pytest.fixture(scope="module")
def climb_full_df(climb_optimizer):
    return climb_optimizer.trajectory(objective="fuel", remove_cruise=False)


@pytest.fixture(scope="module")
def climb_alt_stop_df(climb_optimizer):
    return climb_optimizer.trajectory(
        objective="fuel", alt_stop=30000, remove_cruise=False
    )


@pytest.fixture(scope="module")
def climb_alt_stop_low_df(climb_optimizer):
    return climb_optimizer.trajectory(
        objective="fuel", alt_stop=25000, remove_cruise=False
    )


class TestClimb:
    def test_valid_trajectory(self, climb_clipped_df):
        df = climb_clipped_df
        assert df is not None
        assert len(df) > 0
        for col in ("altitude", "heading", "vertical_rate"):
            assert col in df.columns

    def test_altitude_increases(self, climb_clipped_df):
        assert climb_clipped_df.altitude.iloc[-1] > climb_clipped_df.altitude.iloc[0]

    def test_remove_cruise_clips(self, climb_clipped_df, climb_full_df):
        assert len(climb_clipped_df) <= len(climb_full_df)
        assert (climb_clipped_df.vertical_rate > 100).all()

    def test_remove_cruise_false_includes_cruise(self, climb_full_df):
        assert (climb_full_df.vertical_rate.abs() < 100).any()

    def test_alt_stop(self, climb_alt_stop_df):
        assert abs(climb_alt_stop_df.altitude.max() - 30000) < 500

    def test_alt_stop_vs_default(self, climb_full_df, climb_alt_stop_low_df):
        assert climb_alt_stop_low_df.altitude.max() < climb_full_df.altitude.max()

    def test_heading_reasonable(self, climb_clipped_df):
        df = climb_clipped_df
        assert df.heading.max() - df.heading.min() < 30

    def test_mass_decreases(self, climb_clipped_df):
        df = climb_clipped_df
        assert df.mass.iloc[-1] < df.mass.iloc[0]

    def test_fuel_cost_column(self, climb_clipped_df):
        df = climb_clipped_df
        assert "fuel_cost" in df.columns
        assert (df["fuel_cost"].dropna() >= 0).all()


def test_climb_north_south_route_converges():
    """Regression: the colinearity constraint must handle near-vertical geometries.

    Rewritten as cross-product, no division by (xp_2 - xp_1), so N-S routes
    don't blow up.
    """
    import opentop as top

    # Amsterdam → Copenhagen: bearing ~57°, i.e. mostly N; xp_2 - xp_1 is
    # small relative to yp_2 - yp_1. Pre-fix this may still converge due to
    # luck; this test is mainly a guard that future changes preserve the
    # refactored constraint form.
    opt = top.Cruise("A320", (52.308, 4.764), (55.618, 12.656), m0=0.85)
    opt.setup(max_iter=800)
    dfcr = opt.trajectory(objective="fuel")
    assert dfcr is not None

    clb = top.Climb("A320", (52.308, 4.764), (55.618, 12.656), m0=0.85)
    clb.setup(max_iter=800)
    dfcl = clb.trajectory(objective="fuel", df_cruise=dfcr)
    assert dfcl is not None
