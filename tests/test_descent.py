"""Tests for the Descent trajectory optimizer."""

import pytest

import opentop as top


@pytest.fixture(scope="module")
def descent_optimizer(aircraft_type, medium_flight):
    return top.Descent(
        aircraft_type,
        medium_flight["origin"],
        medium_flight["destination"],
        medium_flight["m0"],
    )


@pytest.fixture(scope="module")
def descent_clipped_df(descent_optimizer):
    return descent_optimizer.trajectory(objective="fuel")


@pytest.fixture(scope="module")
def descent_full_df(descent_optimizer):
    return descent_optimizer.trajectory(objective="fuel", remove_cruise=False)


@pytest.fixture(scope="module")
def descent_alt_start_df(descent_optimizer):
    return descent_optimizer.trajectory(
        objective="fuel", alt_start=30000, remove_cruise=False
    )


class TestDescent:
    def test_valid_trajectory(self, descent_clipped_df):
        df = descent_clipped_df
        assert df is not None
        assert len(df) > 0
        for col in ("altitude", "heading", "vertical_rate"):
            assert col in df.columns

    def test_remove_cruise_clips(self, descent_clipped_df, descent_full_df):
        assert len(descent_clipped_df) <= len(descent_full_df)
        assert (descent_clipped_df.vertical_rate < -100).all()

    def test_remove_cruise_false_includes_cruise(self, descent_full_df):
        assert (descent_full_df.vertical_rate.abs() < 100).any()

    def test_ends_low(self, descent_full_df):
        assert descent_full_df.altitude.iloc[-1] < 1000

    def test_alt_start(self, descent_alt_start_df):
        assert abs(descent_alt_start_df.altitude.iloc[0] - 30000) < 500

    def test_heading_reasonable(self, descent_full_df):
        df = descent_full_df
        assert df.heading.max() - df.heading.min() < 30

    def test_mass_decreases(self, descent_full_df):
        df = descent_full_df
        assert df.mass.iloc[-1] < df.mass.iloc[0]

    def test_fuel_cost_column(self, descent_full_df):
        df = descent_full_df
        assert "fuel_cost" in df.columns
        assert (df["fuel_cost"] >= 0).all()
