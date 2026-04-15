"""Tests for the Cruise trajectory optimizer."""

import pytest

import opentop as top


@pytest.fixture(scope="module")
def cruise_df(aircraft_type, short_flight):
    optimizer = top.Cruise(
        aircraft_type,
        short_flight["origin"],
        short_flight["destination"],
        short_flight["m0"],
    )
    return optimizer.trajectory(objective="fuel")


@pytest.fixture(scope="module")
def cruise_time_df(aircraft_type, short_flight):
    optimizer = top.Cruise(
        aircraft_type,
        short_flight["origin"],
        short_flight["destination"],
        short_flight["m0"],
    )
    return optimizer.trajectory(objective="time")


@pytest.fixture(scope="module")
def cruise_medium_df(aircraft_type, medium_flight):
    optimizer = top.Cruise(
        aircraft_type,
        medium_flight["origin"],
        medium_flight["destination"],
        medium_flight["m0"],
    )
    return optimizer.trajectory(objective="fuel")


class TestCruise:
    def test_valid_trajectory(self, cruise_df):
        df = cruise_df
        assert df is not None
        assert len(df) > 0
        for col in ("altitude", "heading", "mach"):
            assert col in df.columns

    def test_altitude_reasonable(self, cruise_df):
        assert cruise_df.altitude.min() > 20000
        assert cruise_df.altitude.max() < 45000

    def test_heading_reasonable(self, cruise_df):
        assert cruise_df.heading.max() - cruise_df.heading.min() < 30

    def test_mass_decreases(self, cruise_df):
        assert cruise_df.mass.iloc[-1] < cruise_df.mass.iloc[0]

    def test_fuel_cost_column(self, cruise_df):
        assert "fuel_cost" in cruise_df.columns
        assert (cruise_df["fuel_cost"].dropna() >= 0).all()

    def test_grid_cost_nan_without_interpolant(self, cruise_df):
        assert "grid_cost" in cruise_df.columns
        assert cruise_df["grid_cost"].isna().all()

    def test_time_objective(self, cruise_time_df):
        assert cruise_time_df is not None
        assert len(cruise_time_df) > 0

    def test_medium_route(self, cruise_medium_df):
        df = cruise_medium_df
        assert df is not None
        assert len(df) > 0
        assert df.mass.iloc[-1] < df.mass.iloc[0]
