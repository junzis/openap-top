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

