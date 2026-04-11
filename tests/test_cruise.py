"""Tests for the Cruise trajectory optimizer."""

import pytest

import opentop as top


class TestCruise:
    """Tests for the Cruise optimizer."""

    def test_cruise_default(self, aircraft_type, short_flight):
        """Cruise should produce a valid trajectory with default settings."""
        optimizer = top.Cruise(
            aircraft_type,
            short_flight["origin"],
            short_flight["destination"],
            short_flight["m0"],
        )
        df = optimizer.trajectory(objective="fuel")

        assert df is not None
        assert len(df) > 0
        assert "altitude" in df.columns
        assert "heading" in df.columns
        assert "mach" in df.columns

    def test_cruise_altitude_reasonable(self, aircraft_type, short_flight):
        """Cruise altitude should be at a reasonable flight level."""
        optimizer = top.Cruise(
            aircraft_type,
            short_flight["origin"],
            short_flight["destination"],
            short_flight["m0"],
        )
        df = optimizer.trajectory(objective="fuel")

        assert df.altitude.min() > 20000
        assert df.altitude.max() < 45000

    def test_cruise_heading_reasonable(self, aircraft_type, short_flight):
        """Heading should stay within a reasonable range."""
        optimizer = top.Cruise(
            aircraft_type,
            short_flight["origin"],
            short_flight["destination"],
            short_flight["m0"],
        )
        df = optimizer.trajectory(objective="fuel")

        heading_range = df.heading.max() - df.heading.min()
        assert heading_range < 30

    def test_cruise_mass_decreases(self, aircraft_type, short_flight):
        """Mass should decrease during cruise (fuel burned)."""
        optimizer = top.Cruise(
            aircraft_type,
            short_flight["origin"],
            short_flight["destination"],
            short_flight["m0"],
        )
        df = optimizer.trajectory(objective="fuel")

        assert df.mass.iloc[-1] < df.mass.iloc[0]

    def test_cruise_fuel_cost_column(self, aircraft_type, short_flight):
        """Cruise trajectory should include fuel_cost column."""
        optimizer = top.Cruise(
            aircraft_type,
            short_flight["origin"],
            short_flight["destination"],
            short_flight["m0"],
        )
        df = optimizer.trajectory(objective="fuel")

        assert "fuel_cost" in df.columns
        assert (df["fuel_cost"] >= 0).all()

    def test_cruise_grid_cost_nan_without_interpolant(self, aircraft_type, short_flight):
        """Grid cost should be NaN when no interpolant is provided."""
        optimizer = top.Cruise(
            aircraft_type,
            short_flight["origin"],
            short_flight["destination"],
            short_flight["m0"],
        )
        df = optimizer.trajectory(objective="fuel")

        assert "grid_cost" in df.columns
        assert df["grid_cost"].isna().all()

    def test_cruise_time_objective(self, aircraft_type, short_flight):
        """Cruise with time objective should produce a valid trajectory."""
        optimizer = top.Cruise(
            aircraft_type,
            short_flight["origin"],
            short_flight["destination"],
            short_flight["m0"],
        )
        df = optimizer.trajectory(objective="time")

        assert df is not None
        assert len(df) > 0

    def test_cruise_medium_route(self, aircraft_type, medium_flight):
        """Cruise should work on a medium-distance route."""
        optimizer = top.Cruise(
            aircraft_type,
            medium_flight["origin"],
            medium_flight["destination"],
            medium_flight["m0"],
        )
        df = optimizer.trajectory(objective="fuel")

        assert df is not None
        assert len(df) > 0
        assert df.mass.iloc[-1] < df.mass.iloc[0]
