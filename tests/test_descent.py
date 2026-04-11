"""Tests for the Descent trajectory optimizer."""

import pytest

import opentop as top


class TestDescent:
    """Tests for the Descent optimizer."""

    def test_descent_default(self, aircraft_type, medium_flight):
        """Descent should produce a valid trajectory with default settings."""
        optimizer = top.Descent(
            aircraft_type,
            medium_flight["origin"],
            medium_flight["destination"],
            medium_flight["m0"],
        )
        df = optimizer.trajectory(objective="fuel")

        assert df is not None
        assert len(df) > 0
        assert "altitude" in df.columns
        assert "heading" in df.columns
        assert "vertical_rate" in df.columns

    def test_descent_remove_cruise_default(self, aircraft_type, medium_flight):
        """Default remove_cruise=True should clip cruise segment."""
        optimizer = top.Descent(
            aircraft_type,
            medium_flight["origin"],
            medium_flight["destination"],
            medium_flight["m0"],
        )
        df_clipped = optimizer.trajectory(objective="fuel")
        df_full = optimizer.trajectory(objective="fuel", remove_cruise=False)

        assert len(df_clipped) <= len(df_full)
        assert (df_clipped.vertical_rate < -100).all()

    def test_descent_remove_cruise_false(self, aircraft_type, medium_flight):
        """remove_cruise=False should include cruise segment."""
        optimizer = top.Descent(
            aircraft_type,
            medium_flight["origin"],
            medium_flight["destination"],
            medium_flight["m0"],
        )
        df = optimizer.trajectory(objective="fuel", remove_cruise=False)

        # Full trajectory should have points with near-zero vertical rate
        assert (df.vertical_rate.abs() < 100).any()

    def test_descent_altitude_decreases_at_end(self, aircraft_type, medium_flight):
        """Descent trajectory should end at low altitude."""
        optimizer = top.Descent(
            aircraft_type,
            medium_flight["origin"],
            medium_flight["destination"],
            medium_flight["m0"],
        )
        df = optimizer.trajectory(objective="fuel", remove_cruise=False)

        assert df.altitude.iloc[-1] < 1000

    def test_descent_alt_start(self, aircraft_type, medium_flight):
        """alt_start should control the start altitude."""
        optimizer = top.Descent(
            aircraft_type,
            medium_flight["origin"],
            medium_flight["destination"],
            medium_flight["m0"],
        )
        df = optimizer.trajectory(
            objective="fuel", alt_start=30000, remove_cruise=False
        )

        assert abs(df.altitude.iloc[0] - 30000) < 500

    def test_descent_heading_reasonable(self, aircraft_type, medium_flight):
        """Heading should stay within a reasonable range."""
        optimizer = top.Descent(
            aircraft_type,
            medium_flight["origin"],
            medium_flight["destination"],
            medium_flight["m0"],
        )
        df = optimizer.trajectory(objective="fuel", remove_cruise=False)

        heading_range = df.heading.max() - df.heading.min()
        assert heading_range < 30

    def test_descent_mass_decreases(self, aircraft_type, medium_flight):
        """Mass should decrease during descent (fuel burned)."""
        optimizer = top.Descent(
            aircraft_type,
            medium_flight["origin"],
            medium_flight["destination"],
            medium_flight["m0"],
        )
        df = optimizer.trajectory(objective="fuel", remove_cruise=False)

        assert df.mass.iloc[-1] < df.mass.iloc[0]

    def test_descent_fuel_cost_column(self, aircraft_type, medium_flight):
        """Descent trajectory should include fuel_cost column."""
        optimizer = top.Descent(
            aircraft_type,
            medium_flight["origin"],
            medium_flight["destination"],
            medium_flight["m0"],
        )
        df = optimizer.trajectory(objective="fuel", remove_cruise=False)

        assert "fuel_cost" in df.columns
        assert (df["fuel_cost"] >= 0).all()
