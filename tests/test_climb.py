"""Tests for the Climb trajectory optimizer."""

import pytest

import opentop as top


class TestClimb:
    """Tests for the Climb optimizer."""

    def test_climb_default(self, aircraft_type, medium_flight):
        """Climb should produce a valid trajectory with default settings."""
        optimizer = top.Climb(
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

    def test_climb_altitude_increases(self, aircraft_type, medium_flight):
        """Climb trajectory altitude should generally increase."""
        optimizer = top.Climb(
            aircraft_type,
            medium_flight["origin"],
            medium_flight["destination"],
            medium_flight["m0"],
        )
        df = optimizer.trajectory(objective="fuel")

        assert df.altitude.iloc[-1] > df.altitude.iloc[0]

    def test_climb_remove_cruise_default(self, aircraft_type, medium_flight):
        """Default remove_cruise=True should clip cruise segment."""
        optimizer = top.Climb(
            aircraft_type,
            medium_flight["origin"],
            medium_flight["destination"],
            medium_flight["m0"],
        )
        df_clipped = optimizer.trajectory(objective="fuel")
        df_full = optimizer.trajectory(objective="fuel", remove_cruise=False)

        assert len(df_clipped) <= len(df_full)
        assert (df_clipped.vertical_rate > 100).all()

    def test_climb_remove_cruise_false(self, aircraft_type, medium_flight):
        """remove_cruise=False should include cruise segment."""
        optimizer = top.Climb(
            aircraft_type,
            medium_flight["origin"],
            medium_flight["destination"],
            medium_flight["m0"],
        )
        df = optimizer.trajectory(objective="fuel", remove_cruise=False)

        # Full trajectory should have points with near-zero vertical rate
        assert (df.vertical_rate.abs() < 100).any()

    def test_climb_alt_stop(self, aircraft_type, medium_flight):
        """alt_stop should control the top of climb altitude."""
        optimizer = top.Climb(
            aircraft_type,
            medium_flight["origin"],
            medium_flight["destination"],
            medium_flight["m0"],
        )
        df = optimizer.trajectory(
            objective="fuel", alt_stop=30000, remove_cruise=False
        )

        assert abs(df.altitude.max() - 30000) < 500

    def test_climb_alt_stop_vs_default(self, aircraft_type, medium_flight):
        """alt_stop should produce different altitude than default."""
        optimizer = top.Climb(
            aircraft_type,
            medium_flight["origin"],
            medium_flight["destination"],
            medium_flight["m0"],
        )
        df_default = optimizer.trajectory(objective="fuel", remove_cruise=False)
        df_low = optimizer.trajectory(
            objective="fuel", alt_stop=25000, remove_cruise=False
        )

        assert df_low.altitude.max() < df_default.altitude.max()

    def test_climb_heading_reasonable(self, aircraft_type, medium_flight):
        """Heading should stay within a reasonable range."""
        optimizer = top.Climb(
            aircraft_type,
            medium_flight["origin"],
            medium_flight["destination"],
            medium_flight["m0"],
        )
        df = optimizer.trajectory(objective="fuel")

        heading_range = df.heading.max() - df.heading.min()
        assert heading_range < 30

    def test_climb_mass_decreases(self, aircraft_type, medium_flight):
        """Mass should decrease during climb (fuel burned)."""
        optimizer = top.Climb(
            aircraft_type,
            medium_flight["origin"],
            medium_flight["destination"],
            medium_flight["m0"],
        )
        df = optimizer.trajectory(objective="fuel")

        assert df.mass.iloc[-1] < df.mass.iloc[0]

    def test_climb_fuel_cost_column(self, aircraft_type, medium_flight):
        """Climb trajectory should include fuel_cost column."""
        optimizer = top.Climb(
            aircraft_type,
            medium_flight["origin"],
            medium_flight["destination"],
            medium_flight["m0"],
        )
        df = optimizer.trajectory(objective="fuel")

        assert "fuel_cost" in df.columns
        assert (df["fuel_cost"] >= 0).all()
