"""Tests for CompleteFlight and MultiPhase trajectory optimizers."""

import pytest

from openap import top


class TestCompleteFlight:
    """Tests for the CompleteFlight optimizer."""

    def test_complete_flight_default(self, aircraft_type, short_flight):
        """CompleteFlight should produce a valid trajectory."""
        optimizer = top.CompleteFlight(
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

    def test_complete_flight_starts_and_ends_low(self, aircraft_type, short_flight):
        """CompleteFlight should start and end at low altitude."""
        optimizer = top.CompleteFlight(
            aircraft_type,
            short_flight["origin"],
            short_flight["destination"],
            short_flight["m0"],
        )
        df = optimizer.trajectory(objective="fuel")

        assert df.altitude.iloc[0] < 1000
        assert df.altitude.iloc[-1] < 1000

    def test_complete_flight_climbs_and_descends(self, aircraft_type, short_flight):
        """CompleteFlight should climb to cruise and descend."""
        optimizer = top.CompleteFlight(
            aircraft_type,
            short_flight["origin"],
            short_flight["destination"],
            short_flight["m0"],
        )
        df = optimizer.trajectory(objective="fuel")

        assert df.altitude.max() > 20000

    def test_complete_flight_mass_decreases(self, aircraft_type, short_flight):
        """Mass should decrease over the flight."""
        optimizer = top.CompleteFlight(
            aircraft_type,
            short_flight["origin"],
            short_flight["destination"],
            short_flight["m0"],
        )
        df = optimizer.trajectory(objective="fuel")

        assert df.mass.iloc[-1] < df.mass.iloc[0]

    def test_complete_flight_heading_reasonable(self, aircraft_type, short_flight):
        """Heading should stay within a reasonable range."""
        optimizer = top.CompleteFlight(
            aircraft_type,
            short_flight["origin"],
            short_flight["destination"],
            short_flight["m0"],
        )
        df = optimizer.trajectory(objective="fuel")

        heading_range = df.heading.max() - df.heading.min()
        assert heading_range < 30

    def test_complete_flight_fuel_cost_column(self, aircraft_type, short_flight):
        """CompleteFlight trajectory should include fuel_cost column."""
        optimizer = top.CompleteFlight(
            aircraft_type,
            short_flight["origin"],
            short_flight["destination"],
            short_flight["m0"],
        )
        df = optimizer.trajectory(objective="fuel")

        assert "fuel_cost" in df.columns
        assert (df["fuel_cost"] >= 0).all()

    def test_complete_flight_medium_route(self, aircraft_type, medium_flight):
        """CompleteFlight should work on a medium-distance route."""
        optimizer = top.CompleteFlight(
            aircraft_type,
            medium_flight["origin"],
            medium_flight["destination"],
            medium_flight["m0"],
        )
        df = optimizer.trajectory(objective="fuel")

        assert df is not None
        assert len(df) > 0


class TestMultiPhase:
    """Tests for the MultiPhase optimizer."""

    def test_multiphase_default(self, aircraft_type, medium_flight):
        """MultiPhase should produce a valid trajectory."""
        optimizer = top.MultiPhase(
            aircraft_type,
            medium_flight["origin"],
            medium_flight["destination"],
            medium_flight["m0"],
        )
        df = optimizer.trajectory(objective="fuel")

        assert df is not None
        assert len(df) > 0
        assert "altitude" in df.columns

    def test_multiphase_starts_and_ends_low(self, aircraft_type, medium_flight):
        """MultiPhase should start and end at low altitude."""
        optimizer = top.MultiPhase(
            aircraft_type,
            medium_flight["origin"],
            medium_flight["destination"],
            medium_flight["m0"],
        )
        df = optimizer.trajectory(objective="fuel")

        assert df.altitude.iloc[0] < 5000
        assert df.altitude.iloc[-1] < 1000

    def test_multiphase_mass_decreases(self, aircraft_type, medium_flight):
        """Mass should decrease over the flight."""
        optimizer = top.MultiPhase(
            aircraft_type,
            medium_flight["origin"],
            medium_flight["destination"],
            medium_flight["m0"],
        )
        df = optimizer.trajectory(objective="fuel")

        assert df.mass.iloc[-1] < df.mass.iloc[0]

    def test_multiphase_solver_stats(self, aircraft_type, medium_flight):
        """get_solver_stats should return dict with phase keys."""
        optimizer = top.MultiPhase(
            aircraft_type,
            medium_flight["origin"],
            medium_flight["destination"],
            medium_flight["m0"],
        )
        optimizer.trajectory(objective="fuel")

        stats = optimizer.get_solver_stats()

        assert isinstance(stats, dict)
        assert "climb" in stats
        assert "cruise" in stats
        assert "descent" in stats
