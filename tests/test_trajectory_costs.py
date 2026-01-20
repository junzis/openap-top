"""Tests for fuel_cost and grid_cost columns in trajectory output."""

import numpy as np
import pandas as pd
import pytest

from openap import top


class TestTrajectoryFuelCost:
    """Tests for fuel_cost column in trajectory DataFrame."""

    def test_cruise_trajectory_has_fuel_cost_column(self, aircraft_type, short_flight):
        """Cruise trajectory should include fuel_cost column."""
        optimizer = top.Cruise(
            aircraft_type,
            short_flight["origin"],
            short_flight["destination"],
            short_flight["m0"],
        )
        optimizer.setup(nodes=20)
        flight = optimizer.trajectory(objective="fuel")

        assert flight is not None
        assert "fuel_cost" in flight.columns

    def test_fuel_cost_values_are_positive(self, aircraft_type, short_flight):
        """Fuel cost values should be positive."""
        optimizer = top.Cruise(
            aircraft_type,
            short_flight["origin"],
            short_flight["destination"],
            short_flight["m0"],
        )
        optimizer.setup(nodes=20)
        flight = optimizer.trajectory(objective="fuel")

        assert (flight["fuel_cost"] >= 0).all()

    def test_fuel_cost_sum_matches_mass_difference(self, aircraft_type, short_flight):
        """Sum of fuel_cost should approximately match mass difference."""
        optimizer = top.Cruise(
            aircraft_type,
            short_flight["origin"],
            short_flight["destination"],
            short_flight["m0"],
        )
        optimizer.setup(nodes=20)
        flight = optimizer.trajectory(objective="fuel")

        mass_diff = flight["mass"].iloc[0] - flight["mass"].iloc[-1]
        fuel_cost_sum = flight["fuel_cost"].sum()

        # Allow 10% tolerance due to numerical integration differences
        assert abs(fuel_cost_sum - mass_diff) / mass_diff < 0.1

    def test_complete_flight_has_fuel_cost_column(self, aircraft_type, short_flight):
        """CompleteFlight trajectory should include fuel_cost column."""
        optimizer = top.CompleteFlight(
            aircraft_type,
            short_flight["origin"],
            short_flight["destination"],
            short_flight["m0"],
        )
        optimizer.setup(nodes=30)
        flight = optimizer.trajectory(objective="fuel")

        assert flight is not None
        assert "fuel_cost" in flight.columns
        assert (flight["fuel_cost"] >= 0).all()


class TestTrajectoryGridCost:
    """Tests for grid_cost column in trajectory DataFrame."""

    def test_grid_cost_is_nan_without_interpolant(self, aircraft_type, short_flight):
        """Grid cost should be NaN when no interpolant is provided."""
        optimizer = top.Cruise(
            aircraft_type,
            short_flight["origin"],
            short_flight["destination"],
            short_flight["m0"],
        )
        optimizer.setup(nodes=20)
        flight = optimizer.trajectory(objective="fuel")

        assert "grid_cost" in flight.columns
        assert flight["grid_cost"].isna().all()

    def test_grid_cost_column_always_present(self, aircraft_type, short_flight):
        """Grid cost column should always be present in output."""
        optimizer = top.Cruise(
            aircraft_type,
            short_flight["origin"],
            short_flight["destination"],
            short_flight["m0"],
        )
        optimizer.setup(nodes=20)
        flight = optimizer.trajectory(objective="fuel")

        assert "grid_cost" in flight.columns


class TestMultiPhaseSolverStats:
    """Tests for MultiPhase.get_solver_stats() method."""

    def test_multiphase_has_get_solver_stats(self):
        """MultiPhase should have get_solver_stats method."""
        assert hasattr(top.MultiPhase, "get_solver_stats")

    def test_get_solver_stats_returns_dict(self, aircraft_type, medium_flight):
        """get_solver_stats should return dict with climb, cruise, descent keys."""
        optimizer = top.MultiPhase(
            aircraft_type,
            medium_flight["origin"],
            medium_flight["destination"],
            medium_flight["m0"],
        )
        optimizer.setup(nodes=20)
        optimizer.trajectory(objective="fuel")

        stats = optimizer.get_solver_stats()

        assert isinstance(stats, dict)
        assert "climb" in stats
        assert "cruise" in stats
        assert "descent" in stats


class TestToTrajectoryKwargs:
    """Tests for kwargs passing to to_trajectory."""

    def test_kwargs_passed_through_cruise(self, aircraft_type, short_flight):
        """Kwargs should be passed through to to_trajectory in Cruise."""
        optimizer = top.Cruise(
            aircraft_type,
            short_flight["origin"],
            short_flight["destination"],
            short_flight["m0"],
        )
        optimizer.setup(nodes=20)

        # This should not raise - kwargs are passed through
        flight = optimizer.trajectory(objective="fuel", interpolant=None)

        assert flight is not None
        assert "fuel_cost" in flight.columns
        assert "grid_cost" in flight.columns
