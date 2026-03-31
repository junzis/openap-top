"""Tests for NLP variable scaling."""

import numpy as np
import pytest
from openap.aero import fpm, ft

from openap import top


class TestScalingInfrastructure:
    """Tests for Base scaling methods."""

    def test_default_scales_are_one(self):
        """All scale factors should default to 1.0."""
        optimizer = top.Cruise("A320", "EHAM", "EDDF", 0.85)
        assert optimizer.scale_x == 1.0
        assert optimizer.scale_y == 1.0
        assert optimizer.scale_h == 1.0
        assert optimizer.scale_m == 1.0
        assert optimizer.scale_t == 1.0
        assert optimizer.scale_mach == 1.0
        assert optimizer.scale_vs == 1.0
        assert optimizer.scale_psi == 1.0
        assert optimizer.scale_force == 1.0
        assert optimizer.scale_energy == 1.0
        assert optimizer.scale_obj == 1.0

    def test_set_scaling_updates_values(self):
        """set_scaling should update the specified scale factors."""
        optimizer = top.Cruise("A320", "EHAM", "EDDF", 0.85)
        optimizer.set_scaling(scale_x=1000.0, scale_h=12000.0)
        assert optimizer.scale_x == 1000.0
        assert optimizer.scale_h == 12000.0
        # Others remain at default
        assert optimizer.scale_y == 1.0

    def test_set_scaling_rejects_invalid_keys(self):
        """set_scaling should raise AttributeError for unknown scale keys."""
        optimizer = top.Cruise("A320", "EHAM", "EDDF", 0.85)
        with pytest.raises(AttributeError, match="Unknown scaling key"):
            optimizer.set_scaling(scale_xx=100.0)

    def test_reset_scaling_restores_defaults(self):
        """reset_scaling should set all factors back to 1.0."""
        optimizer = top.Cruise("A320", "EHAM", "EDDF", 0.85)
        optimizer.set_scaling(scale_x=5000.0, scale_m=70000.0)
        optimizer.reset_scaling()
        assert optimizer.scale_x == 1.0
        assert optimizer.scale_m == 1.0


class TestCruiseScaling:
    """Tests for Cruise with scaling enabled."""

    def test_init_conditions_scaling_sets_factors(self):
        """When scaling=True, init_conditions should set non-trivial scale factors."""
        optimizer = top.Cruise("A320", "EHAM", "EDDF", 0.85)
        optimizer.init_conditions(scaling=True)
        assert optimizer.scale_x > 1.0
        assert optimizer.scale_m > 1.0
        assert optimizer.scale_t > 1.0

    def test_init_conditions_no_scaling_keeps_defaults(self):
        """When scaling=False, init_conditions should reset factors to 1.0."""
        optimizer = top.Cruise("A320", "EHAM", "EDDF", 0.85)
        optimizer.init_conditions(scaling=True)
        assert optimizer.scale_x > 1.0
        optimizer.init_conditions(scaling=False)
        assert optimizer.scale_x == 1.0
        assert optimizer.scale_m == 1.0

    def test_scaled_bounds_are_order_one(self):
        """Scaled bounds should be roughly O(1)."""
        optimizer = top.Cruise("A320", "EHAM", "EDDF", 0.85)
        optimizer.init_conditions(scaling=True)
        for val in optimizer.x_lb + optimizer.x_ub:
            assert abs(val) < 100, f"Bound {val} is not O(1)"

    def test_scaled_guess_is_order_one(self):
        """Scaled initial guess should be roughly O(1)."""
        optimizer = top.Cruise("A320", "EHAM", "EDDF", 0.85)
        optimizer.init_conditions(scaling=True)
        for row in optimizer.x_guess:
            for val in row:
                assert abs(val) < 100, f"Guess value {val} is not O(1)"

    def test_cruise_trajectory_with_scaling(self, aircraft_type, short_flight):
        """Cruise with scaling=True should produce a valid trajectory."""
        optimizer = top.Cruise(
            aircraft_type,
            short_flight["origin"],
            short_flight["destination"],
            short_flight["m0"],
        )
        df = optimizer.trajectory(objective="fuel", scaling=True)

        assert df is not None
        assert len(df) > 0
        assert df.altitude.min() > 20000
        assert df.altitude.max() < 45000
        assert df.mass.iloc[-1] < df.mass.iloc[0]

    def test_cruise_scaling_vs_unscaled_similar(self, aircraft_type, short_flight):
        """Scaled and unscaled Cruise should produce similar fuel burn."""
        optimizer = top.Cruise(
            aircraft_type,
            short_flight["origin"],
            short_flight["destination"],
            short_flight["m0"],
        )
        df_unscaled = optimizer.trajectory(objective="fuel", scaling=False)
        df_scaled = optimizer.trajectory(objective="fuel", scaling=True)

        if df_unscaled is not None and df_scaled is not None:
            fuel_unscaled = df_unscaled.mass.iloc[0] - df_unscaled.mass.iloc[-1]
            fuel_scaled = df_scaled.mass.iloc[0] - df_scaled.mass.iloc[-1]
            assert abs(fuel_scaled - fuel_unscaled) / fuel_unscaled < 0.10
