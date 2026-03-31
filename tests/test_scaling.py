"""Tests for NLP variable scaling."""

import pytest

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
