"""Tests for the auto_rescale_objective option on trajectory()."""

from openap import top


class TestAutoRescale:
    """Tests for auto_rescale_objective kwarg."""

    def test_fuel_only_same_optimum(self, aircraft_type, short_flight):
        """auto_rescale=True must land on the same fuel-only optimum as default."""
        opt1 = top.Cruise(
            aircraft_type,
            short_flight["origin"],
            short_flight["destination"],
            short_flight["m0"],
        )
        opt1.setup(max_iter=500)
        df1 = opt1.trajectory(objective="fuel")

        opt2 = top.Cruise(
            aircraft_type,
            short_flight["origin"],
            short_flight["destination"],
            short_flight["m0"],
        )
        opt2.setup(max_iter=500)
        df2 = opt2.trajectory(objective="fuel", auto_rescale_objective=True)

        fuel1 = df1.mass.iloc[0] - df1.mass.iloc[-1]
        fuel2 = df2.mass.iloc[0] - df2.mass.iloc[-1]
        assert abs(fuel1 - fuel2) < 1.0, (
            f"Fuel burn mismatch: {fuel1:.2f} vs {fuel2:.2f} kg"
        )

    def test_objective_value_is_physical(self, aircraft_type, short_flight):
        """With auto_rescale, objective_value must be reported in physical units."""
        opt = top.Cruise(
            aircraft_type,
            short_flight["origin"],
            short_flight["destination"],
            short_flight["m0"],
        )
        opt.setup(max_iter=500)
        df = opt.trajectory(objective="fuel", auto_rescale_objective=True)

        fuel = df.mass.iloc[0] - df.mass.iloc[-1]
        # objective_value should match the physical fuel burn, not the
        # internal scaled value (which would be O(1) near the optimum).
        assert abs(opt.objective_value - fuel) / fuel < 0.01

    def test_rescale_factor_stored(self, aircraft_type, short_flight):
        """When auto_rescale is on, the factor should be exposed."""
        opt = top.Cruise(
            aircraft_type,
            short_flight["origin"],
            short_flight["destination"],
            short_flight["m0"],
        )
        opt.setup(max_iter=500)
        opt.trajectory(objective="fuel", auto_rescale_objective=True)
        assert opt._objective_rescale > 1.0

    def test_rescale_off_by_default(self, aircraft_type, short_flight):
        """Without the kwarg, _objective_rescale should be 1.0 (no-op)."""
        opt = top.Cruise(
            aircraft_type,
            short_flight["origin"],
            short_flight["destination"],
            short_flight["m0"],
        )
        opt.setup(max_iter=500)
        opt.trajectory(objective="fuel")
        assert opt._objective_rescale == 1.0

    def test_zero_objective_floor(self, aircraft_type, short_flight):
        """If the objective at the initial guess is tiny, rescale must floor at 1.0."""
        opt = top.Cruise(
            aircraft_type,
            short_flight["origin"],
            short_flight["destination"],
            short_flight["m0"],
        )
        opt.setup(max_iter=500)

        # Custom objective that integrates to near zero at init (tiny constant)
        def tiny_obj(x, u, dt, **kwargs):
            return opt.obj_fuel(x, u, dt, **kwargs) * 1e-20

        opt.trajectory(objective=tiny_obj, auto_rescale_objective=True)
        # The floor max(abs(f0), 1.0) should clamp to 1.0 since f0 is tiny
        assert opt._objective_rescale == 1.0

    def test_completeflight_fuel(self, aircraft_type, short_flight):
        """auto_rescale should work on CompleteFlight too."""
        opt = top.CompleteFlight(
            aircraft_type,
            short_flight["origin"],
            short_flight["destination"],
            short_flight["m0"],
        )
        opt.setup(max_iter=500)
        df = opt.trajectory(objective="fuel", auto_rescale_objective=True)

        assert df is not None
        assert opt.solver.stats()["success"]
        fuel = df.mass.iloc[0] - df.mass.iloc[-1]
        assert 1000 < fuel < 20_000, f"Unreasonable fuel burn {fuel:.0f} kg"
