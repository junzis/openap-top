"""Tests for the auto_rescale_objective kwarg on trajectory()."""

import pytest

import opentop as top


@pytest.fixture(scope="module")
def rescaled_optimizer_df(aircraft_type, short_flight):
    opt = top.Cruise(
        aircraft_type,
        short_flight["origin"],
        short_flight["destination"],
        short_flight["m0"],
    )
    opt.setup(max_iter=500)
    df = opt.trajectory(objective="fuel", auto_rescale_objective=True)
    return opt, df


@pytest.fixture(scope="module")
def baseline_optimizer_df(aircraft_type, short_flight):
    opt = top.Cruise(
        aircraft_type,
        short_flight["origin"],
        short_flight["destination"],
        short_flight["m0"],
    )
    opt.setup(max_iter=500)
    df = opt.trajectory(objective="fuel")
    return opt, df


class TestAutoRescale:
    def test_fuel_only_same_optimum(self, baseline_optimizer_df, rescaled_optimizer_df):
        _, df1 = baseline_optimizer_df
        _, df2 = rescaled_optimizer_df
        fuel1 = df1.mass.iloc[0] - df1.mass.iloc[-1]
        fuel2 = df2.mass.iloc[0] - df2.mass.iloc[-1]
        assert abs(fuel1 - fuel2) < 1.0

    def test_objective_value_is_physical(self, rescaled_optimizer_df):
        opt, df = rescaled_optimizer_df
        fuel = df.mass.iloc[0] - df.mass.iloc[-1]
        assert abs(opt.objective_value - fuel) / fuel < 0.01

    def test_rescale_factor_stored(self, rescaled_optimizer_df):
        opt, _ = rescaled_optimizer_df
        # Fuel objective at initial guess is ~1e3 kg → rescale ≫ 1
        assert opt._objective_rescale > 1.0

    def test_rescale_off_by_default(self, baseline_optimizer_df):
        opt, _ = baseline_optimizer_df
        assert opt._objective_rescale == 1.0

    def test_tiny_objective_rescales_up(self, aircraft_type, short_flight):
        """When the physical objective is tiny (e.g. climate metrics),
        the rescale must divide by that tiny value so IPOPT sees O(1).
        """
        opt = top.Cruise(
            aircraft_type,
            short_flight["origin"],
            short_flight["destination"],
            short_flight["m0"],
        )
        opt.setup(max_iter=500)

        def tiny_obj(x, u, dt, **kwargs):
            return opt.obj_fuel(x, u, dt, **kwargs) * 1e-12

        opt.trajectory(objective=tiny_obj, auto_rescale_objective=True)
        assert 0 < opt._objective_rescale < 1.0

    def test_zero_objective_no_rescale(self, aircraft_type, short_flight):
        """If the objective at the initial guess is essentially zero,
        skip rescaling to avoid divide-by-zero.
        """
        opt = top.Cruise(
            aircraft_type,
            short_flight["origin"],
            short_flight["destination"],
            short_flight["m0"],
        )
        opt.setup(max_iter=500)

        def zero_obj(x, u, dt, **kwargs):
            return opt.obj_fuel(x, u, dt, **kwargs) * 0.0

        opt.trajectory(objective=zero_obj, auto_rescale_objective=True)
        assert opt._objective_rescale == 1.0
