"""Tests for waypoint-constrained trajectories."""

from math import pi
from typing import cast

import openap
import pytest
from openap.aero import fpm

import opentop as top
import pandas as pd


def _midpoint_waypoint(opt):
    xp_0, yp_0 = opt.proj(opt.lon1, opt.lat1)
    xp_f, yp_f = opt.proj(opt.lon2, opt.lat2)
    lon, lat = opt.proj((xp_0 + xp_f) / 2, (yp_0 + yp_f) / 2, inverse=True)
    return lat, lon


def test_cruise_waypoint_solution_passes_near_waypoint(aircraft_type, short_flight):
    opt = top.Cruise(
        aircraft_type,
        short_flight["origin"],
        short_flight["destination"],
        short_flight["m0"],
    )
    waypoint = _midpoint_waypoint(opt)

    df = cast(
        pd.DataFrame,
        opt.trajectory(
            objective="fuel",
            waypoints=[waypoint],
            waypoint_tolerance_m=10_000,
        ),
    )
    assert isinstance(df, pd.DataFrame)

    distances = [
        openap.aero.distance(lat, lon, waypoint[0], waypoint[1])
        for lat, lon in zip(df.latitude, df.longitude)
    ]
    assert min(distances) <= 10_500
    assert df.ts.diff().dropna().gt(0).all()
    vertical_acceleration = (df.vertical_rate.diff() / df.ts.diff()).dropna()
    assert vertical_acceleration.abs().max() <= 5.1


def test_waypoint_variable_timestep_supports_time_objective(
    monkeypatch, aircraft_type, short_flight
):
    opt = top.Cruise(
        aircraft_type,
        short_flight["origin"],
        short_flight["destination"],
        short_flight["m0"],
    )
    waypoint = _midpoint_waypoint(opt)

    class FakeSolution:
        def stats(self):
            return {"success": True}

    def fake_solve(X, U, **kwargs):
        opt._last_solution = FakeSolution()
        return pd.DataFrame(
            {
                "altitude": [30_000.0, 30_000.0],
                "mass": [opt.mass_init, opt.mass_init - 1.0],
            }
        )

    monkeypatch.setattr(opt, "_solve", fake_solve)

    df = opt.trajectory(objective="time", waypoints=[waypoint])

    assert df is not None
    assert opt._variable_timestep is True
    assert len(opt._interval_dts) == opt.nodes


def test_default_waypoint_nodes_follow_route_distance():
    opt = top.CompleteFlight("A320", "EHAM", "LIRF", 0.85)
    waypoints = []
    for fix_name in ("SIGEN", "FUSSE", "ROKIB"):
        lat, lon, _ = openap.nav.fix(fix_name)
        waypoints.append((float(lat), float(lon)))

    assert opt._waypoint_node_indices(waypoints) == [7, 15, 20]


def test_explicit_setup_nodes_can_be_below_auto_minimum():
    opt = top.Cruise("A320", "EHAM", "LIRF", 0.85)

    opt.setup(nodes=15)

    assert opt.nodes == 15


def test_explicit_setup_nodes_can_exceed_auto_maximum():
    opt = top.Cruise("A320", "EHAM", "LIRF", 0.85)

    opt.setup(nodes=121)

    assert opt.nodes == 121


def test_variable_timestep_default_bounds_are_moderately_relaxed():
    opt = top.CompleteFlight("A320", "EHAM", "LIRF", 0.85)
    opt.init_conditions()

    dt_min, dt_max, dt_guess = opt._variable_timestep_bounds(7200)
    interval_guess = 7200 / opt.nodes

    assert dt_min == pytest.approx(0.65 * interval_guess)
    assert dt_max == pytest.approx(1.65 * interval_guess)
    assert dt_guess == pytest.approx(interval_guess)


def test_control_change_rate_uses_each_interval_duration():
    opt = top.Cruise("A320", "EHAM", "LIRF", 0.85)
    opt._variable_timestep = True
    opt._interval_dts = [10.0, 20.0]
    controls = [
        [0.7, 0.0, 0.0],
        [0.7, 50 * fpm, 5 * pi / 180],
        [0.7, 100 * fpm, 10 * pi / 180],
    ]

    assert opt._control_change_rate(controls, 0, 1) == pytest.approx(5 * fpm)
    assert opt._control_change_rate(controls, 1, 1) == pytest.approx(2.5 * fpm)
    assert opt._control_change_rate(controls, 0, 2) == pytest.approx(0.5 * pi / 180)


@pytest.mark.parametrize(
    ("dt_min", "dt_max"),
    [
        (0.0, 100.0),
        (-1.0, 100.0),
        (100.0, 0.0),
        (100.0, -1.0),
        (100.0, 50.0),
    ],
)
def test_variable_timestep_bounds_must_be_positive_and_ordered(dt_min, dt_max):
    opt = top.CompleteFlight("A320", "EHAM", "LIRF", 0.85)
    opt.init_conditions()

    with pytest.raises(ValueError, match="timestep bounds"):
        opt._variable_timestep_bounds(7200, dt_min=dt_min, dt_max=dt_max)


def test_waypoint_latitude_is_validated(aircraft_type, short_flight):
    opt = top.Cruise(
        aircraft_type,
        short_flight["origin"],
        short_flight["destination"],
        short_flight["m0"],
    )

    with pytest.raises(ValueError, match="latitude"):
        opt.trajectory(objective="fuel", waypoints=[(95.0, 0.0)])
