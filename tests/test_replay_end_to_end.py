"""End-to-end replay integration test.

Uses the committed flight_ryr880w_2023-01-05.parquet and contrail_4d.casadi
fixtures to exercise the full replay→optimize chain without hitting OpenSky
or ERA5. Bypasses `build_meteo_and_wind` (the ERA5 fetch) by loading the
pre-built interpolant directly; unit tests cover that function with mocks.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import opentop as top
from opentop import replay, tools

FIXTURES = Path(__file__).parent / "fixtures"
FLIGHT = FIXTURES / "flight_ryr880w_2023-01-05.parquet"
INTERP = FIXTURES / "contrail_4d.casadi"


@pytest.fixture(scope="module")
def flight_df():
    if not FLIGHT.exists():
        pytest.skip(f"Missing fixture {FLIGHT}; see tests/fixtures/README.md")
    df = replay.fetch_flight(
        callsign="RYR880W",
        start="2023-01-05 09:00",
        stop="2023-01-05 13:00",
        source=FLIGHT,
    )
    # Filter out OpenSky altitude outliers (spurious baro_altitude spikes)
    # and keep only proper cruise-phase samples.
    df = df[(df["altitude"] > 20000) & (df["altitude"] < 45000)].reset_index(drop=True)
    return df


@pytest.fixture(scope="module")
def interpolant():
    if not INTERP.exists():
        pytest.skip(f"Missing fixture {INTERP}; see tests/fixtures/README.md")
    return tools.load_interpolant(str(INTERP))


def test_replay_end_to_end_cruise(flight_df, interpolant):
    """Replay a real flight with a blended fuel+contrail objective."""
    lat0 = float(flight_df["latitude"].iloc[0])
    lon0 = float(flight_df["longitude"].iloc[0])
    lat1 = float(flight_df["latitude"].iloc[-1])
    lon1 = float(flight_df["longitude"].iloc[-1])

    opt = top.Cruise("B738", (lat0, lon0), (lat1, lon1), m0=0.85)
    opt.setup(max_iter=500)

    def blended(x, u, dt, **kwargs):
        grid = opt.obj_grid_cost(
            x,
            u,
            dt,
            interpolant=kwargs["interpolant"],
            n_dim=4,
            time_dependent=True,
        )
        fuel = opt.obj_fuel(x, u, dt)
        return grid + 0.1 * fuel

    df = opt.trajectory(
        objective=blended,
        interpolant=interpolant,
        n_dim=4,
        time_dependent=True,
    )

    assert df is not None, "optimizer returned None"
    assert opt.success, f"solver failed: {opt.stats}"

    fuel_kg = float(df.mass.iloc[0] - df.mass.iloc[-1])
    assert 2000 < fuel_kg < 12000, f"fuel {fuel_kg:.0f} kg out of sane range"

    # Trajectory stays inside the interpolant's domain.
    assert df.longitude.min() > -12, f"lon min {df.longitude.min()} below grid"
    assert df.longitude.max() < 17, f"lon max {df.longitude.max()} above grid"
    assert df.latitude.min() > 36, f"lat min {df.latitude.min()} below grid"
    assert df.latitude.max() < 57, f"lat max {df.latitude.max()} above grid"

    assert "grid_cost" in df.columns
