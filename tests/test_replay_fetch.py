"""Tests for opentop.replay.fetch_flight."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import numpy as np
import pandas as pd
from opentop import replay

FIXTURE = Path(__file__).parent / "fixtures" / "flight_ryr880w_2023-01-05.parquet"


def _fake_opensky_flight_df():
    """Minimal traffic-compatible DataFrame."""
    n = 20
    return pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2023-01-05 09:48", periods=n, freq="1min", tz="UTC"
            ),
            "latitude": np.linspace(52.4, 40.5, n),
            "longitude": np.linspace(13.5, -3.6, n),
            "altitude": np.linspace(10000, 35000, n),
            "icao24": "4bb9b1",
            "callsign": "RYR880W",
            "last_position": pd.Timestamp("2023-01-05 09:48", tz="UTC"),
            "onground": False,
        }
    )


def test_fetch_flight_opensky_happy_path():
    # traffic 2.13 exposes opensky lazily via __getattr__; accessing it on pandas ≥ 2.0
    # raises ImportError (DatetimeTZBlock removed).  Probe both the package and the
    # attribute so this test skips cleanly in broken environments.
    _traffic_data = pytest.importorskip("traffic.data")
    try:
        _traffic_data.opensky
    except Exception as _exc:
        pytest.skip(f"traffic.data.opensky unavailable: {_exc}")

    fake_flight = MagicMock()
    fake_flight.data = _fake_opensky_flight_df()

    with patch("traffic.data.opensky.history", return_value=fake_flight):
        df = replay.fetch_flight(
            callsign="RYR880W",
            start="2023-01-05 09:00",
            stop="2023-01-05 13:00",
            source="opensky",
        )

    assert {"timestamp", "latitude", "longitude", "altitude", "icao24"}.issubset(
        df.columns
    )
    assert len(df) > 0
    assert df["icao24"].iloc[0] == "4bb9b1"


def test_fetch_flight_opensky_no_results_raises():
    _traffic_data = pytest.importorskip("traffic.data")
    try:
        _traffic_data.opensky
    except Exception as _exc:
        pytest.skip(f"traffic.data.opensky unavailable: {_exc}")

    with patch("traffic.data.opensky.history", return_value=None):
        with pytest.raises(ValueError, match="No flight data"):
            replay.fetch_flight(
                callsign="NOSUCH",
                start="2023-01-05 00:00",
                stop="2023-01-05 23:59",
                source="opensky",
            )


def test_fetch_flight_opensky_picks_main_icao24_when_multiple():
    _traffic_data = pytest.importorskip("traffic.data")
    try:
        _traffic_data.opensky
    except Exception as _exc:
        pytest.skip(f"traffic.data.opensky unavailable: {_exc}")

    df_mixed = _fake_opensky_flight_df()
    # Inject a few rows with a different icao24 (minority → should be dropped)
    df_mixed.loc[df_mixed.index[:3], "icao24"] = "cafe00"
    fake_flight = MagicMock()
    fake_flight.data = df_mixed

    with patch("traffic.data.opensky.history", return_value=fake_flight):
        df = replay.fetch_flight(
            callsign="RYR880W",
            start="2023-01-05 09:00",
            stop="2023-01-05 13:00",
            source="opensky",
        )

    assert (df["icao24"] == "4bb9b1").all()


def test_fetch_flight_from_parquet_fixture():
    """Loading the committed OpenSky fixture returns a sane DataFrame."""
    if not FIXTURE.exists():
        pytest.skip(f"Missing fixture {FIXTURE}; regenerate via OpenSky.")

    df = replay.fetch_flight(
        callsign="RYR880W",
        start="2023-01-05 09:00",
        stop="2023-01-05 13:00",
        source=FIXTURE,
    )

    assert {"timestamp", "latitude", "longitude", "altitude", "icao24"}.issubset(
        df.columns
    )
    assert (df["callsign"].str.strip() == "RYR880W").all()
    assert df["icao24"].nunique() == 1, "expected exactly one icao24 after filtering"
    assert df.altitude.max() > 30_000, "cruise altitude should be above FL300"


def test_fetch_flight_from_file_missing_raises():
    with pytest.raises(FileNotFoundError):
        replay.fetch_flight(
            callsign="X",
            start="2023-01-05 00:00",
            stop="2023-01-05 23:59",
            source=Path("/tmp/does_not_exist_12345.parquet"),
        )
