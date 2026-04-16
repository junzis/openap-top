"""Tests for opentop.replay.build_meteo_and_wind."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

import numpy as np
import pandas as pd
from opentop import replay


def _fake_flight_df():
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
        }
    )


def _fake_meteo_df(n=100):
    """Minimal ERA5-like DataFrame with the columns the wind extractor needs."""
    return pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2023-01-05 09:00", periods=n, freq="15min", tz="UTC"
            ),
            "latitude": np.linspace(40, 54, n),
            "longitude": np.linspace(-5, 15, n),
            "altitude": np.linspace(1000, 45000, n),
            "temperature": np.full(n, 220.0),
            "specific_humidity": np.full(n, 1e-4),
            "u_component_of_wind": np.full(n, 10.0),
            "v_component_of_wind": np.full(n, 2.0),
        }
    )


def test_build_meteo_and_wind_returns_two_dataframes():
    pytest.importorskip("fastmeteo")
    fake_era5 = MagicMock()
    fake_era5.interpolate.return_value = _fake_meteo_df()

    with patch("fastmeteo.source.ArcoEra5", return_value=fake_era5):
        meteo, wind = replay.build_meteo_and_wind(
            _fake_flight_df(),
            era5_store="/tmp/opentop-era5-test",
        )

    assert "temperature" in meteo.columns
    assert "specific_humidity" in meteo.columns

    for col in ("ts", "latitude", "longitude", "h", "u", "v"):
        assert col in wind.columns, f"wind missing column {col}"
    assert (wind["h"] > 0).all(), "h should be metres"
    assert (wind["ts"] >= 0).all(), "ts should start at zero"
