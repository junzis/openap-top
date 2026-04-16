"""Tests for opentop.replay.normalize_flight_for_vis."""

from __future__ import annotations

import numpy as np
import pandas as pd
from opentop import replay


def test_normalize_adds_ts_tas_and_vertical_rate():
    n = 10
    raw = pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2023-01-05 09:00", periods=n, freq="1min", tz="UTC"
            ),
            "latitude": np.linspace(52.4, 40.5, n),
            "longitude": np.linspace(13.5, -3.6, n),
            "altitude": np.full(n, 35000),
            "groundspeed": np.full(n, 450.0),
        }
    )

    out = replay.normalize_flight_for_vis(raw)
    assert "ts" in out.columns and out["ts"].iloc[0] == 0
    assert "tas" in out.columns and (out["tas"] == 450.0).all()
    assert "vertical_rate" in out.columns and (out["vertical_rate"] == 0.0).all()


def test_normalize_preserves_existing_columns():
    n = 5
    raw = pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2023-01-05 09:00", periods=n, freq="1min", tz="UTC"
            ),
            "ts": np.linspace(0, 300, n),
            "tas": np.full(n, 480.0),
            "vertical_rate": np.full(n, 1000.0),
            "latitude": np.zeros(n),
            "longitude": np.zeros(n),
        }
    )

    out = replay.normalize_flight_for_vis(raw)
    assert (out["ts"] == raw["ts"]).all()
    assert (out["tas"] == 480.0).all()
    assert (out["vertical_rate"] == 1000.0).all()


def test_normalize_does_not_mutate_input():
    n = 5
    raw = pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2023-01-05 09:00", periods=n, freq="1min", tz="UTC"
            ),
            "latitude": np.zeros(n),
            "longitude": np.zeros(n),
            "altitude": np.full(n, 35000),
            "groundspeed": np.full(n, 450.0),
        }
    )
    before_cols = set(raw.columns)
    _ = replay.normalize_flight_for_vis(raw)
    assert set(raw.columns) == before_cols
