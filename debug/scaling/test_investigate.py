"""Unit tests for investigate.py pure helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from investigate import slice_grid  # noqa: E402


def _make_grid_fixture() -> pd.DataFrame:
    """4 timestamps x 3 heights x 5 lats x 5 lons = 300 rows."""
    rows = []
    for ts_idx, ts_sec in enumerate([0, 3600, 7200, 10800]):
        timestamp = pd.Timestamp("2022-02-20 10:00:00", tz="UTC") + pd.Timedelta(
            seconds=ts_sec
        )
        for h in [9000.0, 10000.0, 11000.0]:
            for lat in [40.0, 42.0, 44.0, 46.0, 48.0]:
                for lon in [-5.0, 0.0, 5.0, 10.0, 15.0]:
                    rows.append(
                        dict(
                            timestamp=timestamp,
                            ts=float(ts_sec),
                            height=h,
                            latitude=lat,
                            longitude=lon,
                            cost=float(ts_idx + h / 1000 + lat + lon),
                        )
                    )
    return pd.DataFrame(rows)


def test_slice_grid_filters_by_bbox_and_rebases_ts():
    df = _make_grid_fixture()
    sliced = slice_grid(
        df,
        t0="2022-02-20 10:00:00+00:00",
        t1="2022-02-20 12:00:00+00:00",
        lat_min=41.0,
        lat_max=47.0,
        lon_min=-1.0,
        lon_max=11.0,
    )
    # time: 3 timestamps (0s, 3600s, 7200s); lats in (41, 47): 42, 44, 46; lons in (-1, 11): 0, 5, 10
    assert sliced.latitude.unique().tolist() == [42.0, 44.0, 46.0]
    assert sorted(sliced.longitude.unique().tolist()) == [0.0, 5.0, 10.0]
    assert sliced.ts.min() == 0.0
    assert sliced.ts.max() == 7200.0
    assert sliced.ts.nunique() == 3
    # Required columns for top.tools.interpolant_from_dataframe
    for col in ("ts", "height", "latitude", "longitude", "cost"):
        assert col in sliced.columns


def test_slice_grid_rebases_ts_relative_to_t0():
    df = _make_grid_fixture()
    sliced = slice_grid(
        df,
        t0="2022-02-20 11:00:00+00:00",  # ts=3600 in original
        t1="2022-02-20 13:00:00+00:00",  # ts=10800 in original
        lat_min=40.0,
        lat_max=48.0,
        lon_min=-5.0,
        lon_max=15.0,
    )
    # After rebase: smallest ts should be 0
    assert sliced.ts.min() == 0.0
    # Gap between timestamps preserved: 3600s
    assert sorted(sliced.ts.unique().tolist()) == [0.0, 3600.0, 7200.0]
