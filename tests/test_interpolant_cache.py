"""Tests for the interpolant cache utilities in openap.top.tools."""

import casadi as ca
import numpy as np
import pandas as pd
import pytest

from openap.top import tools


def _make_grid(n_lat=5, n_lon=5, n_h=4, n_ts=3) -> pd.DataFrame:
    """Build a tiny 4D deterministic cost grid for unit tests."""
    lats = np.linspace(40.0, 50.0, n_lat)
    lons = np.linspace(-5.0, 5.0, n_lon)
    heights = np.linspace(6000.0, 12000.0, n_h)
    tss = np.linspace(0.0, 3600.0, n_ts)

    rows = []
    for ts in tss:
        for h in heights:
            for lat in lats:
                for lon in lons:
                    # Deterministic non-trivial cost field (smooth in h, lat, lon)
                    cost = 0.5 * np.exp(
                        -((h - 10000) ** 2) / 1e7
                        - ((lat - 45) ** 2) / 10
                        - ((lon - 0) ** 2) / 10
                    )
                    rows.append(
                        dict(
                            latitude=lat,
                            longitude=lon,
                            height=h,
                            ts=ts,
                            cost=cost,
                        )
                    )
    return pd.DataFrame(rows)


class TestInterpolantCache:
    """Save/load/cache tests using a tiny linear interpolant (fast to build)."""

    def test_save_and_load_roundtrip(self, tmp_path):
        df = _make_grid()
        original = tools.interpolant_from_dataframe(df, shape="linear")
        cache_file = tmp_path / "grid.casadi"

        tools.save_interpolant(original, cache_file)
        assert cache_file.exists()
        assert cache_file.stat().st_size > 0

        loaded = tools.load_interpolant(cache_file)

        # Verify loaded interpolant produces the same values as the original
        # at several test points.
        for lon, lat, h, ts in [
            (0.0, 45.0, 10000.0, 1800.0),
            (-2.5, 42.0, 7000.0, 0.0),
            (3.0, 48.0, 11000.0, 3600.0),
        ]:
            original_val = float(original(ca.vertcat(lon, lat, h, ts)))
            loaded_val = float(loaded(ca.vertcat(lon, lat, h, ts)))
            assert original_val == pytest.approx(loaded_val, rel=1e-10)

    def test_save_creates_parent_dirs(self, tmp_path):
        df = _make_grid()
        interp = tools.interpolant_from_dataframe(df, shape="linear")
        nested = tmp_path / "a" / "b" / "c" / "grid.casadi"

        tools.save_interpolant(interp, nested)
        assert nested.exists()

    def test_load_raises_on_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            tools.load_interpolant(tmp_path / "does_not_exist.casadi")

    def test_cached_interpolant_builds_on_first_call(self, tmp_path):
        df = _make_grid()
        cache_file = tmp_path / "grid.casadi"
        assert not cache_file.exists()

        interp = tools.cached_interpolant_from_dataframe(
            df, cache_file, shape="linear"
        )
        assert cache_file.exists()

        val = float(interp(ca.vertcat(0.0, 45.0, 10000.0, 1800.0)))
        assert val > 0  # cost field peaks near (lat=45, lon=0, h=10000)

    def test_cached_interpolant_reuses_cache_on_second_call(self, tmp_path):
        df = _make_grid()
        cache_file = tmp_path / "grid.casadi"

        # First call: builds and saves
        first = tools.cached_interpolant_from_dataframe(
            df, cache_file, shape="linear"
        )
        mtime_first = cache_file.stat().st_mtime

        # Second call: should load without rebuilding (cache file untouched)
        second = tools.cached_interpolant_from_dataframe(
            df, cache_file, shape="linear"
        )
        mtime_second = cache_file.stat().st_mtime
        assert mtime_first == mtime_second

        # Values should match
        point = ca.vertcat(1.0, 44.0, 9000.0, 1800.0)
        assert float(first(point)) == pytest.approx(float(second(point)), rel=1e-10)

    def test_cached_interpolant_supports_3d_grid(self, tmp_path):
        """Grid without a 'ts' column should still cache correctly."""
        df = _make_grid(n_ts=1).drop(columns=["ts"])
        cache_file = tmp_path / "grid3d.casadi"

        first = tools.cached_interpolant_from_dataframe(
            df, cache_file, shape="linear"
        )
        assert cache_file.exists()

        second = tools.cached_interpolant_from_dataframe(
            df, cache_file, shape="linear"
        )
        point = ca.vertcat(0.0, 45.0, 10000.0)
        assert float(first(point)) == pytest.approx(float(second(point)), rel=1e-10)
