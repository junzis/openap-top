"""Tests for opentop.replay.build_contrail_interpolant."""

from __future__ import annotations

import pytest

import numpy as np
import pandas as pd

pytest.importorskip("scipy")

import casadi as ca

from opentop import replay


def _fake_meteo_df():
    """8x8x8x4 grid — satisfies bspline minimum-points requirement."""
    lons = np.linspace(0, 14, 8)
    lats = np.linspace(40, 54, 8)
    alts = np.linspace(20000, 44000, 8)  # feet, within contrail band
    times = pd.date_range("2023-01-05 10:00", periods=4, freq="1h", tz="UTC")

    rows = []
    for t in times:
        for alt in alts:
            for lat in lats:
                for lon in lons:
                    rows.append(
                        {
                            "timestamp": t,
                            "latitude": lat,
                            "longitude": lon,
                            "altitude": alt,
                            "temperature": 220.0,
                            "specific_humidity": 3e-4,
                        }
                    )
    return pd.DataFrame(rows)


def test_build_contrail_interpolant_returns_casadi_function():
    meteo = _fake_meteo_df()
    interp = replay.build_contrail_interpolant(meteo, sigma=2)

    assert isinstance(interp, ca.Function)
    assert interp.n_in() == 1
    assert interp.sparsity_in(0).size1() == 4


def test_build_contrail_interpolant_finite_at_sample_point():
    meteo = _fake_meteo_df()
    interp = replay.build_contrail_interpolant(meteo, sigma=2)
    # Sample inside bbox: lon=7, lat=47, h=10000m, ts=3600s
    value = float(interp([7.0, 47.0, 10_000, 3600]))
    assert np.isfinite(value)
