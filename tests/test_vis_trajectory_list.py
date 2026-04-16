"""Tests for vis.trajectory accepting a list of DataFrames."""

import matplotlib

matplotlib.use("Agg")  # headless for CI

import numpy as np
import opentop.vis as vis
import pandas as pd


def _toy_df(alt0=30_000, alt1=35_000, lon0=13.5, lon1=-3.6, lat0=52.4, lat1=40.5):
    n = 30
    return pd.DataFrame(
        {
            "ts": np.linspace(0, 7200, n),
            "altitude": np.linspace(alt0, alt1, n),
            "tas": np.full(n, 450.0),
            "vertical_rate": np.zeros(n),
            "longitude": np.linspace(lon0, lon1, n),
            "latitude": np.linspace(lat0, lat1, n),
        }
    )


def test_trajectory_accepts_single_df():
    df = _toy_df()
    plt = vis.trajectory(df)
    assert plt is not None
    plt.close("all")


def test_trajectory_accepts_list_of_dfs():
    df_a = _toy_df(alt1=36_000)
    df_b = _toy_df(alt1=34_000)
    plt = vis.trajectory([df_a, df_b])
    assert plt is not None
    plt.close("all")


def test_trajectory_list_with_custom_labels():
    df_a = _toy_df()
    df_b = _toy_df(alt1=33_000)
    plt = vis.trajectory([df_a, df_b], labels=["actual", "optimized"])
    assert plt is not None
    plt.close("all")


def test_trajectory_empty_list_raises():
    import pytest

    with pytest.raises(ValueError, match="at least one"):
        vis.trajectory([])
