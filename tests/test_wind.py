"""End-to-end wind integration test.

Pins the Cruise.enable_wind → PolyWind → xdot wind branch. Uses a synthetic
constant-tailwind field so the solver's behaviour is predictable.
"""
import numpy as np
import pandas as pd
import pytest

import opentop as top


@pytest.fixture(scope="module")
def constant_tailwind_df():
    """Constant 10 m/s eastward, 0 m/s northward, covering an EHAM-EDDF
    bounding box at multiple altitudes and times."""
    rows = []
    for lon in np.linspace(0, 15, 5):
        for lat in np.linspace(48, 55, 5):
            for h in (1000, 5000, 10000, 12000):
                for ts in (0, 18000, 36000):
                    rows.append({
                        "longitude": lon, "latitude": lat, "h": h,
                        "ts": ts, "u": 10.0, "v": 0.0,
                    })
    return pd.DataFrame(rows)


def test_cruise_with_constant_tailwind_converges(constant_tailwind_df):
    """Cruise with a steady 10 m/s tailwind must solve cleanly."""
    opt = top.Cruise("A320", (52.308, 4.764), (50.033, 8.570), m0=0.85)
    opt.setup(max_iter=500)
    opt.enable_wind(constant_tailwind_df)
    df = opt.trajectory(objective="fuel")

    assert df is not None, "trajectory returned None"
    assert opt.success, f"solver failed: {opt.stats}"
    # Realistic cruise altitude (above FL300 / ~9100 m).
    assert df["altitude"].max() > 9_000 * 3.28084, \
        f"max altitude too low: {df['altitude'].max()} ft"
