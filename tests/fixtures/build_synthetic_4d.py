"""Build a tiny synthetic 4D cost grid and cache it to synthetic_4d.casadi.

Run once:
    uv run --with '.' python tests/fixtures/build_synthetic_4d.py

The output should be well under 2 MB — small enough to commit.
"""

from pathlib import Path

import numpy as np
import pandas as pd

import opentop.tools as tools


def build():
    lons = np.linspace(0.0, 15.0, 8)
    lats = np.linspace(48.0, 55.0, 8)
    heights = np.linspace(3000.0, 12000.0, 8)
    tss = np.linspace(0.0, 3600.0, 8)

    rows = []
    for lon in lons:
        for lat in lats:
            for h in heights:
                for ts in tss:
                    # Smooth synthetic cost: a Gaussian bubble centered at
                    # (lon=7.5, lat=51), zero outside the 5-11km band.
                    if h < 5000 or h > 11000:
                        c = 0.0
                    else:
                        c = 0.5 * np.exp(-((lon - 7.5) ** 2 + (lat - 51) ** 2) / 10)
                    rows.append(
                        {
                            "longitude": lon,
                            "latitude": lat,
                            "height": h,
                            "ts": ts,
                            "cost": c,
                        }
                    )
    df = pd.DataFrame(rows)
    out = Path(__file__).parent / "synthetic_4d.casadi"
    tools.cached_interpolant_from_dataframe(df, str(out), shape="bspline")
    print(f"wrote {out} ({out.stat().st_size / 1024:.1f} kB)")


if __name__ == "__main__":
    build()
