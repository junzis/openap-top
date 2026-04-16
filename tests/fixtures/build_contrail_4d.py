# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "opentop",
#     "fastmeteo",
#     "scipy",
#     "openap",
# ]
# ///
"""Generate the contrail_4d.casadi fixture from ERA5 meteo data.

Builds a 4D CasADi bspline interpolant of ATR20 contrail persistence cost
(0/1 Gaussian-smoothed with sigma) over a rectangular lon/lat/alt/time grid
derived from ERA5 data for the given route and time window.

Usage (requires ERA5 Zarr store on disk or network):

    uv run tests/fixtures/build_contrail_4d.py \\
        --origin 52.362,13.501 --dest 40.472,-3.563 \\
        --start 2023-01-05T09:48 --stop 2023-01-05T12:00 \\
        --sigma 2 \\
        --out tests/fixtures/contrail_4d.casadi

The default ERA5 store path is /tmp/era5-zarr (ArcoEra5 will download
data from ARCO on first use, which may take some time for large grids).
Pass --era5-store to override.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Contrail physics helpers (ported verbatim from debug/epsilon_constraint/
# driver_epsilon_constraint.py, using openap.contrail + openap.aero)
# ---------------------------------------------------------------------------


def agg_conditions(flight):
    """Annotate a meteo DataFrame with contrail flags.

    Adds columns: rhi, crit_temp, sac, issr, persistent.

    Args:
        flight: DataFrame with columns temperature, specific_humidity,
            altitude (feet), and pressure-derived quantities resolvable
            via openap.aero.pressure(altitude_m).

    Returns:
        DataFrame with contrail condition columns appended.
    """
    from openap import aero, contrail

    f = flight.assign(
        rhi=lambda d: contrail.relative_humidity(
            d.specific_humidity,
            aero.pressure(d.altitude * aero.ft),
            d.temperature,
            to="ice",
        ),
        crit_temp=lambda d: contrail.critical_temperature_water(
            aero.pressure(d.altitude * aero.ft)
        ),
        sac=lambda d: d.temperature < d.crit_temp,
        issr=lambda d: d.rhi > 1,
        persistent=lambda d: d.sac & d.issr,
    )
    return f


# ---------------------------------------------------------------------------
# ERA5 grid builder
# ---------------------------------------------------------------------------


def build_meteo_grid(
    origin_lat: float,
    origin_lon: float,
    dest_lat: float,
    dest_lon: float,
    start: str,
    stop: str,
    time_buffer_hours: float = 3.0,
    era5_store: str = "/tmp/era5-zarr",
):
    """Pull ERA5 meteo for a bbox grid spanning origin→dest and start→stop.

    Computes a rectangular lat/lon/alt/time grid (same approach as
    WeatherData.interpolate in debug/epsilon_constraint/data.py), then
    queries ArcoEra5 to fill in the meteo fields.

    Args:
        origin_lat, origin_lon: Origin coordinates (degrees).
        dest_lat, dest_lon: Destination coordinates (degrees).
        start: ISO datetime string for window start (e.g. "2023-01-05T09:48").
        stop:  ISO datetime string for window end   (e.g. "2023-01-05T12:00").
        time_buffer_hours: Extra hours appended to the time window to cover
            longer-than-great-circle trajectories. Default 3.0 h.
        era5_store: Local Zarr store path for ArcoEra5.

    Returns:
        pd.DataFrame: Meteo grid with ERA5 fields interpolated at each point.
    """
    from fastmeteo.source import ArcoEra5

    import numpy as np
    import pandas as pd

    era5 = ArcoEra5(local_store=era5_store)

    stop_padded = pd.Timestamp(stop) + pd.Timedelta(hours=time_buffer_hours)
    timestamps = pd.date_range(start, stop_padded, freq="1h")

    latmin = math.ceil(min(origin_lat, dest_lat)) - 2
    latmax = math.ceil(max(origin_lat, dest_lat)) + 2
    lonmin = math.ceil(min(origin_lon, dest_lon)) - 4
    lonmax = math.ceil(max(origin_lon, dest_lon)) + 4

    latitudes = np.arange(latmin, latmax, 1)
    longitudes = np.arange(lonmin, lonmax, 1)
    # Same altitude grid as the debug pipeline: 1 000-46 000 ft in 1500-ft steps
    altitudes = np.arange(1000, 46000, 1500)

    latitudes, longitudes, altitudes, times = np.meshgrid(
        latitudes, longitudes, altitudes, timestamps
    )

    grid = pd.DataFrame().assign(
        latitude=latitudes.flatten(),
        longitude=longitudes.flatten(),
        altitude=altitudes.flatten(),
        timestamp=times.flatten(),
    )

    meteo = era5.interpolate(grid)
    return meteo


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def build_interpolant(
    origin_lat: float,
    origin_lon: float,
    dest_lat: float,
    dest_lon: float,
    start: str,
    stop: str,
    sigma: int,
    out: str,
    era5_store: str = "/tmp/era5-zarr",
):
    """Full pipeline: ERA5 → contrail flags → Gaussian smoothing → bspline.

    Args:
        origin_lat, origin_lon: Origin coordinates (degrees).
        dest_lat, dest_lon:     Destination coordinates (degrees).
        start: ISO datetime string for window start.
        stop:  ISO datetime string for window end.
        sigma: Gaussian smoothing sigma applied over (height, lat, lon) axes.
        out:   Output path for the .casadi interpolant file.
        era5_store: Local Zarr store path for ArcoEra5.
    """
    from openap import aero
    from scipy.ndimage import gaussian_filter

    import opentop.tools as tools

    print(f"Building contrail interpolant → {out}", file=sys.stderr)

    print(
        f"  Building rectangular ERA5 grid for "
        f"({origin_lat:.3f},{origin_lon:.3f})→({dest_lat:.3f},{dest_lon:.3f}) "
        f"{start} - {stop}",
        file=sys.stderr,
    )
    meteo = build_meteo_grid(
        origin_lat,
        origin_lon,
        dest_lat,
        dest_lon,
        start,
        stop,
        era5_store=era5_store,
    )

    print(f"  Meteo grid: {len(meteo):,} rows", file=sys.stderr)

    # ------------------------------------------------------------------
    # 2. Compute contrail conditions
    # ------------------------------------------------------------------
    print("  Computing contrail conditions…", file=sys.stderr)
    contrail_df = agg_conditions(meteo)

    # ------------------------------------------------------------------
    # 3. Reshape cost into (ts, height, lat, lon) and smooth
    # ------------------------------------------------------------------
    print(
        f"  Applying Gaussian filter sigma=(0, {sigma}, {sigma}, {sigma})…",
        file=sys.stderr,
    )
    df_cost = contrail_df.assign(
        height=lambda x: x.altitude * aero.ft,
        cost=lambda x: x.persistent.astype(float),
        ts=lambda x: (x.timestamp - x.timestamp.iloc[0]).dt.total_seconds(),
    ).sort_values(["ts", "height", "latitude", "longitude"])

    cost_array = df_cost.cost.values.reshape(
        df_cost.ts.nunique(),
        df_cost.height.nunique(),
        df_cost.latitude.nunique(),
        df_cost.longitude.nunique(),
    )
    cost_smoothed = gaussian_filter(
        cost_array, sigma=(0, sigma, sigma, sigma), mode="nearest"
    )
    df_cost = df_cost.assign(cost=cost_smoothed.flatten()).fillna(0)

    # ------------------------------------------------------------------
    # 4. Build bspline interpolant and save
    # ------------------------------------------------------------------
    print("  Building bspline interpolant…", file=sys.stderr)
    interpolant = tools.interpolant_from_dataframe(df_cost)

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tools.save_interpolant(interpolant, out_path)
    print(
        f"  Saved → {out_path} ({out_path.stat().st_size / 1e6:.2f} MB)",
        file=sys.stderr,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_latlon(s: str, name: str) -> tuple[float, float]:
    """Parse a 'lat,lon' string into a (lat, lon) float pair."""
    parts = s.split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            f"{name} must be 'lat,lon' (e.g. 52.362,13.501), got {s!r}"
        )
    try:
        return float(parts[0]), float(parts[1])
    except ValueError:
        raise argparse.ArgumentTypeError(f"{name} values must be numeric, got {s!r}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate a 4D CasADi contrail interpolant from ERA5 meteo data.\n\n"
            "Example:\n"
            "  uv run tests/fixtures/build_contrail_4d.py \\\n"
            "    --origin 52.362,13.501 --dest 40.472,-3.563 \\\n"
            "    --start 2023-01-05T09:48 --stop 2023-01-05T12:00 \\\n"
            "    --sigma 2 --out tests/fixtures/contrail_4d.casadi"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--origin",
        required=True,
        metavar="LAT,LON",
        help="Origin coordinates as 'lat,lon' (e.g. 52.362,13.501 for EDDB).",
    )
    parser.add_argument(
        "--dest",
        required=True,
        metavar="LAT,LON",
        help="Destination coordinates as 'lat,lon' (e.g. 40.472,-3.563 for LEMD).",
    )
    parser.add_argument(
        "--start",
        required=True,
        metavar="DATETIME",
        help="Time-window start as ISO datetime (e.g. 2023-01-05T09:48).",
    )
    parser.add_argument(
        "--stop",
        required=True,
        metavar="DATETIME",
        help="Time-window end as ISO datetime (e.g. 2023-01-05T12:00).",
    )
    parser.add_argument(
        "--sigma",
        type=int,
        default=2,
        help="Gaussian smoothing sigma applied over (height, lat, lon). Default: 2.",
    )
    parser.add_argument(
        "--out",
        default="tests/fixtures/contrail_4d.casadi",
        metavar="PATH",
        help=(
            "Output path for the .casadi file. "
            "Default: tests/fixtures/contrail_4d.casadi."
        ),
    )
    parser.add_argument(
        "--era5-store",
        default="/tmp/era5-zarr",
        metavar="PATH",
        help=(
            "Local Zarr store path for ArcoEra5. "
            "Default: /tmp/era5-zarr. "
            "First run downloads ERA5 data (~GB) from ARCO."
        ),
    )

    args = parser.parse_args()

    origin_lat, origin_lon = _parse_latlon(args.origin, "--origin")
    dest_lat, dest_lon = _parse_latlon(args.dest, "--dest")

    build_interpolant(
        origin_lat=origin_lat,
        origin_lon=origin_lon,
        dest_lat=dest_lat,
        dest_lon=dest_lon,
        start=args.start,
        stop=args.stop,
        sigma=args.sigma,
        out=args.out,
        era5_store=args.era5_store,
    )


if __name__ == "__main__":
    main()
