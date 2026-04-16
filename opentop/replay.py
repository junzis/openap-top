"""opentop replay: business logic for fetching real flights and deriving
optimizer-ready inputs.

Functions in this module are pure-ish and reusable — the CLI wrapper in
`opentop/cli/replay.py` is a thin click layer around these.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Union

import pandas as pd

FlightSource = Union[Literal["opensky"], str, Path]


def fetch_flight(
    callsign: str,
    start: Union[str, pd.Timestamp],
    stop: Union[str, pd.Timestamp],
    source: FlightSource = "opensky",
) -> pd.DataFrame:
    """Return a cleaned flight DataFrame for CALLSIGN in the given window.

    Columns: timestamp, latitude, longitude, altitude, icao24, callsign.

    Args:
        callsign: Aircraft callsign (e.g. "RYR880W").
        start: Window start (ISO string or Timestamp).
        stop:  Window end.
        source: "opensky" to query OpenSky live, or a path to a
            Traffic-compatible parquet/csv.

    Raises:
        ValueError: if no flight data is returned.
        ImportError: if `traffic` is not installed (for source="opensky").
        FileNotFoundError: if source is a path that doesn't exist.
    """
    if source == "opensky":
        return _fetch_from_opensky(callsign, start, stop)
    return _fetch_from_file(Path(source), callsign)


def _fetch_from_opensky(
    callsign: str,
    start: Union[str, pd.Timestamp],
    stop: Union[str, pd.Timestamp],
) -> pd.DataFrame:
    try:
        from traffic.data import opensky
    except ImportError as exc:
        raise ImportError(
            "opentop replay requires the `traffic` package. "
            'Install with: pip install "opentop[replay]"'
        ) from exc

    flight = opensky.history(start, stop, callsign=callsign, return_flight=True)
    if flight is None or flight.data.empty:
        raise ValueError(
            f"No flight data retrieved for callsign {callsign!r} "
            f"in window {start} - {stop}."
        )

    df = flight.data

    # If multiple ICAO24s came back, pick the most-frequent one.
    counts = df["icao24"].value_counts()
    main_icao24 = counts.idxmax()
    df = df[df["icao24"] == main_icao24].copy()

    # Drop traffic-specific columns that are optional downstream.
    df = df.drop(columns=["last_position", "onground"], errors="ignore")
    df = df[df["latitude"].notnull()].reset_index(drop=True)

    if df.empty:
        raise ValueError("Flight has no valid trajectory points after filtering.")

    return df


def _fetch_from_file(path: Path, callsign: str | None = None) -> pd.DataFrame:
    """Load a saved Traffic-compatible flight file (parquet/csv).

    If `callsign` is given, filter to matching rows; otherwise return all rows.
    """
    if not path.exists():
        raise FileNotFoundError(f"Flight file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".parquet":
        df = pd.read_parquet(path)
    elif suffix == ".csv":
        df = pd.read_csv(path, parse_dates=["timestamp"])
    else:
        raise ValueError(
            f"Unsupported flight file format {suffix!r}. Use .parquet or .csv."
        )

    if callsign and "callsign" in df.columns:
        df = df[df["callsign"].str.strip() == callsign].copy()
        if df.empty:
            raise ValueError(f"No rows with callsign={callsign!r} in {path}.")

    return df.reset_index(drop=True)  # type: ignore[return-value]


def infer_aircraft(flight_df: pd.DataFrame) -> str | None:
    """Map the flight's primary ICAO24 to an aircraft type (e.g. 'B738').

    Returns None if the lookup fails, the icao24 is unknown, or the
    `traffic` package is not installed.
    """
    if "icao24" not in flight_df.columns or flight_df.empty:
        return None

    icao24 = flight_df["icao24"].iloc[0]

    try:
        from traffic.data import aircraft
    except ImportError:
        return None

    try:
        matches = aircraft.get(icao24)
    except Exception:
        return None

    if matches is None or (hasattr(matches, "empty") and matches.empty):
        return None
    if "typecode" not in matches.columns:
        return None

    typecode = matches["typecode"].iloc[0]
    if pd.isna(typecode):
        return None
    return str(typecode).strip() or None


def build_meteo_and_wind(
    flight_df: pd.DataFrame,
    era5_store: Union[str, Path],
    time_buffer_hours: float = 3.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch ERA5 meteo for the bbox/time-window around the flight.

    Returns (meteo_df, wind_df):
      meteo_df: ERA5 fields on a bbox x altitudes x times grid.
      wind_df:  columns (ts, latitude, longitude, h, u, v) ready for
                `Base.enable_wind(wind_df)`.
    """
    try:
        from fastmeteo.source import ArcoEra5
    except ImportError as exc:
        raise ImportError(
            "opentop replay requires the `fastmeteo` package. "
            'Install with: pip install "opentop[replay]"'
        ) from exc

    import math

    from openap.aero import ft

    import numpy as np

    era5 = ArcoEra5(local_store=str(era5_store))

    start = pd.Timestamp(flight_df["timestamp"].min())  # type: ignore[arg-type]
    stop = pd.Timestamp(flight_df["timestamp"].max())  # type: ignore[arg-type]
    stop_padded = stop + pd.Timedelta(hours=time_buffer_hours)
    timestamps = pd.date_range(start, stop_padded, freq="1h")

    lat_min = math.ceil(float(flight_df["latitude"].min())) - 2  # type: ignore[arg-type]
    lat_max = math.ceil(float(flight_df["latitude"].max())) + 2  # type: ignore[arg-type]
    lon_min = math.ceil(float(flight_df["longitude"].min())) - 4  # type: ignore[arg-type]
    lon_max = math.ceil(float(flight_df["longitude"].max())) + 4  # type: ignore[arg-type]
    latitudes = np.arange(lat_min, lat_max, 1)
    longitudes = np.arange(lon_min, lon_max, 1)
    altitudes = np.arange(1000, 46000, 1500)  # feet

    lat_g, lon_g, alt_g, time_g = np.meshgrid(
        latitudes, longitudes, altitudes, timestamps
    )
    grid = pd.DataFrame(
        {
            "latitude": lat_g.flatten(),
            "longitude": lon_g.flatten(),
            "altitude": alt_g.flatten(),
            "timestamp": time_g.flatten(),
        }
    )
    meteo = era5.interpolate(grid)

    wind = (
        meteo.rename(
            columns={
                "u_component_of_wind": "u",
                "v_component_of_wind": "v",
            }
        )
        .assign(
            ts=lambda x: (x["timestamp"] - x["timestamp"].iloc[0]).dt.total_seconds(),
            h=lambda x: x["altitude"] * ft,
        )[["ts", "latitude", "longitude", "h", "u", "v"]]
        .reset_index(drop=True)
    )

    return meteo, wind


def _agg_conditions(meteo: pd.DataFrame) -> pd.DataFrame:
    """Annotate a meteo DataFrame with contrail flags.

    Adds columns: rhi, crit_temp, sac, issr, persistent.
    Ported from debug/epsilon_constraint/driver_epsilon_constraint.py.
    """
    from openap import aero, contrail

    return meteo.assign(
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


def build_contrail_interpolant(meteo_df: pd.DataFrame, sigma: int = 2):
    """Build a 4D CasADi bspline interpolant of contrail persistence cost.

    Pipeline: openap contrail physics → 4D reshape → Gaussian smoothing over
    (height, lat, lon) → opentop.tools.interpolant_from_dataframe(bspline).

    Args:
        meteo_df: ERA5 meteo with columns timestamp, latitude, longitude,
            altitude (ft), temperature, specific_humidity.
        sigma: Gaussian smoothing sigma applied over (height, lat, lon) axes.

    Returns:
        ca.Function — 4D bspline interpolant over (lon, lat, h_m, ts_s).
    """
    try:
        from scipy.ndimage import gaussian_filter
    except ImportError as exc:
        raise ImportError(
            "opentop replay requires the `scipy` package. "
            'Install with: pip install "opentop[replay]"'
        ) from exc

    from openap.aero import ft

    from opentop import tools

    contrail_df = _agg_conditions(meteo_df)

    df_cost = contrail_df.assign(
        height=lambda x: x.altitude * ft,
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

    return tools.interpolant_from_dataframe(df_cost)
