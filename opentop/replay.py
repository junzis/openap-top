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
