from typing import Optional

import casadi as ca
import xarray as xr
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

import numpy as np
import pandas as pd
from openap import aero


def read_grids(paths: str | list[str], engine=None) -> pd.DataFrame:
    """Read meteorological grid data from GRIB/NetCDF files.

    Args:
        paths: File path(s). The file with the lowest time value
            should correspond to take-off time.
        engine: File reader engine ('cfgrib', 'netcdf4', etc.).

    Returns:
        pd.DataFrame: Grid data with longitude adjusted to [-180, 180],
            height column 'h' (meters), and 'ts' (seconds from start).
    """
    df = (
        xr.open_mfdataset(paths, engine=engine)
        .to_dataframe()
        .reset_index()
        .drop(columns=["step", "valid_time"])
        .assign(longitude=lambda d: (d.longitude + 180) % 360 - 180)
        .assign(h=lambda d: aero.h_isa(d.isobaricInhPa * 100))
        .assign(ts=lambda d: (d.time - d.time.min()).dt.total_seconds())
    )
    return df


def make_projection(lat1, lon1, lat2, lon2):
    """Create a pyproj Lambert Conformal Conic projection centered on two points.

    Args:
        lat1, lon1: Latitude and longitude of the first point.
        lat2, lon2: Latitude and longitude of the second point.

    Returns:
        pyproj.Proj: Projection object. Call proj(lon, lat) to get (x, y)
        in meters, or proj(x, y, inverse=True) to get (lon, lat).
    """
    from pyproj import Proj

    return Proj(
        proj="lcc",
        ellps="WGS84",
        lat_1=min(lat1, lat2),
        lat_2=max(lat1, lat2),
        lat_0=(lat1 + lat2) / 2,
        lon_0=(lon1 + lon2) / 2,
    )


class PolyWind:
    """
    A class to model wind fields using second order polynomial regression.
    """

    def __init__(self, windfield: pd.DataFrame, proj, lat1, lon1, lat2, lon2, margin=5):
        self.wind = windfield

        # select region based on airports
        df = (
            self.wind.query(f"longitude <= {max(lon1, lon2) + margin}")
            .query(f"longitude >= {(min(lon1, lon2)) - margin}")
            .query(f"latitude <= {max(lat1, lat2) + margin}")
            .query(f"latitude >= {min(lat1, lat2) - margin}")
            .query("h <= 13000")
        )

        x, y = proj(df.longitude, df.latitude)

        df = df.assign(x=x, y=y)

        model = make_pipeline(PolynomialFeatures(2), Ridge())
        model.fit(df[["x", "y", "h", "ts"]], df[["u", "v"]])

        features = model["polynomialfeatures"].get_feature_names_out()
        features = [string.replace("^", "**") for string in features]
        features = [string.replace(" ", "*") for string in features]

        self.features = features
        self.coef_u, self.coef_v = model["ridge"].coef_

    def calc_u(self, x, y, h, ts):
        """Compute eastward wind component (m/s) at given position."""
        u = sum(
            [
                eval(f, {}, {"x": x, "y": y, "h": h, "ts": ts}) * c
                for (f, c) in zip(self.features, self.coef_u)
            ]
        )
        return u

    def calc_v(self, x, y, h, ts):
        """Compute northward wind component (m/s) at given position."""
        v = sum(
            [
                eval(f, {}, {"x": x, "y": y, "h": h, "ts": ts}) * c
                for (f, c) in zip(self.features, self.coef_v)
            ]
        )
        return v


def construct_interpolant(
    longitude: np.array,
    latitude: np.array,
    height: np.array,
    grid_value: np.array,
    timestamp: Optional[np.array] = None,
    shape: str = "linear",
):
    """Construct a CasADi interpolant for 3D or 4D grid cost.

    Args:
        longitude: Sorted unique longitude values.
        latitude: Sorted unique latitude values.
        height: Sorted unique height values (meters).
        grid_value: Flattened grid values.
        timestamp: Sorted unique timestamps. If provided, creates 4D grid.
        shape: Interpolation type, "linear" or "bspline".

    Returns:
        ca.interpolant: CasADi interpolant object.
    """

    assert shape in ["linear", "bspline"]

    if max(height) > 20_000:
        raise ValueError(
            "Grid contains heights above 20,000 meters. "
            "Your 'height' values might be in feet — they must be in meters."
        )

    if timestamp is None:
        return ca.interpolant(
            "grid_cost", shape, [longitude, latitude, height], grid_value
        )
    else:
        return ca.interpolant(
            "grid_cost", shape, [longitude, latitude, height, timestamp], grid_value
        )


def interpolant_from_dataframe(
    df: pd.DataFrame, shape: str = "linear"
) -> ca.interpolant:
    """Construct a CasADi interpolant from a DataFrame.

    DataFrame must have columns: longitude, latitude, height (m), cost.
    If a 'ts' column is present, creates a 4D (time-dependent) grid.

    Args:
        df: Grid data with required columns.
        shape: Interpolation type, "linear" or "bspline".

    Returns:
        ca.interpolant: CasADi interpolant object.
    """

    assert shape in ["linear", "bspline"], "Shape must be 'linear' or 'bspline'"
    assert "longitude" in df.columns, "Missing 'longitude' column in DataFrame"
    assert "latitude" in df.columns, "Missing 'latitude' column in DataFrame"
    assert "height" in df.columns, "Missing 'height' column in DataFrame"

    if "ts" in df.columns:
        df = df.sort_values(["ts", "height", "latitude", "longitude"], ascending=True)
        return construct_interpolant(
            df.longitude.unique(),
            df.latitude.unique(),
            df.height.unique(),
            df.cost.values,
            df.ts.unique(),
            shape=shape,
        )
    else:
        df = df.sort_values(["height", "latitude", "longitude"], ascending=True)
        return construct_interpolant(
            df.longitude.unique(),
            df.latitude.unique(),
            df.height.unique(),
            df.cost.values,
            shape=shape,
        )
