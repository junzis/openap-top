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
    """
    Parameters:
    paths (str or list of str): The paths can be a single path or a list of paths.
        You must ensure the file with the lowest `time` value corresponds to the
        take-off time of your flight.
    engine (str, optional): The engine to use for reading the grib files.
        Defaults to None. Options are 'cfgrib' and 'netcdf4', etc.

    Returns:
    pd.DataFrame: A pandas DataFrame containing the data from the grib files.

    The DataFrame includes the following transformations:
    - Adjusts the 'longitude' column to be within the range [-180, 180].
    - Adds a column 'h' calculated using the 'isobaricInhPa' column.
    - Adds a column 'ts' representing the total seconds from the minimum time.
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
        u = sum(
            [
                eval(f, {}, {"x": x, "y": y, "h": h, "ts": ts}) * c
                for (f, c) in zip(self.features, self.coef_u)
            ]
        )
        return u

    def calc_v(self, x, y, h, ts):
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
    """
    This function is used to create the 3d or 4d grid based cost function.

    It interpolates grid values based on the given longitude, latitude, height,
        timestamp, and grid_value arrays.

    Parameters:
        longitude (np.array): Array of longitudes.
        latitude (np.array): Array of latitudes.
        height (np.array): Array of heights (in meters).
        grid_value (np.array): Array of grid values.
        timestamp (Optional[np.array], optional): Array of timestamps. Defaults to None.
        shape (str, optional): Interpolation shape. Defaults to "linear".

    Returns:
        ca.interpolant: Casadi interpolant object representing the grid values.
    """

    assert shape in ["linear", "bspline"]

    if max(height) > 20_000:
        raise Warning(
            """Grid contains heights above 20,000 meters. You 'height' might be feet
            Make sure the 'height' values are in meters."""
        )

    if timestamp is None:
        return ca.interpolant(
            "grid_cost", shape, [longitude, latitude, height], grid_value
        )
    else:
        return ca.interpolant(
            "grid_cost", shape, [longitude, latitude, height, timestamp], grid_value
        )


def interp_grid(
    longitude, latitude, height, grid_value, timestamp=None, shape="linear"
):
    raise DeprecationWarning(
        "Function interp_grid() is deprecated, "
        "use interpolant_from_dataframe() instead."
    )


def interpolant_from_dataframe(
    df: pd.DataFrame, shape: str = "linear"
) -> ca.interpolant:
    """
    This function is used to create the 3d or 4d grid based cost function.

    It interpolates grid values based on the given DataFrame. The DataFrame must
    contain columns 'longitude', 'latitude', 'height' (meters), and 'cost'.

    If the DataFrame contains a 'ts' column, it will be used as the timestamp,
    and the grid will be treated as 4d.

    Parameters:
        df (pd.DataFrame): DataFrame containing the grid values.
        shape (str, optional): Interpolation shape. Defaults to "linear".

    Returns:
        ca.interpolant: Casadi interpolant object representing the grid values.
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
