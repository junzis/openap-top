from pathlib import Path
from typing import Optional, Union

import casadi as ca
import xarray as xr
from sklearn.linear_model import Ridge
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
    """Polynomial wind model. Fitted via sklearn Ridge + PolynomialFeatures(2).

    Works with both numeric inputs (returns float) and CasADi symbolic inputs
    (SX/MX — returns a symbolic expression usable inside an NLP).
    """

    def __init__(self, wind, proj, lat1, lon1, lat2, lon2, margin=1.5):
        self.wind = wind
        self.proj = proj

        df = (
            self.wind.query(f"longitude <= {max(lon1, lon2) + margin}")
            .query(f"longitude >= {(min(lon1, lon2)) - margin}")
            .query(f"latitude <= {max(lat1, lat2) + margin}")
            .query(f"latitude >= {min(lat1, lat2) - margin}")
            .query("h <= 13000")
        )
        x, y = proj(df.longitude, df.latitude)
        df = df.assign(x=x, y=y)

        self._poly = PolynomialFeatures(2)
        X_train = df[["x", "y", "h", "ts"]].values
        self._poly.fit(X_train)
        ridge = Ridge()
        ridge.fit(self._poly.transform(X_train), df[["u", "v"]].values)
        self._coef = ridge.coef_          # shape (2, n_features)
        self._intercept = ridge.intercept_  # shape (2,)

    def _feature_vec(self, x, y, h, ts):
        """Build feature row. Polymorphic: numeric → np.ndarray; CasADi → list of SX/MX."""
        if isinstance(x, (ca.SX, ca.MX, ca.DM)):
            powers = self._poly.powers_   # (n_features, 4)
            feats = []
            for row in powers:
                term = 1
                for var, p in zip((x, y, h, ts), row):
                    if p == 0:
                        continue
                    term = term * (var ** int(p))
                feats.append(term)
            return feats
        else:
            return self._poly.transform(np.array([[x, y, h, ts]], dtype=float))[0]

    def calc_u(self, x, y, h, ts):
        """Compute eastward wind component (m/s) at given position."""
        feats = self._feature_vec(x, y, h, ts)
        if isinstance(x, (ca.SX, ca.MX, ca.DM)):
            expr = self._intercept[0]
            for c, f in zip(self._coef[0], feats):
                expr = expr + c * f
            return expr
        return float(np.dot(self._coef[0], feats) + self._intercept[0])

    def calc_v(self, x, y, h, ts):
        """Compute northward wind component (m/s) at given position."""
        feats = self._feature_vec(x, y, h, ts)
        if isinstance(x, (ca.SX, ca.MX, ca.DM)):
            expr = self._intercept[1]
            for c, f in zip(self._coef[1], feats):
                expr = expr + c * f
            return expr
        return float(np.dot(self._coef[1], feats) + self._intercept[1])


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
    df: pd.DataFrame, shape: str = "bspline"
) -> ca.interpolant:
    """Construct a CasADi interpolant from a DataFrame.

    DataFrame must have columns: longitude, latitude, height (m), cost.
    If a 'ts' column is present, creates a 4D (time-dependent) grid.

    Args:
        df: Grid data with required columns.
        shape: Interpolation type, "bspline" (default, smooth derivatives)
            or "linear". "bspline" is strongly recommended for any
            non-trivial grid: linear interpolants have discontinuous
            derivatives at every grid-cell boundary, which causes IPOPT's
            line search to oscillate on blended or constrained objectives.

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


def save_interpolant(interpolant: ca.Function, path: Union[str, Path]) -> None:
    """Serialize a CasADi interpolant to disk for later reuse.

    Building a bspline interpolant over a large 4D cost grid is expensive
    (cubic-degree tensor-product spline construction can take minutes for
    grids of ~10^5 points). Saving the built interpolant lets subsequent
    runs skip the rebuild.

    Args:
        interpolant: CasADi Function returned by
            ``interpolant_from_dataframe`` or ``construct_interpolant``.
        path: Destination file path. The ``.casadi`` extension is
            conventional but not enforced.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    interpolant.save(str(path))


def load_interpolant(path: Union[str, Path]) -> ca.Function:
    """Load a previously-saved CasADi interpolant from disk.

    Args:
        path: File path written by :func:`save_interpolant`.

    Returns:
        ca.Function: The deserialized interpolant, usable immediately
        with ``Base.obj_grid_cost`` via the ``interpolant`` kwarg.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No cached interpolant at {path}")
    return ca.Function.load(str(path))


def cached_interpolant_from_dataframe(
    df: pd.DataFrame,
    cache_path: Union[str, Path],
    shape: str = "bspline",
) -> ca.Function:
    """Load a cached interpolant from disk, or build-and-save if absent.

    This is the recommended helper for workflows that reuse the same cost
    grid across many trajectories: the first call builds the interpolant
    (which can take minutes for bspline on a large grid) and writes it to
    ``cache_path``; subsequent calls skip the build entirely.

    Args:
        df: Grid DataFrame — same layout expected by
            :func:`interpolant_from_dataframe` (columns: ``longitude``,
            ``latitude``, ``height`` in meters, ``cost``; optionally ``ts``).
        cache_path: Destination file path for the serialized interpolant.
            Parent directories are created if needed. No cache invalidation
            is performed — if ``df`` changes, the caller must delete the
            cache file or use a different path.
        shape: Interpolation type passed to
            :func:`interpolant_from_dataframe` on a cache miss. Default
            ``"bspline"`` (smooth derivatives, clean convergence for
            non-convex grid-cost objectives).

    Returns:
        ca.Function: The interpolant, either loaded from ``cache_path`` or
        freshly built and then saved there.
    """
    cache_path = Path(cache_path)
    if cache_path.exists():
        return load_interpolant(cache_path)
    interpolant = interpolant_from_dataframe(df, shape=shape)
    save_interpolant(interpolant, cache_path)
    return interpolant
