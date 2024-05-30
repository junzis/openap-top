import casadi as ca
import cfgrib
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

from openap import aero


def read_grib(fgrib):
    df = (
        cfgrib.open_dataset(
            fgrib,
            backend_kwargs={
                "filter_by_keys": {"typeOfLevel": "isobaricInhPa"},
                "indexpath": "",
            },
        )
        .to_dataframe()
        .reset_index()
        .assign(longitude=lambda d: (d.longitude + 180) % 360 - 180)
        .assign(h=lambda d: aero.h_isa(d.isobaricInhPa * 100))
    )
    return df


class PolyWind:
    def __init__(self, windfield: pd.DataFrame, proj, lat1, lon1, lat2, lon2, margin=5):
        self.wind = windfield

        # select region based on airports
        df = (
            self.wind.query(f"longitude <= {max(lon1, lon2) + margin}")
            .query(f"longitude >= {(min(lon1, lon2)) - margin}")
            .query(f"latitude <= {max(lat1, lat2) + margin}")
            .query(f"latitude >= {min(lat1, lat2) - margin}")
            .query(f"h >= 5000")
            .query(f"h <= 13000")
        )

        x, y = proj(df.longitude, df.latitude)

        df = df.assign(x=x, y=y)

        model = make_pipeline(PolynomialFeatures(2), Ridge())
        model.fit(df[["x", "y", "h"]], df[["u", "v"]])

        features = model["polynomialfeatures"].get_feature_names_out()
        features = [string.replace("^", "**") for string in features]
        features = [string.replace(" ", "*") for string in features]

        self.features = features
        self.coef_u, self.coef_v = model["ridge"].coef_

    def calc_u(self, x, y, h):
        u = sum(
            [
                eval(f, {}, {"x": x, "y": y, "h": h}) * c
                for (f, c) in zip(self.features, self.coef_u)
            ]
        )
        return u

    def calc_v(self, x, y, h):
        v = sum(
            [
                eval(f, {}, {"x": x, "y": y, "h": h}) * c
                for (f, c) in zip(self.features, self.coef_v)
            ]
        )
        return v


def interp_grid(Lon, Lat, H, T, V, shape="linear"):
    interpolant = ca.interpolant("grid_cost", shape, [Lon, Lat, H, T], V)
    return interpolant
