import cfgrib
import pandas as pd
from openap import aero, nav
from pyproj import Proj
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge


def read_grib(fgrib):
    df = (
        cfgrib.open_dataset(
            fgrib,
            backend_kwargs={"filter_by_keys": {"typeOfLevel": "isobaricInhPa"}},
        )
        .to_dataframe()
        .reset_index()
        .assign(longitude=lambda d: d.longitude - 180)
        .assign(h=lambda d: aero.h_isa(d.isobaricInhPa * 100))
    )[["latitude", "longitude", "h", "u", "v"]]
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

        features = model["polynomialfeatures"].get_feature_names()
        features = [string.replace("^", "**") for string in features]
        features = [string.replace(" ", "*") for string in features]

        self.features = features
        self.coef_u, self.coef_v = model["ridge"].coef_

    def calc_u(self, x, y, h):
        u = sum(
            [
                eval(f, {}, {"x0": x, "x1": y, "x2": h}) * c
                for (f, c) in zip(self.features, self.coef_u)
            ]
        )
        return u

    def calc_v(self, x, y, h):
        v = sum(
            [
                eval(f, {}, {"x0": x, "x1": y, "x2": h}) * c
                for (f, c) in zip(self.features, self.coef_v)
            ]
        )
        return v
