import warnings

import matplotlib.pyplot as plt
from cartopy import crs as ccrs
from cartopy.feature import BORDERS, LAND, OCEAN
from matplotlib.gridspec import GridSpec

import numpy as np
from openap import aero

warnings.filterwarnings("ignore")


def map(df, windfield=None, ax=None, barb_steps=10):
    lat1, lon1 = df.latitude.iloc[0], df.longitude.iloc[0]
    lat2, lon2 = df.latitude.iloc[-1], df.longitude.iloc[-1]

    latmin, latmax = min(lat1, lat2), max(lat1, lat2)
    lonmin, lonmax = min(lon1, lon2), max(lon1, lon2)

    if ax is None:
        ax = plt.axes(
            projection=ccrs.TransverseMercator(
                central_longitude=df.longitude.mean(),
                central_latitude=df.latitude.mean(),
            )
        )

    ax.set_extent([lonmin - 4, lonmax + 4, latmin - 2, latmax + 2])
    ax.add_feature(OCEAN, facecolor="#d1e0e0", zorder=-1, lw=0)
    ax.add_feature(LAND, facecolor="#f5f5f5", lw=0)
    ax.add_feature(BORDERS, lw=0.5, color="gray")
    ax.gridlines(draw_labels=True, color="gray", alpha=0.5, ls="--")
    ax.coastlines(resolution="50m", lw=0.5, color="gray")

    if windfield is not None:
        # get the closed altitude
        h_median = df.altitude.median() * aero.ft
        fl = int(round(h_median / aero.ft / 100, -1))
        idx = np.argmin(abs(windfield.h.unique() - h_median))
        df_wind = (
            windfield.query(f"h=={windfield.h.unique()[idx]}")
            .query(f"longitude <= {lonmax + 2}")
            .query(f"longitude >= {lonmin - 2}")
            .query(f"latitude <= {latmax + 2}")
            .query(f"latitude >= {latmin - 2}")
        )

        ax.barbs(
            df_wind.longitude.values[::barb_steps],
            df_wind.latitude.values[::barb_steps],
            df_wind.u.values[::barb_steps],
            df_wind.v.values[::barb_steps],
            transform=ccrs.PlateCarree(),
            color="k",
            length=5,
            lw=0.5,
            label=f"Wind FL{fl}",
        )

    # great circle
    ax.scatter(lon1, lat1, c="darkgreen", transform=ccrs.Geodetic())
    ax.scatter(lon2, lat2, c="tab:red", transform=ccrs.Geodetic())

    ax.plot(
        [lon1, lon2],
        [lat1, lat2],
        label="Great Circle",
        color="tab:red",
        ls="--",
        transform=ccrs.Geodetic(),
    )

    # trajectory
    ax.plot(
        df.longitude,
        df.latitude,
        color="tab:green",
        transform=ccrs.Geodetic(),
        linewidth=2,
        marker=".",
        label="Optimal",
    )

    ax.legend()

    return plt


def trajectory(df, windfield=None, barb_steps=10):
    fig = plt.figure(figsize=(12, 5))

    gs = GridSpec(3, 2)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(df.ts, df.altitude, lw=2, marker=".")
    ax1.set_ylabel("altitude (ft)")
    ax1.set_ylim(0, 45_000)
    ax1.grid(ls=":")

    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(df.ts, df.tas, lw=2, marker=".")
    ax2.set_ylabel("TAS")
    ax2.set_ylim(0, 600)
    ax2.grid(ls=":")

    ax3 = fig.add_subplot(gs[2, 0])
    ax3.plot(df.ts, df.vertical_rate, lw=2, marker=".")
    ax3.set_ylabel("VS (ft/min)")
    ax3.set_ylim(-3000, 3000)
    ax3.grid(ls=":")

    ax5 = fig.add_subplot(
        gs[:, 1],
        projection=ccrs.TransverseMercator(
            central_longitude=df.longitude.mean(),
            central_latitude=df.latitude.mean(),
        ),
    )

    map(df, windfield, ax=ax5, barb_steps=barb_steps)

    plt.tight_layout()

    return plt
