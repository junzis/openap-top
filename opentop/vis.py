import warnings

import matplotlib.pyplot as plt
from cartopy import crs as ccrs
from cartopy.feature import BORDERS, LAND, OCEAN
from matplotlib.gridspec import GridSpec
from openap import aero

import numpy as np
import pandas as pd


def plot_map(df, windfield=None, ax=None, barb_steps=10, color=None, label=None):
    """Plot trajectory on a map with optional wind barbs.

    Args:
        df: Trajectory DataFrame with latitude/longitude columns.
        windfield: Wind data DataFrame. If provided, draws wind barbs.
        ax: Matplotlib axes with cartopy projection. Created if None.
        barb_steps: Step interval for wind barb subsampling.

    Returns:
        matplotlib.pyplot module.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, module="cartopy")

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

        ax.set_extent([lonmin - 4, lonmax + 4, latmin - 2, latmax + 2])  # type: ignore[attr-defined]  # cartopy GeoAxes methods not in matplotlib stubs
        ax.add_feature(OCEAN, facecolor="#d1e0e0", zorder=-1, lw=0)  # type: ignore[attr-defined]
        ax.add_feature(LAND, facecolor="#f5f5f5", lw=0)  # type: ignore[attr-defined]
        ax.add_feature(BORDERS, lw=0.5, color="gray")  # type: ignore[attr-defined]
        ax.gridlines(draw_labels=True, color="gray", alpha=0.5, ls="--")  # type: ignore[attr-defined]
        ax.coastlines(resolution="50m", lw=0.5, color="gray")  # type: ignore[attr-defined]

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
            color=color if color is not None else "tab:green",
            transform=ccrs.Geodetic(),
            linewidth=2,
            marker=".",
            label=label if label is not None else "Optimal",
        )

        ax.legend()

        return plt


def trajectory(
    df,
    windfield=None,
    labels=None,
    barb_steps=10,
):
    """Plot trajectory profiles (altitude, TAS, VS) alongside a map.

    Args:
        df: Single DataFrame or list of DataFrames for overlay.
        windfield: Wind data DataFrame for map wind barbs (first df only).
        labels: Legend labels when df is a list. Defaults to
            ["actual", "optimized"] for 2 dfs, ["trajectory N", ...] else.
        barb_steps: Step interval for wind barb subsampling.

    Returns:
        matplotlib.pyplot module.
    """
    if isinstance(df, pd.DataFrame):
        dfs = [df]
        is_list = False
    else:
        dfs = list(df)
        is_list = True

    if not dfs:
        raise ValueError("vis.trajectory requires at least one DataFrame")

    if labels is None:
        if len(dfs) == 2:
            labels = ["actual", "optimized"]
        else:
            labels = [f"trajectory {i + 1}" for i in range(len(dfs))]

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    fig = plt.figure(figsize=(12, 5))
    gs = GridSpec(3, 2)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[2, 0])

    for i, d in enumerate(dfs):
        color = colors[i % len(colors)]
        label = labels[i] if is_list else None
        ax1.plot(d.ts, d.altitude, lw=2, marker=".", color=color, label=label)
        ax2.plot(d.ts, d.tas, lw=2, marker=".", color=color, label=label)
        ax3.plot(d.ts, d.vertical_rate, lw=2, marker=".", color=color, label=label)

    ax1.set_ylabel("altitude (ft)")
    ax1.set_ylim(0, 45_000)
    ax1.grid(ls=":")
    ax2.set_ylabel("TAS")
    ax2.set_ylim(0, 600)
    ax2.grid(ls=":")
    ax3.set_ylabel("VS (ft/min)")
    ax3.set_ylim(-3000, 3000)
    ax3.grid(ls=":")

    if is_list:
        ax1.legend(loc="best", fontsize=8)

    mean_lon = float(np.mean([d.longitude.mean() for d in dfs]))
    mean_lat = float(np.mean([d.latitude.mean() for d in dfs]))
    ax5 = fig.add_subplot(
        gs[:, 1],
        projection=ccrs.TransverseMercator(
            central_longitude=mean_lon,
            central_latitude=mean_lat,
        ),
    )

    for i, d in enumerate(dfs):
        color = colors[i % len(colors)]
        label = labels[i] if is_list else None
        plot_map(
            d,
            windfield=windfield if i == 0 else None,
            ax=ax5,
            barb_steps=barb_steps,
            color=color,
            label=label,
        )

    if is_list:
        ax5.legend(loc="best", fontsize=8)

    plt.tight_layout()
    return plt
