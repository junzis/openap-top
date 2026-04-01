"""Compare IPOPT NLP scaling methods: gradient-based vs none.

Tests fuel-only and contrail+CO2 optimization across multiple routes.
Contrail+CO2 objective from openap.dev/optimize/contrails.html.
"""

import sys
import time
import urllib
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import gaussian_filter

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import openap
import openap.casadi as oc

openap_local = project_root / "openap"
if hasattr(openap, "__path__"):
    openap.__path__.insert(0, str(openap_local))

from openap import top

# --- Configuration ---
AIRCRAFT = "A320"
M0 = 0.85

ROUTES = [
    ("EHAM", "LGAV"),  # Amsterdam - Athens (~2100 km)
    ("EHAM", "EDDF"),  # Amsterdam - Frankfurt (~370 km, short)
    ("EGLL", "LEMD"),  # London - Madrid (~1260 km, medium)
]

SCALING_METHODS = ["gradient-based", "none"]

DATA_DIR = Path(__file__).parent / "tmp"


def prepare_contrail_data():
    """Download and prepare contrail cost data (from openap.dev example)."""
    DATA_DIR.mkdir(exist_ok=True)
    nc_path = DATA_DIR / "contrail.nc"

    if not nc_path.exists():
        print("Downloading contrail data...")
        urllib.request.urlretrieve(
            "https://opendap.4tu.nl/thredds/fileServer/data2/djht/"
            "bea8a3fe-e34c-4598-9f94-c5a5c63348e5/1/contrail_original.nc",
            str(nc_path),
        )

    ds = xr.open_dataset(str(nc_path)).sel(time="2015-12-18")

    level_pressure = [
        0.0, 10.0, 30.0, 50.0, 70.0, 90.0787, 110.6606, 132.3968,
        155.7909, 181.1544, 208.6494, 238.3258, 270.1530, 304.0465,
        339.8891, 377.5467, 416.8789, 457.7442, 500.0, 543.4970,
        588.0685, 633.5144, 679.5799, 725.9285, 772.1102, 817.5241,
        861.3757, 902.6287, 939.9520, 971.6610, 995.6532, 1009.3396,
    ]

    df = (
        ds.to_dataframe()
        .reset_index()
        .assign(lev=lambda x: x.lev.astype(int))
        .merge(
            pd.DataFrame(level_pressure, columns=["hPa"]).reset_index(names="lev"),
            on="lev",
        )
        .assign(height=lambda x: openap.aero.h_isa(x.hPa * 100).round(-2))
        .assign(longitude=lambda x: ((x.lon + 180) % 360 - 180))
        .query("height<15000")
    )

    df_cost = (
        df.rename(columns={"lat": "latitude", "atr20_contrail": "cost"})[
            ["time", "latitude", "longitude", "hPa", "height", "cost"]
        ]
        .query("-20<longitude<40 and 30<latitude<70 and time.dt.hour==12")
        .sort_values(["height", "latitude", "longitude"])
    )

    cost = df_cost.cost.values.reshape(
        df_cost.height.nunique(),
        df_cost.latitude.nunique(),
        df_cost.longitude.nunique(),
    )
    cost_ = gaussian_filter(cost, sigma=1, mode="nearest")
    df_cost = df_cost.assign(cost=cost_.flatten())

    return df_cost


def make_contrail_objective(optimizer):
    """Create the contrail+CO2 objective function (from openap.dev example)."""

    def objective(x, u, dt, **kwargs):
        vtas = oc.aero.mach2tas(u[0], x[2])
        kwargs.pop("n_dim", None)
        kwargs.pop("time_dependent", None)
        contrail_cost = (
            optimizer.obj_grid_cost(
                x, u, dt, n_dim=3, time_dependent=False, **kwargs
            )
            * vtas
            * 1e-3
        )
        co2_cost = optimizer.obj_fuel(x, u, dt, **kwargs) * 7.03e-15
        return contrail_cost + co2_cost

    return objective


def run_opt(OptimizerClass, origin, dest, scaling_method, objective="fuel",
            df_cost=None, fuel_optimal=None):
    """Run a single optimization and return results."""
    optimizer = OptimizerClass(AIRCRAFT, origin, dest, M0)

    setup_kwargs = {
        "debug": True,
        "ipopt_kwargs": {"nlp_scaling_method": scaling_method},
    }
    if df_cost is not None:
        setup_kwargs["max_iterations"] = 2000
    optimizer.setup(**setup_kwargs)

    traj_kwargs = {}
    if df_cost is not None:
        interpolant = top.tools.interpolant_from_dataframe(df_cost)
        traj_kwargs.update(
            objective=make_contrail_objective(optimizer),
            interpolant=interpolant,
            n_dim=3,
            return_failed=True,
        )
        if fuel_optimal is not None:
            traj_kwargs["initial_guess"] = fuel_optimal
    else:
        traj_kwargs["objective"] = objective

    t0 = time.time()
    df = optimizer.trajectory(**traj_kwargs)
    elapsed = time.time() - t0

    stats = optimizer.solver.stats()
    obj_val = float(optimizer.solution["f"])
    return df, elapsed, stats, obj_val


def print_comparison(label, results):
    """Print side-by-side comparison of results."""
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")

    for method, (df, elapsed, stats, obj_val) in results.items():
        success = stats.get("success", False)
        iters = stats.get("iter_count", "N/A")

        print(f"\n  [{method}]")
        print(f"    Success:    {success}")
        print(f"    Iterations: {iters}")
        print(f"    Wall time:  {elapsed:.1f}s")
        print(f"    NLP obj(f): {obj_val:.6e}")

        if df is not None:
            fuel_burn = df.mass.iloc[0] - df.mass.iloc[-1]
            print(f"    Fuel burn:  {fuel_burn:.1f} kg")
            print(f"    Alt range:  {df.altitude.min():.0f} - {df.altitude.max():.0f} ft")
            print(f"    Mach range: {df.mach.min():.3f} - {df.mach.max():.3f}")
            print(f"    Total fuel cost:  {df.fuel_cost.sum():.4f}")
        else:
            print(f"    Result: FAILED (None)")

    # Print diff if both succeeded
    methods = list(results.keys())
    if len(methods) == 2:
        df1, _, _, obj1 = results[methods[0]]
        df2, _, _, obj2 = results[methods[1]]
        if df1 is not None and df2 is not None:
            fuel1 = df1.mass.iloc[0] - df1.mass.iloc[-1]
            fuel2 = df2.mass.iloc[0] - df2.mass.iloc[-1]
            pct = abs(fuel1 - fuel2) / max(fuel1, 1e-15) * 100
            print(f"\n  Fuel burn diff:  {abs(fuel1 - fuel2):.1f} kg ({pct:.2f}%)")
            if abs(obj1) > 1e-15:
                opct = abs(obj1 - obj2) / abs(obj1) * 100
                print(f"  NLP obj diff:    {abs(obj1 - obj2):.6e} ({opct:.2f}%)")


if __name__ == "__main__":
    print("Comparing IPOPT NLP scaling methods: gradient-based vs none")
    print(f"Aircraft: {AIRCRAFT}, M0: {M0}")
    print(f"Routes: {ROUTES}")

    print("\nPreparing contrail cost data...")
    df_cost = prepare_contrail_data()
    print(f"  Cost data shape: {df_cost.shape}")

    for origin, dest in ROUTES:
        route_label = f"{origin}-{dest}"
        print(f"\n\n{'#'*70}")
        print(f"  Route: {route_label}")
        print(f"{'#'*70}")

        # Cruise fuel
        cruise_fuel_results = {}
        for method in SCALING_METHODS:
            print(f"\n--- Cruise fuel {method} ---")
            cruise_fuel_results[method] = run_opt(
                top.Cruise, origin, dest, method, objective="fuel"
            )
        print_comparison(f"Cruise Fuel - {route_label}", cruise_fuel_results)

        fuel_optimal = cruise_fuel_results["gradient-based"][0]

        # Cruise contrail+CO2
        cruise_cc_results = {}
        for method in SCALING_METHODS:
            print(f"\n--- Cruise contrail+CO2 {method} ---")
            cruise_cc_results[method] = run_opt(
                top.Cruise, origin, dest, method,
                df_cost=df_cost, fuel_optimal=fuel_optimal,
            )
        print_comparison(f"Cruise Contrail+CO2 - {route_label}", cruise_cc_results)

        # CompleteFlight fuel
        full_fuel_results = {}
        for method in SCALING_METHODS:
            print(f"\n--- CompleteFlight fuel {method} ---")
            full_fuel_results[method] = run_opt(
                top.CompleteFlight, origin, dest, method, objective="fuel"
            )
        print_comparison(f"CompleteFlight Fuel - {route_label}", full_fuel_results)

        full_fuel_optimal = full_fuel_results["gradient-based"][0]

        # CompleteFlight contrail+CO2
        full_cc_results = {}
        for method in SCALING_METHODS:
            print(f"\n--- CompleteFlight contrail+CO2 {method} ---")
            full_cc_results[method] = run_opt(
                top.CompleteFlight, origin, dest, method,
                df_cost=df_cost, fuel_optimal=full_fuel_optimal,
            )
        print_comparison(
            f"CompleteFlight Contrail+CO2 - {route_label}", full_cc_results
        )
