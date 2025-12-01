# %%
import time
import warnings
import numpy as np
from openap_top import top
from scipy.ndimage import gaussian_filter
import pandas as pd
from traffic.data import opensky
from openap import contrail, aero
from fastmeteo.source import ArcoEra5
import math
from traffic.core import Flight

def objective(x, u, dt, **kwargs):
    grid_cost = optimizer.obj_grid_cost(
        x, u, dt, time_dependent=True, n_dim=4, **kwargs
    )
    fuel_cost = optimizer.obj_fuel(x, u, dt, **kwargs)
    return grid_cost * 0.5 + fuel_cost * 0.5

# %%
warnings.filterwarnings("ignore")

actype = "B738"
start = "2023-01-05 09:48"
stop = "2023-01-05 12:00"

m0 = 0.85

flight_0 = opensky.history(start, stop, callsign="RYR880W", return_flight=True)
counts = flight_0.data['icao24'].value_counts()
main_icao24 = counts.idxmax()
flight_main = Flight(flight_0.data[flight_0.data['icao24'] == main_icao24].copy())
t = flight_main.drop(["last_position", "onground"], axis=1).query("latitude.notnull()")

era5 = ArcoEra5(local_store="/tmp/era5-zarr")
# Enable wind
origin_lat = t.data.latitude.iloc[0]
origin_lon = t.data.longitude.iloc[0]
destination_lat = t.data.latitude.iloc[-1]
destination_lon = t.data.longitude.iloc[-1]

latmin = math.ceil(min(origin_lat, destination_lat)) - 2
latmax = math.ceil(max(origin_lat, destination_lat)) + 2
lonmin = math.ceil(min(origin_lon, destination_lon)) - 4
lonmax = math.ceil(max(origin_lon, destination_lon)) + 4

# creates numpy arrays
latitudes = np.arange(latmin, latmax, 1)
longitudes = np.arange(lonmin, lonmax, 1)
altitudes = np.arange(1000, 46000, 1500)
timestamps = pd.date_range(start, stop, freq="1H")

latitudes, longitudes, altitudes, times = np.meshgrid(
    latitudes, longitudes, altitudes, timestamps
)

grid = pd.DataFrame().assign(
    latitude=latitudes.flatten(),
    longitude=longitudes.flatten(),
    altitude=altitudes.flatten(),
    timestamp=times.flatten(),
)

meteo = era5.interpolate(grid)
f = meteo.assign(
    rhi=lambda d: contrail.relative_humidity(d.specific_humidity,
                                             aero.pressure(d.altitude * aero.ft),
                                             d.temperature, to="ice"),
    crit_temp=lambda d: contrail.critical_temperature_water(aero.pressure(d.altitude * aero.ft)),
    sac=lambda d: d.temperature < d.crit_temp,
    issr=lambda d: d.rhi > 1,
    persistent=lambda d: d.sac & d.issr
)

# Base optimizer
optimizer = top.CompleteFlight(
    "A320",
    (origin_lat, origin_lon),
    (destination_lat, destination_lon),
    m0,
)
# optimizer = top.Cruise(
#     actype,
#     (origin_lat, origin_lon),
#     (destination_lat, destination_lon),
#     m0,
# )

wind = (
    meteo.rename(columns={"u_component_of_wind": "u", "v_component_of_wind": "v"})
    .assign(ts=lambda x: (x.timestamp - x.timestamp.iloc[0]).dt.total_seconds())
    .eval("h = altitude * 0.3048")
)[["ts", "latitude", "longitude", "h", "u", "v"]]

# Prepare contrail cost grid
df_cost = (
    f.assign(
        height=lambda x: x.altitude * aero.ft,
        cost=lambda x: x.persistent.astype(float),
        ts=lambda x: (x.timestamp - x.timestamp.iloc[0]).dt.total_seconds(),
    )
    .sort_values(["ts", "height", "latitude", "longitude"])
)

cost = df_cost.cost.values.reshape(
    df_cost.ts.nunique(),
    df_cost.height.nunique(),
    df_cost.latitude.nunique(),
    df_cost.longitude.nunique(),
)

cost_ = gaussian_filter(cost, sigma=(0, 2, 2, 2), mode="nearest")
df_cost = df_cost.assign(cost=cost_.flatten()).fillna(0)
interpolant = top.tools.interpolant_from_dataframe(df_cost)

# Baseline trajectories
optimizer_fuel = optimizer.trajectory(objective="fuel", interpolant=interpolant)

# Evaluate components post-solution
grid_cost_val = float(np.sum(optimizer_fuel["grid_cost"]))
fuel_cost_val = float(np.sum(optimizer_fuel["fuel_cost"]))
obj = optimizer.solver.stats()["iterations"]["obj"][-1]
status = optimizer.solver.stats()['success']

print(
    f"Grid Cost: {grid_cost_val:.2f} | Fuel Cost: {fuel_cost_val:.2f} | Objective: {obj:.2f} | Success:{status}")

# For MultiPhase case
# obj_cr = optimizer.get_solver_stats()["cruise"]["iterations"]["obj"][-1]
# status_cr = optimizer.get_solver_stats()["cruise"]["success"]
# obj_cl = optimizer.get_solver_stats()["climb"]["iterations"]["obj"][-1]
# status_cl = optimizer.get_solver_stats()["climb"]["success"]
# obj_de = optimizer.get_solver_stats()["descent"]["iterations"]["obj"][-1]
# status_de = optimizer.get_solver_stats()["descent"]["success"]

# print(
#     f"Grid Cost: {grid_cost_val:.2f} | Fuel Cost: {fuel_cost_val:.2f} | Objective climb: {obj_cl:.2f} | Success climb:{status_cl} | Objective cruise: {obj_cr:.2f} | Success cruise:{status_cr} | Objective descent: {obj_de:.2f} | Success descent:{status_de} ")

optimizer.enable_wind(wind)
optimizer_wind = optimizer.trajectory(objective="fuel", initial_guess=optimizer_fuel, interpolant=interpolant)

# Evaluate components post-solution
grid_cost_val = float(np.sum(optimizer_wind["grid_cost"]))
fuel_cost_val = float(np.sum(optimizer_wind["fuel_cost"]))

obj = optimizer.solver.stats()["iterations"]["obj"][-1]
status = optimizer.solver.stats()['success']

print(
    f"Grid Cost: {grid_cost_val:.2f} | Fuel Cost: {fuel_cost_val:.2f} | Objective: {obj:.2f} | Success:{status}")

#For MultiPhase case
# obj_cr = optimizer.get_solver_stats()["cruise"]["iterations"]["obj"][-1]
# status_cr = optimizer.get_solver_stats()["cruise"]["success"]
# obj_cl = optimizer.get_solver_stats()["climb"]["iterations"]["obj"][-1]
# status_cl = optimizer.get_solver_stats()["climb"]["success"]
# obj_de = optimizer.get_solver_stats()["descent"]["iterations"]["obj"][-1]
# status_de = optimizer.get_solver_stats()["descent"]["success"]
#
# print(
#     f"Grid Cost: {grid_cost_val:.2f} | Fuel Cost: {fuel_cost_val:.2f} | Objective climb: {obj_cl:.2f} | Success climb:{status_cl} | Objective cruise: {obj_cr:.2f} | Success cruise:{status_cr} | Objective descent: {obj_de:.2f} | Success descent:{status_de} ")


optimizer_contrail = optimizer.trajectory(objective=objective, interpolant=interpolant,
                                          initial_guess=optimizer_wind)

# Evaluate components post-solution
grid_cost_val = float(np.sum(optimizer_contrail["grid_cost"]))
fuel_cost_val = float(np.sum(optimizer_contrail["fuel_cost"]))

obj = optimizer.solver.stats()["iterations"]["obj"][-1]
status = optimizer.solver.stats()['success']
print(
    f"Grid Cost: {grid_cost_val:.2f} | Fuel Cost: {fuel_cost_val:.2f} | Objective: {obj:.2f} | Success:{status}")

# obj_cr = optimizer.get_solver_stats()["cruise"]["iterations"]["obj"][-1]
# status_cr = optimizer.get_solver_stats()["cruise"]["success"]
# obj_cl = optimizer.get_solver_stats()["climb"]["iterations"]["obj"][-1]
# status_cl = optimizer.get_solver_stats()["climb"]["success"]
# obj_de = optimizer.get_solver_stats()["descent"]["iterations"]["obj"][-1]
# status_de = optimizer.get_solver_stats()["descent"]["success"]
#
# print(
#     f"Grid Cost: {grid_cost_val:.2f} | Fuel Cost: {fuel_cost_val:.2f} | Objective climb: {obj_cl:.2f} | Success climb:{status_cl} | Objective cruise: {obj_cr:.2f} | Success cruise:{status_cr} | Objective descent: {obj_de:.2f} | Success descent:{status_de} ")

