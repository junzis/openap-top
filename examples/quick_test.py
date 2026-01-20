# %%
import time
import warnings

import numpy as np

from openap import top

# %%
warnings.filterwarnings("ignore")

actype = "A320"
origin = "EHAM"
destination = "LGAV"
m0 = 0.85

optimizer = top.CompleteFlight(actype, origin, destination, m0)
# optimizer = top.MultiPhase(actype, origin, destination, m0)
# optimizer = top.Cruise(actype, origin, destination, m0)
# optimizer = top.Climb(actype, origin, destination, m0)
# optimizer = top.Descent(actype, origin, destination, m0)

start = time.time()

flight = optimizer.trajectory(objective="fuel")

# Demonstrate accessing the new fuel_cost column
fuel_cost_val = float(np.sum(flight["fuel_cost"]))
obj = optimizer.solver.stats()["iterations"]["obj"][-1]
status = optimizer.solver.stats()["success"]
print(f"Fuel Cost: {fuel_cost_val:.2f} | Objective: {obj:.2f} | Status: {status}")

# flight = optimizer.trajectory(objective="ci:30")
# flight = optimizer.trajectory(objective="gwp100")
# flight = optimizer.trajectory(objective="gtp100")
# flight = optimizer.trajectory(objective=("ci:90", "ci:10", "ci:20"))  # Multiphase

print(flight)

print(f"\nOptimal trajectory was generated in {round(time.time() - start)} seconds.\n")

