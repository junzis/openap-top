import time
import warnings

from openap import top

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
# flight = optimizer.trajectory(objective="ci:30")
# flight = optimizer.trajectory(objective="gwp100")
# flight = optimizer.trajectory(objective="gtp100")
# flight = optimizer.trajectory(objective=("ci:90", "ci:10", "ci:20"))  # Multiphase


print(flight)

print(f"\nOptimal trajectory was generated in {round(time.time() - start)} seconds.\n")
