import time
import warnings

import openap.top as otop

warnings.filterwarnings("ignore")

actype = "A320"
origin = "EHAM"
destination = "LGAV"
m0 = 0.85

optimizer = otop.CompleteFlight(actype, origin, destination, m0)
# optimizer = otop.Cruise(actype, origin, destination, m0)
# optimizer = otop.Climb(actype, origin, destination, m0)
# optimizer = otop.Descent(actype, origin, destination, m0)

start = time.time()

flight = optimizer.trajectory(objective="fuel")
# flight = optimizer.trajectory(objective="ci:30")
# flight = optimizer.trajectory(objective="gwp100")
# flight = optimizer.trajectory(objective="gtp100")


print(flight)

print(f"\nOptimal trajectory was generated in {round(time.time() - start)} seconds.\n")
