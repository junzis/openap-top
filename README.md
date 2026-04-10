# OpenAP Trajectory Optimizer

> **v2.0 — major rewrite.** The NLP construction has moved to CasADi's Opti stack, eliminating ~400 lines of boilerplate and enabling several bug fixes and cleanups. Fuel-optimal results are bit-identical to v1.11.0. A few APIs have changed — see [What's New in 2.0](#whats-new-in-20) below.

Flight trajectory optimizer based on the [OpenAP](https://github.com/junzis/openap) aircraft performance model.

`openap-top` uses non-linear optimal control via direct collocation (CasADi + IPOPT) to generate optimal flight trajectories. It provides simple interfaces for:

- Complete flight trajectories (takeoff → cruise → landing)
- Individual phases: climb, cruise, descent
- Fuel-optimal, time-optimal, cost-index, and climate-optimal objectives
- Wind integration
- Custom 4D grid cost functions (contrails, weather, airspace)
- User-defined objective functions and constraints

## 🕮 User Guide

Detailed guide and examples: <https://openap.dev/optimize>.

## Install

From PyPI:

```sh
pip install --upgrade openap-top
```

From the development branch:

```sh
pip install --upgrade git+https://github.com/junzis/openap
pip install --upgrade git+https://github.com/junzis/openap-top
```

`openap-top` is a namespace extension of `openap` and is imported as `openap.top`.

## Quick Start

### A simple optimal flight

```python
from openap import top

optimizer = top.CompleteFlight("A320", "EHAM", "LGAV", m0=0.85)
flight = optimizer.trajectory(objective="fuel")
```

`flight` is a Pandas DataFrame with columns for position, altitude, mass,
Mach, TAS, vertical rate, heading, and per-segment `fuel_cost`.

![example_optimal_flight](./docs/_static/flight_dataframe.png)

### Other built-in objectives

```python
optimizer.trajectory(objective="time")      # minimum time
optimizer.trajectory(objective="ci:30")     # cost index 30
optimizer.trajectory(objective="gwp100")    # 100-yr global warming potential
optimizer.trajectory(objective="gtp100")    # 100-yr global temperature potential
```

The supported climate metrics are `gwp20`, `gwp50`, `gwp100`, `gtp20`, `gtp50`, `gtp100`.

### Choosing a different engine

```python
optimizer = top.CompleteFlight(
    "A320", "EHAM", "LGAV", m0=0.85, engine="CFM56-5B4"
)
```

### Flight phase optimizers

```python
cruise    = top.Cruise("A320", "EHAM", "LGAV", m0=0.85).trajectory()
climb     = top.Climb("A320", "EHAM", "LGAV", m0=0.85).trajectory()
descent   = top.Descent("A320", "EHAM", "LGAV", m0=0.85).trajectory()

# Multi-phase: optimizes climb, cruise, descent sequentially
full = top.MultiPhase("A320", "EHAM", "LGAV", m0=0.85).trajectory()
```

`Cruise` also supports constant-altitude, constant-Mach, and fixed-track modes:

```python
opt = top.Cruise("A320", "EHAM", "LGAV", m0=0.85)
opt.fix_cruise_altitude()
opt.fix_mach_number()
opt.fix_track_angle()
flight = opt.trajectory()
```

### Wind integration

Download ERA5 (or similar) meteorological data in GRIB format, then:

```python
from openap import top

windfield = top.tools.read_grids("wind.grib")

optimizer = top.CompleteFlight("A320", "EHAM", "LGAV", m0=0.85)
optimizer.enable_wind(windfield)

flight = optimizer.trajectory(objective="fuel")
top.vis.trajectory(flight, windfield=windfield, barb_steps=15)
```

![example_optimal_flight](./docs/_static/optimal_flight_complete_example.png)

### Custom grid cost (contrails, weather, airspace)

Build a CasADi interpolant from a DataFrame with columns `longitude`, `latitude`, `height` (m), `cost`, and optionally `ts`:

```python
interpolant = top.tools.interpolant_from_dataframe(df_cost)

def contrail_objective(x, u, dt, **kwargs):
    grid_cost = optimizer.obj_grid_cost(
        x, u, dt, interpolant=kwargs["interpolant"], n_dim=3
    )
    fuel_cost = optimizer.obj_fuel(x, u, dt)
    return grid_cost + fuel_cost

flight = optimizer.trajectory(
    objective=contrail_objective,
    interpolant=interpolant,
    n_dim=3,
)
```

See <https://openap.dev/optimize/contrails.html> for a full contrail + CO₂ example.

### Custom objective functions

Any callable with signature `(x, u, dt, **kwargs) -> ca.MX` can be used:

```python
def my_objective(x, u, dt, **kwargs):
    # x: state [xp, yp, h, mass, ts]
    # u: control [mach, vs, psi]
    return your_cost_expression

flight = optimizer.trajectory(objective=my_objective)
```

## Accessing Solver Results

After calling `.trajectory()`, the optimizer exposes:

```python
optimizer.solver          # CasADi OptiSol object
optimizer.solver.stats()  # solver statistics dict ("success", "iter_count", ...)
optimizer.objective_value # final objective value (float)
```

## Benchmarks

Run benchmarks across versions to verify performance:

```sh
./benchmark.sh              # Benchmark HEAD (local dev code)
./benchmark.sh 2.0.0        # Benchmark a specific PyPI release
./benchmark.sh v1.11.0 v2.0.0  # Benchmark multiple versions sequentially
```

Reports land in `tests/benchmarks/<version>.txt`.

## What's New in 2.0

Version 2.0 is a major refactor built on CasADi's Opti stack. Most user code needs no changes, but a few APIs have moved:

| v1.x | v2.0 |
|---|---|
| `optimizer.change_engine("CFM56-5B4")` | `top.Cruise(..., engine="CFM56-5B4")` |
| `optimizer.solution["f"]` | `optimizer.objective_value` |
| `optimizer.solver` was a `ca.nlpsol` callable | Now a `ca.OptiSol` object |
| `setup(max_iteration=..., max_iterations=...)` | `setup(max_iter=...)` |

See the [changelog](https://github.com/junzis/openap-top/releases) for details.

## License

GNU LGPL v3
