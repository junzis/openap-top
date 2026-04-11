# OpenAP Trajectory Optimizer

> **v2.0** is a major refactor: (1) the NLP construction has moved to CasADi's Opti stack, (2) the module has been renamed from ``openap.top`` to a standalone ``opentop`` package, and (3) the CLI ``opentop`` is new. See [What's New in 2.0](#whats-new-in-20) below.

Flight trajectory optimizer based on the [OpenAP](https://github.com/junzis/openap) aircraft performance model.

`opentop` uses non-linear optimal control via direct collocation (CasADi + IPOPT) to generate optimal flight trajectories. It provides simple interfaces for:

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
pip install --upgrade opentop
```

From the development branch:

```sh
pip install --upgrade git+https://github.com/junzis/openap
pip install --upgrade git+https://github.com/junzis/openap-top
```

`opentop` is a standalone package. Prior to v2.0 it shipped as `openap.top`, a namespace extension of `openap`; v2.0 drops that and installs as a top-level `opentop` package instead.

## Quick Start

### A simple optimal flight

```python
import opentop

optimizer = opentop.CompleteFlight("A320", "EHAM", "LGAV", m0=0.85)
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
optimizer = opentop.CompleteFlight(
    "A320", "EHAM", "LGAV", m0=0.85, engine="CFM56-5B4"
)
```

### Flight phase optimizers

```python
cruise    = opentop.Cruise("A320", "EHAM", "LGAV", m0=0.85).trajectory()
climb     = opentop.Climb("A320", "EHAM", "LGAV", m0=0.85).trajectory()
descent   = opentop.Descent("A320", "EHAM", "LGAV", m0=0.85).trajectory()

# Multi-phase: optimizes climb, cruise, descent sequentially
full = opentop.MultiPhase("A320", "EHAM", "LGAV", m0=0.85).trajectory()
```

`Cruise` also supports constant-altitude, constant-Mach, and fixed-track modes:

```python
opt = opentop.Cruise("A320", "EHAM", "LGAV", m0=0.85)
opt.fix_cruise_altitude()
opt.fix_mach_number()
opt.fix_track_angle()
flight = opt.trajectory()
```

### Wind integration

Download ERA5 (or similar) meteorological data in GRIB format, then:

```python
import opentop

windfield = opentop.tools.read_grids("wind.grib")

optimizer = opentop.CompleteFlight("A320", "EHAM", "LGAV", m0=0.85)
optimizer.enable_wind(windfield)

flight = optimizer.trajectory(objective="fuel")
opentop.vis.trajectory(flight, windfield=windfield, barb_steps=15)
```

![example_optimal_flight](./docs/_static/optimal_flight_complete_example.png)

### Custom grid cost (contrails, weather, airspace)

Build a CasADi interpolant from a DataFrame with columns `longitude`, `latitude`, `height` (m), `cost`, and optionally `ts`:

```python
interpolant = opentop.tools.interpolant_from_dataframe(df_cost)

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

### Improving convergence on hard blended objectives

IPOPT's termination tolerances assume an objective of magnitude O(1). On non-convex blended objectives (contrail + CO₂ being the canonical case), the total cost can span several orders of magnitude, and the solver may hit `max_iter` without converging or drift to a worse local optimum.

The opt-in `auto_rescale_objective=True` kwarg evaluates the objective at the initial guess, divides the symbolic objective by `max(|f(x0)|, 1.0)`, and restores physical units in `objective_value` post-solve. Mathematically this is a no-op on the optimal `x`, but it re-calibrates IPOPT's tolerance semantics.

```python
optimizer = opentop.CompleteFlight("A320", "EHAM", "LGAV", m0=0.85)
warmstart = optimizer.trajectory(objective="fuel")

flight = optimizer.trajectory(
    objective=contrail_co2_blend,
    interpolant=interpolant,
    initial_guess=warmstart,
    n_dim=4,
    time_dependent=True,
    auto_rescale_objective=True,   # rescue stalled solves on blended objectives
)
```

On the `EHAM-LGAV` CompleteFlight contrail + CO₂ case, this turns a `max_iter` failure (~300 s wall time) into a ~100-iter success (~15 s) — see `debug/scaling/investigation/` for the full empirical matrix. The recipe is: start from a fuel-only warmstart, then re-solve with the blended objective and `auto_rescale_objective=True`.

Default is `False` — behavior is unchanged unless you opt in.

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
./benchmark.sh                 # Benchmark HEAD (local dev code)
./benchmark.sh v2.0.0          # Benchmark a specific PyPI release
./benchmark.sh v1.11.0 v2.0.0  # Benchmark multiple versions sequentially
```

Reports land in `tests/benchmarks/<version>.txt`.

## What's New in 2.0

Version 2.0 is a major refactor. Most user code keeps the same shape, but a few things have moved:

| v1.x | v2.0 |
|---|---|
| `pip install openap-top` | `pip install opentop` |
| `from openap import top` | `import opentop` |
| `top.Cruise(...)` | `opentop.Cruise(...)` |
| `optimizer.change_engine()` dropped | `opentop.Cruise(..., engine="CFM56-5B4")` |
| `optimizer.solution["f"]` | `optimizer.objective_value` |
| `optimizer.solver` was a `ca.nlpsol` callable | now a `ca.OptiSol` object |
| `setup(max_iteration=...)` | `setup(max_iter=...)` |
| — | new CLI: `opentop optimize ORIGIN DEST ...` and `opentop gengrid ...` |
| — | new `auto_rescale_objective=True` kwarg on `trajectory()` |
| — | new `opentop.tools.cached_interpolant_from_dataframe()` for disk-cached bspline interpolants |

The NLP construction moved to CasADi's Opti stack, which removed ~400 lines of boilerplate and cleaned up several bugs. The module rename from `openap.top` to `opentop` eliminates the namespace-extension install mode that used to require `.pth` tricks.

See the [changelog](https://github.com/junzis/openap-top/releases) for details.

## License

GNU LGPL v3
