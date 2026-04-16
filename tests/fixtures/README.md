# Test fixtures

## contrail_4d.casadi

A 4D CasADi bspline interpolant of ATR20 contrail cost for a real-world
route (EDDB → LEMD) on 2023-01-05, built from ERA5 meteo fields with a
Gaussian smoothing (σ=2) applied to the raw contrail forcing grid.

Axes (in this order):
- longitude (degrees)
- latitude (degrees)
- height (metres)
- ts (seconds from start)

Used by `tests/test_grid_4d.py`, `tests/benchmark.py`, and
`tests/compare_nlp_scaling.py` to exercise the time-dependent grid-cost
optimization path with physically realistic inputs.

**Regeneration:** the upstream pipeline (ERA5 download → contrail physics →
Gaussian smoothing → bspline fit) lives in `debug/epsilon_constraint/`
and requires external dependencies (`traffic`, `fastmeteo`, ERA5 Zarr
store). If the fixture ever needs to be rebuilt with different routing
or dates, run those scripts and copy the resulting `.casadi` here.

## complete_flight_golden.json

Baseline objective/fuel/iters for the CompleteFlight EHAM→LGAV A320
golden-smoke regression test. Recorded at commit f32255a.
