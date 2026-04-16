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

**Regeneration:** run `tests/fixtures/build_contrail_4d.py` via `uv run`;
the script has PEP 723 inline deps (`fastmeteo`, `scipy`, `openap`) so no
project-level install is needed. Requires access to an ERA5 Zarr store
(default `/tmp/era5-zarr`; first run downloads ~GB of data from the ARCO
public mirror).

    uv run tests/fixtures/build_contrail_4d.py \
        --origin 52.362,13.501 --dest 40.472,-3.563 \
        --start 2023-01-05T09:48 --stop 2023-01-05T12:00 \
        --sigma 2 \
        --out tests/fixtures/contrail_4d.casadi

## complete_flight_golden.json

Baseline objective/fuel/iters for the CompleteFlight EHAM→LGAV A320
golden-smoke regression test. Recorded at commit f32255a.
