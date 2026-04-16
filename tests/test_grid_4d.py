"""Integration: time-dependent (4D) grid cost on a short cruise.

Pins the obj_grid_cost(n_dim=4, time_dependent=True) path before Phase 3
restructures it. Uses a cached bspline interpolant under tests/fixtures/
so this test re-runs in ~1s plus solver time.
"""

from pathlib import Path

import pytest

import opentop as top
import opentop.tools as tools

FIXTURE = Path(__file__).parent / "fixtures" / "synthetic_4d.casadi"


@pytest.fixture(scope="module")
def interp_4d():
    if not FIXTURE.exists():
        pytest.skip(
            f"{FIXTURE} not built; "
            "run `uv run --with '.' python tests/fixtures/build_synthetic_4d.py`"
        )
    return tools.load_interpolant(str(FIXTURE))


def test_cruise_with_4d_grid_cost_converges(interp_4d):
    opt = top.Cruise("A320", (52.308, 4.764), (50.033, 8.570), m0=0.85)
    opt.setup(max_iter=800)

    def blended(x, u, dt, **kwargs):
        grid = opt.obj_grid_cost(
            x,
            u,
            dt,
            interpolant=kwargs["interpolant"],
            n_dim=4,
            time_dependent=True,
        )
        return grid + opt.obj_fuel(x, u, dt)

    df = opt.trajectory(
        objective=blended,
        interpolant=interp_4d,
        n_dim=4,
        time_dependent=True,
    )
    assert df is not None, "trajectory returned None"
    assert opt.solver.stats()["success"], f"solver failed: {opt.solver.stats()}"
    assert "grid_cost" in df.columns, "grid_cost column missing from trajectory"  # type: ignore[union-attr]  # trajectory() without result_object always returns DataFrame
