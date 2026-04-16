"""Golden-smoke regression for CompleteFlight.

If this test fails by >1% on the objective, something in a subsequent
Phase 3 / Phase 4 commit changed the optimization outcome on the
canonical EHAM→LGAV route. Investigate before proceeding — the refactor
is not meant to shift optima by more than noise.
"""

import json
from pathlib import Path

import opentop as top

GOLDEN = Path(__file__).parent / "fixtures" / "complete_flight_golden.json"
TOLERANCE = 0.01  # 1% across refactor


def test_complete_flight_golden_objective_within_1pct():
    with GOLDEN.open() as f:
        record = json.load(f)

    opt = top.CompleteFlight(record["aircraft"], "EHAM", "LGAV", m0=record["m0"])
    opt.setup(max_iter=1500)
    df = opt.trajectory(objective=record["objective_spec"])

    assert df is not None

    assert opt.objective_value is not None
    obj_now = float(opt.objective_value)
    baseline = record["objective"]
    drift = abs(obj_now - baseline) / baseline

    # Check drift first, so a cap-hit doesn't mask a numerical regression.
    assert drift < TOLERANCE, (
        f"golden-smoke drift {drift * 100:.3f}% > {TOLERANCE * 100:.1f}%: "
        f"baseline={baseline} now={obj_now} "
        f"success={opt.success} iters={opt.stats['iter_count']} "
        f"(baseline recorded at {record['commit_sha']})"
    )
    # Then sanity-check the solver did converge (< max_iter cap).
    assert opt.success, f"solver failed: {opt.stats}"
