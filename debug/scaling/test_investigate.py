"""Unit tests for investigate.py pure helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from investigate import slice_grid  # noqa: E402


def _make_grid_fixture() -> pd.DataFrame:
    """4 timestamps x 3 heights x 5 lats x 5 lons = 300 rows."""
    rows = []
    for ts_idx, ts_sec in enumerate([0, 3600, 7200, 10800]):
        timestamp = pd.Timestamp("2022-02-20 10:00:00", tz="UTC") + pd.Timedelta(
            seconds=ts_sec
        )
        for h in [9000.0, 10000.0, 11000.0]:
            for lat in [40.0, 42.0, 44.0, 46.0, 48.0]:
                for lon in [-5.0, 0.0, 5.0, 10.0, 15.0]:
                    rows.append(
                        dict(
                            timestamp=timestamp,
                            ts=float(ts_sec),
                            height=h,
                            latitude=lat,
                            longitude=lon,
                            cost=float(ts_idx + h / 1000 + lat + lon),
                        )
                    )
    return pd.DataFrame(rows)


def test_slice_grid_filters_by_bbox_and_rebases_ts():
    df = _make_grid_fixture()
    sliced = slice_grid(
        df,
        t0="2022-02-20 10:00:00+00:00",
        t1="2022-02-20 12:00:00+00:00",
        lat_min=41.0,
        lat_max=47.0,
        lon_min=-1.0,
        lon_max=11.0,
    )
    # time: 3 timestamps (0s, 3600s, 7200s); lats in (41, 47): 42, 44, 46; lons in (-1, 11): 0, 5, 10
    assert sliced.latitude.unique().tolist() == [42.0, 44.0, 46.0]
    assert sorted(sliced.longitude.unique().tolist()) == [0.0, 5.0, 10.0]
    assert sliced.ts.min() == 0.0
    assert sliced.ts.max() == 7200.0
    assert sliced.ts.nunique() == 3
    # Required columns for top.tools.interpolant_from_dataframe
    for col in ("ts", "height", "latitude", "longitude", "cost"):
        assert col in sliced.columns


def test_slice_grid_rebases_ts_relative_to_t0():
    df = _make_grid_fixture()
    sliced = slice_grid(
        df,
        t0="2022-02-20 11:00:00+00:00",  # ts=3600 in original
        t1="2022-02-20 13:00:00+00:00",  # ts=10800 in original
        lat_min=40.0,
        lat_max=48.0,
        lon_min=-5.0,
        lon_max=15.0,
    )
    # After rebase: smallest ts should be 0
    assert sliced.ts.min() == 0.0
    # Gap between timestamps preserved: 3600s
    assert sorted(sliced.ts.unique().tolist()) == [0.0, 3600.0, 7200.0]


try:
    from investigate import parse_ipopt_log  # noqa: E402
except ImportError:
    parse_ipopt_log = None


SYNTHETIC_LOG = """\
Ipopt 3.14.17: Running with linear solver MUMPS, Pardiso not available.

List of options:

                                    Name   Value                # times used
                    nlp_scaling_method = gradient-based            1
                             max_iter = 3000                       1

DenseVector "x scaling vector" with 420 elements:
\tx scaling vector[    1]= 1.2345678900000000e-06
\tx scaling vector[    2]= 2.3456789000000000e-06
\tx scaling vector[    3]= 1.0000000000000000e+00
\tx scaling vector[    4]= 9.8765432100000000e-05

DenseVector "c scaling vector" with 315 elements:
\tc scaling vector[    1]= 1.0000000000000000e+00
\tc scaling vector[    2]= 3.3333333333333331e-01

******************************************************************************
This program contains Ipopt, a library for large-scale nonlinear optimization.
******************************************************************************

This is Ipopt version 3.14.17, running with linear solver MUMPS.

Number of nonzeros in equality constraint Jacobian...:     1234
Number of nonzeros in inequality constraint Jacobian.:      567
Number of nonzeros in Lagrangian Hessian.............:     8901

Total number of variables............................:      420
                     variables with only lower bounds:        0
                variables with lower and upper bounds:      420
                     variables with only upper bounds:        0
Total number of equality constraints.................:      315
Total number of inequality constraints...............:      500
        inequality constraints with only lower bounds:      500
   inequality constraints with lower and upper bounds:        0
        inequality constraints with only upper bounds:        0

iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
   0  1.2345678e+00 1.00e-03 1.23e+00 -1.0 0.00e+00    -  0.00e+00 0.00e+00   0
   1  9.8765432e-01 8.00e-04 5.67e-01 -1.0 1.23e-02    -  1.00e+00 1.00e+00f  1
Restoration phase is called at iteration 2.
   2r 9.8765432e-01 8.00e-04 9.99e+02   1.1 0.00e+00    -  0.00e+00 0.00e+00R  1
   3  9.7654321e-01 7.00e-04 3.45e-01 -1.0 5.67e-03    -  1.00e+00 5.00e-01f  2

Number of Iterations....: 3

                                   (scaled)                 (unscaled)
Objective...............:   9.7654321000000000e-01    9.7654321000000000e-01
Dual infeasibility......:   3.4500000000000001e-01    3.4500000000000001e-01
Constraint violation....:   7.0000000000000005e-04    7.0000000000000005e-04
Variable bound violation:   0.0000000000000000e+00    0.0000000000000000e+00
Complementarity.........:   9.9999999999999995e-07    9.9999999999999995e-07
Overall NLP error.......:   3.4500000000000001e-01    3.4500000000000001e-01


Number of objective function evaluations             = 7
EXIT: Solved To Acceptable Level.
"""


def test_parse_ipopt_log_extracts_scaling_factors(tmp_path):
    if parse_ipopt_log is None:
        pytest.skip("parse_ipopt_log not yet implemented")
    log_path = tmp_path / "ipopt.log"
    log_path.write_text(SYNTHETIC_LOG)
    parsed = parse_ipopt_log(log_path)

    assert parsed["x_scaling_count"] == 4
    assert parsed["x_scaling_min"] == pytest.approx(1.2345678900e-06, rel=1e-6)
    assert parsed["x_scaling_max"] == pytest.approx(1.0, rel=1e-6)
    assert parsed["c_scaling_count"] == 2
    assert parsed["c_scaling_min"] == pytest.approx(1.0 / 3, rel=1e-6)
    assert parsed["c_scaling_max"] == pytest.approx(1.0, rel=1e-6)


def test_parse_ipopt_log_counts_restoration_phase(tmp_path):
    if parse_ipopt_log is None:
        pytest.skip("parse_ipopt_log not yet implemented")
    log_path = tmp_path / "ipopt.log"
    log_path.write_text(SYNTHETIC_LOG)
    parsed = parse_ipopt_log(log_path)

    assert parsed["restoration_count"] == 1
    assert parsed["restoration_iterations"] == [2]


def test_parse_ipopt_log_extracts_final_errors(tmp_path):
    if parse_ipopt_log is None:
        pytest.skip("parse_ipopt_log not yet implemented")
    log_path = tmp_path / "ipopt.log"
    log_path.write_text(SYNTHETIC_LOG)
    parsed = parse_ipopt_log(log_path)

    assert parsed["final_scaled_nlp_error"] == pytest.approx(0.345, rel=1e-6)
    assert parsed["final_unscaled_nlp_error"] == pytest.approx(0.345, rel=1e-6)
    assert parsed["final_constraint_violation"] == pytest.approx(7.0e-04, rel=1e-6)
    assert parsed["exit_status"] == "Solved To Acceptable Level."


def test_parse_ipopt_log_missing_file_returns_empty():
    if parse_ipopt_log is None:
        pytest.skip("parse_ipopt_log not yet implemented")
    parsed = parse_ipopt_log(Path("/tmp/does-not-exist-xyz.log"))
    assert parsed == {}
