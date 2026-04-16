"""Pin PolyWind numeric and symbolic output. Regression guard for the eval() rewrite."""

import casadi as ca

import numpy as np
import opentop.tools as tools
import pandas as pd
import pyproj


def _fake_wind_df():
    """Synthetic wind field: u = 5 + 0.01*h, v = 2 - 0.001*ts."""
    rows = []
    for lon in np.linspace(0, 10, 4):
        for lat in np.linspace(45, 55, 4):
            for h in (1000, 5000, 10000):
                for ts in (0, 3600):
                    rows.append(
                        {
                            "longitude": lon,
                            "latitude": lat,
                            "h": h,
                            "ts": ts,
                            "u": 5 + 0.01 * h,
                            "v": 2 - 0.001 * ts,
                        }
                    )
    return pd.DataFrame(rows)


def _proj():
    return pyproj.Proj(proj="lcc", lat_1=46, lat_2=54, lat_0=50, lon_0=5)


def test_polywind_numeric_output_sane():
    df = _fake_wind_df()
    proj = _proj()
    w = tools.PolyWind(df, proj, 46.0, 1.0, 54.0, 9.0)
    x, y = proj(5.0, 50.0)
    u = w.calc_u(x, y, 5000, 1800)
    v = w.calc_v(x, y, 5000, 1800)
    assert np.isfinite(float(u))
    assert np.isfinite(float(v))
    # Synthetic u = 55, v = 0.2; ridge regularization → loose bounds.
    assert 20 < float(u) < 90
    assert -10 < float(v) < 10


def test_polywind_symbolic_path_returns_casadi_expr():
    """When inputs are CasADi SX, output must be a symbolic expression, not a float."""
    df = _fake_wind_df()
    proj = _proj()
    w = tools.PolyWind(df, proj, 46.0, 1.0, 54.0, 9.0)
    x = ca.SX.sym("x")  # type: ignore[arg-type]  # CasADi stubs wrong: SX.sym(str) is valid
    y = ca.SX.sym("y")  # type: ignore[arg-type]
    h = ca.SX.sym("h")  # type: ignore[arg-type]
    ts = ca.SX.sym("ts")  # type: ignore[arg-type]
    u = w.calc_u(x, y, h, ts)
    v = w.calc_v(x, y, h, ts)
    # Result must be a CasADi SX (symbolic), not a plain float.
    assert isinstance(u, ca.SX) or isinstance(u, ca.MX)
    assert isinstance(v, ca.SX) or isinstance(v, ca.MX)


def test_polywind_numeric_matches_symbolic_evaluation():
    """Same input via numeric and symbolic paths should produce the same number."""
    df = _fake_wind_df()
    proj = _proj()
    w = tools.PolyWind(df, proj, 46.0, 1.0, 54.0, 9.0)
    x_num, y_num = proj(5.0, 50.0)
    # numeric path
    u_num = float(w.calc_u(x_num, y_num, 5000, 1800))
    # symbolic path, then substitute
    x, y, h, ts = ca.SX.sym("x"), ca.SX.sym("y"), ca.SX.sym("h"), ca.SX.sym("ts")  # type: ignore[arg-type]  # CasADi stubs wrong
    u_sym = w.calc_u(x, y, h, ts)
    f = ca.Function("f", [x, y, h, ts], [u_sym])
    u_eval = float(f(x_num, y_num, 5000, 1800))  # type: ignore[arg-type]  # CasADi Function.__call__ return type opaque to pyright
    assert abs(u_num - u_eval) < 1e-6, f"numeric {u_num} != symbolic eval {u_eval}"
