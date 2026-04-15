"""Pure-function objectives and dispatch registry.

Every objective takes ``(x, u, dt, **kwargs)`` and returns a CasADi expression
(or a numpy scalar when ``symbolic=False``). Model context — fuelflow, drag,
emission models, proj, etc. — is passed explicitly as keyword arguments by
the ``Base`` delegate methods; no ``self`` access here.

The ``requires_exact_hessian`` attribute on a callable marks objectives that
need IPOPT's exact Hessian for numerical stability (e.g., grid-cost using
bspline interpolants). ``Base._build_opti`` consults this flag to wire the
solver option instead of mutating ``solver_options`` as a side effect.
"""
from __future__ import annotations

from typing import Any, Callable, Union

import casadi as ca
import numpy as np
import openap
import openap.casadi as oc
from openap.aero import fpm, ft, kts

from ._types import ObjectiveFn, Symbolic


def _mark(fn: ObjectiveFn, *, exact_hessian: bool = False) -> ObjectiveFn:
    """Attach the requires_exact_hessian flag (defaults False)."""
    fn.requires_exact_hessian = exact_hessian  # type: ignore[attr-defined]
    return fn


# ---- Core objectives ----


def obj_fuel(
    x: Symbolic,
    u: Symbolic,
    dt: Symbolic,
    *,
    fuelflow: Any,
    dT: Symbolic,
    actype: str,
    engtype: str,
    use_synonym: bool,
    symbolic: bool = True,
    **kwargs: Any,
) -> ca.MX:
    """Fuel burn objective: fuelflow * dt.

    Symbolic path uses the CasADi ``fuelflow`` passed in; non-symbolic path
    builds a numpy ``openap.FuelFlow`` from actype/engtype so callers can
    evaluate the objective on solved numeric trajectories.
    """
    xp, yp, h, m, ts = x[0], x[1], x[2], x[3], x[4]  # type: ignore[index]  # Symbolic includes float but x is always array-like
    mach, vs, psi = u[0], u[1], u[2]  # type: ignore[index]

    if symbolic:
        ff_model = fuelflow
        v = oc.aero.mach2tas(mach, h, dT=dT)  # type: ignore[arg-type]  # openap stubs say int, float/symbolic works
    else:
        ff_model = openap.FuelFlow(
            actype,
            engtype,
            use_synonym=use_synonym,
            force_engine=True,
        )
        v = openap.aero.mach2tas(mach, h, dT=dT)  # type: ignore[arg-type]

    ff = ff_model.enroute(m, v / kts, h / ft, vs / fpm, dT=dT)
    return ff * dt  # type: ignore[return-value]  # casadi arithmetic result is MX at NLP time


def obj_time(x: Symbolic, u: Symbolic, dt: Symbolic, **kwargs: object) -> ca.MX:
    """Minimum time objective."""
    return dt  # type: ignore[return-value]  # dt is Symbolic; ca.MX at NLP time


def obj_ci(
    x: Symbolic,
    u: Symbolic,
    dt: Symbolic,
    *,
    ci: float,
    time_price: float = 25,
    fuel_price: float = 0.8,
    **kwargs: Any,
) -> ca.MX:
    """Cost index objective blending time and fuel costs.

    Args:
        ci: Cost index (0-100). 0 = fuel only, 100 = time only.
        time_price: Cost of time in EUR/min. Default 25.
        fuel_price: Cost of fuel in EUR/L. Default 0.8.
    """
    fuel = obj_fuel(x, u, dt, **kwargs)

    # time cost 25 eur/min
    time_cost = (dt / 60) * time_price

    # fuel cost 0.8 eur/L, Jet A density 0.82
    fuel_cost = fuel * (fuel_price / 0.82)

    obj = ci / 100 * time_cost + (1 - ci / 100) * fuel_cost
    return obj  # type: ignore[return-value]  # arithmetic on Symbolic yields MX at NLP time


# ---- Climate ----

# Climate metric coefficients: (h2o, nox, sox, soot)
# CO2 coefficient is always 1.
_CLIMATE_COEFF: dict[str, tuple] = {
    "gwp20": (0.22, 619, -832, 4288),
    "gwp50": (0.1, 205, -392, 2018),
    "gwp100": (0.06, 114, -226, 1166),
    "gtp20": (0.07, -222, -241, 1245),
    "gtp50": (0.01, -69, -38, 195),
    "gtp100": (0.008, 13, -31, 161),
}


def obj_climate(
    x: Symbolic,
    u: Symbolic,
    dt: Symbolic,
    *,
    metric: str,
    calc_emission: Callable[..., tuple[Symbolic, ...]],
    **kwargs: Any,
) -> ca.MX:
    """Climate impact objective using GWP/GTP metric coefficients.

    ``calc_emission`` is a callable ``(x, u, **kw) -> (co2, h2o, sox, soot, nox)``
    — typically the bound ``Base._calc_emission`` so the objective inherits
    fuelflow/emission model configuration.
    """
    if metric not in _CLIMATE_COEFF:
        raise ValueError(
            f"Unknown climate metric: {metric!r}. "
            f"Valid: {sorted(_CLIMATE_COEFF)}"
        )
    # ``calc_emission`` (Base._calc_emission) takes only ``symbolic=``; other
    # context entries (fuelflow/dT/proj/etc.) are already bound via self.
    emit_kwargs = {k: v for k, v in kwargs.items() if k in ("symbolic",)}
    co2, h2o, sox, soot, nox = calc_emission(x, u, **emit_kwargs)
    c_h2o, c_nox, c_sox, c_soot = _CLIMATE_COEFF[metric]
    cost = co2 + c_h2o * h2o + c_nox * nox + c_sox * sox + c_soot * soot
    return cost * dt  # type: ignore[return-value]  # arithmetic on Symbolic yields MX at NLP time


# ---- Grid cost ----


def _obj_grid_cost_impl(
    x: Symbolic,
    u: Symbolic,
    dt: Symbolic,
    *,
    proj: Any,
    **kwargs: Any,
) -> ca.MX:
    """Grid-based cost objective using a CasADi interpolant.

    Kwargs:
        interpolant: CasADi interpolant function (required).
        symbolic: Use symbolic computation. Default True.
        n_dim: Input dimension, 3 (lon,lat,h) or 4 (+ts). Default 3.
        time_dependent: Multiply cost by dt. Default True.
    """
    xp, yp, h, m, ts = x[0], x[1], x[2], x[3], x[4]  # type: ignore[index]  # Symbolic includes float but x is always array-like

    interpolant: Callable[..., Any] | None = kwargs.get("interpolant", None)  # type: ignore[assignment]
    symbolic = kwargs.get("symbolic", True)
    n_dim = kwargs.get("n_dim", 3)
    time_dependent = kwargs.get("time_dependent", True)
    if n_dim not in (3, 4):
        raise ValueError(f"n_dim must be 3 or 4, got {n_dim}")

    if interpolant is None:
        raise ValueError("obj_grid_cost requires an 'interpolant' keyword argument")

    lon, lat = proj(xp, yp, inverse=True, symbolic=symbolic)

    if n_dim == 3:
        input_data = [lon, lat, h]
    else:
        input_data = [lon, lat, h, ts]

    if symbolic:
        input_data = ca.vertcat(*input_data)
    else:
        input_data = np.array(input_data)

    cost: Any = interpolant(input_data)

    if not symbolic:
        cost = cost.full()[0]

    if time_dependent:
        cost *= dt

    return cost  # type: ignore[return-value]  # cost is MX at NLP time, Any typed to allow numeric path


# Mark grid-cost as needing exact Hessian so Base can gate the solver option.
obj_grid_cost = _mark(_obj_grid_cost_impl, exact_hessian=True)


# ---- Registry + resolver ----


def _make_climate_entry(metric: str) -> ObjectiveFn:
    def fn(x, u, dt, **kw):
        return obj_climate(x, u, dt, metric=metric, **kw)
    fn.__name__ = f"obj_{metric}"
    return _mark(fn)


_OBJECTIVES: dict[str, ObjectiveFn] = {
    "fuel": _mark(obj_fuel),
    "time": _mark(obj_time),
    "gwp20": _make_climate_entry("gwp20"),
    "gwp50": _make_climate_entry("gwp50"),
    "gwp100": _make_climate_entry("gwp100"),
    "gtp20": _make_climate_entry("gtp20"),
    "gtp50": _make_climate_entry("gtp50"),
    "gtp100": _make_climate_entry("gtp100"),
    "grid_cost": obj_grid_cost,
}


def resolve_objective(spec: str | Callable[..., object]) -> ObjectiveFn:
    """Resolve a user-given spec to an ObjectiveFn.

    Accepts:
        - callable: returned as-is.
        - str in the registry: the pure objective function.
        - "ci:N" format: curried ``obj_ci`` with ``ci=N``.

    Raises:
        ValueError: for unknown strings or malformed ``ci:`` specs.
        TypeError: for non-str non-callable specs.
    """
    if callable(spec):
        return spec  # type: ignore[return-value]  # caller's callable may return Any; MX assumed at NLP time
    if isinstance(spec, str):
        if spec.lower().startswith("ci:"):
            try:
                ci = float(spec.split(":", 1)[1])
            except ValueError:
                raise ValueError(f"Invalid cost-index spec: {spec!r}")

            def fn(x, u, dt, **kw):
                return obj_ci(x, u, dt, ci=ci, **kw)

            fn.__name__ = f"obj_ci_{ci}"
            return _mark(fn)
        if spec in _OBJECTIVES:
            return _OBJECTIVES[spec]
        raise ValueError(
            f"Unknown objective: {spec!r}. "
            f"Valid: {sorted(_OBJECTIVES)} or 'ci:N'."
        )
    raise TypeError(
        f"objective must be str or callable, got {type(spec).__name__}"
    )
