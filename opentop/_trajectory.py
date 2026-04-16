"""Numeric-trajectory -> DataFrame conversion. Pure functions.

Turns solver output (numeric state/control arrays plus final time) into the
trajectory DataFrame. No CasADi symbolic ops, no solver state; context
(proj, aircraft, wind, etc.) flows in as explicit arguments.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

import numpy as np
import openap
import pandas as pd
from openap.aero import fpm, ft, kts

from . import _objectives

if TYPE_CHECKING:
    from .tools import PolyWind  # forward-ref; avoids circular import


def to_dataframe(
    ts_final: float,
    x_opt: np.ndarray,
    u_opt: np.ndarray,
    *,
    proj: Any,
    nodes: int,
    dT: float = 0.0,
    wind: Optional["PolyWind"] = None,
    actype: str,
    engtype: str,
    use_synonym: bool = False,
    interpolant: Any = None,
    time_dependent: bool = True,
    n_dim: Optional[int] = None,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, float]:
    """Build a trajectory DataFrame from numeric states, controls, and final time.

    Args:
        ts_final: Final timestamp (scalar).
        x_opt: Optimized states, shape (5, N+1) — [xp, yp, h, mass, ts].
        u_opt: Optimized controls, shape (3, N) — [mach, vs, psi].
        proj: pyproj-style projection callable.
        nodes: Number of control intervals (so the state has ``nodes + 1`` columns).
        dT: ISA temperature offset, Kelvin.
        wind: Optional PolyWind-like object providing ``calc_u``/``calc_v``.
        actype: Aircraft type string (for FuelFlow).
        engtype: Engine type string (for FuelFlow).
        use_synonym: Forwarded to ``openap.FuelFlow``.
        interpolant: Optional CasADi grid-cost interpolant.
        time_dependent: Whether grid cost is time-dependent.
        n_dim: 3 or 4; auto-detected from interpolant if None.

    Returns:
        (df, X, U, dt): the trajectory DataFrame plus the state array (with
        the extrapolated terminal control appended to ``U``) and the segment
        duration ``dt = ts_final / nodes``. The caller assigns these back onto
        the Base instance to preserve historical side-effect semantics.
    """
    if n_dim is None:
        n_dim = interpolant.size1_in(0) if interpolant is not None else 3
    if n_dim not in (3, 4):
        raise ValueError(f"n_dim must be 3 or 4, got {n_dim}")

    X = x_opt if isinstance(x_opt, np.ndarray) else x_opt.full()
    U = u_opt if isinstance(u_opt, np.ndarray) else u_opt.full()

    # Extrapolate the final control point, Uf
    U2 = U[:, -2:-1]
    U1 = U[:, -1:]
    Uf = U1 + (U1 - U2)

    U = np.append(U, Uf, axis=1)
    n = nodes + 1

    dt = ts_final / (n - 1)

    xp, yp, h, mass, ts = X
    mach, vs, psi = U
    lon, lat = proj(xp, yp, inverse=True)
    ts_ = np.linspace(0, ts_final, n).round(4)
    tas = (openap.aero.mach2tas(mach, h, dT=dT) / kts).round(4)  # type: ignore[arg-type]  # openap stubs say int, float works
    alt = (h / ft).round()
    vertrate = (vs / fpm).round()

    # Per-segment fuel cost derived directly from the mass trajectory.
    # mass[k] - mass[k+1] is the exact fuel burnt on the [k, k+1] interval
    # as enforced by the collocation dynamics (higher-order quadrature);
    # this guarantees `fuel_cost.sum() == m0 - m_final` up to floating
    # point. Recomputing via obj_fuel() would use left-endpoint rectangular
    # quadrature and disagree with the physical fuel burn.
    fuel_cost = np.append(-np.diff(mass), np.nan)

    # Grid cost has no state-based equivalent; integrate left-endpoint
    # over the N intervals and pad the terminal row with NaN.
    if interpolant is not None:
        grid_cost_seg = np.asarray(
            _objectives.obj_grid_cost(
                X[:, :-1],
                U[:, :-1],
                dt,
                proj=proj,
                interpolant=interpolant,
                time_dependent=time_dependent,
                n_dim=n_dim,
                symbolic=False,
            )
        ).ravel()
        grid_cost = np.append(grid_cost_seg, np.nan)
    else:
        grid_cost = np.full(n, np.nan)

    df = pd.DataFrame(
        dict(
            mass=mass,
            ts=ts_,
            x=xp,
            y=yp,
            h=h,
            latitude=lat,
            longitude=lon,
            altitude=alt,
            mach=mach.round(6),
            tas=tas,
            vertical_rate=vertrate,
            heading=(np.rad2deg(psi) % 360).round(4),
            fuel_cost=fuel_cost,
            grid_cost=grid_cost,
        )
    )

    fuelflow = openap.FuelFlow(
        actype,
        engtype,
        use_synonym=use_synonym,
        force_engine=True,
    )

    df = df.assign(
        fuelflow=(fuelflow.enroute(mass=df.mass, tas=tas, alt=alt, vs=vertrate, dT=dT))
    )

    if wind:
        wu = np.array(
            [wind.calc_u(xi, yi, hi, ti) for xi, yi, hi, ti in zip(xp, yp, h, ts)]
        )
        wv = np.array(
            [wind.calc_v(xi, yi, hi, ti) for xi, yi, hi, ti in zip(xp, yp, h, ts)]
        )
        df = df.assign(wu=wu, wv=wv)

    return df, X, U, dt
