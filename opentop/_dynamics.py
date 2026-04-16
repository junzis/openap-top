"""Pure-function trajectory dynamics and initial-guess builders.

No state: every function takes inputs, returns outputs. Base and phase
classes delegate to these.
"""

from __future__ import annotations

from typing import Any

import casadi as ca
import openap.casadi as oc
from openap.aero import fpm, ft, kts

import numpy as np

from ._types import Symbolic


def collocation_coeff(polydeg: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute Legendre collocation coefficients.

    Returns:
        tuple: (C, D, B) where C is the derivative matrix,
            D is the continuity vector, B is the quadrature vector.
    """
    # Get collocation points using Legendre polynomials
    tau_root = np.append(0, ca.collocation_points(polydeg, "legendre"))

    # C[i,j] = time derivative of Lagrange polynomial i evaluated at collocation point j
    C = np.zeros((polydeg + 1, polydeg + 1))

    # D[j] = Lagrange polynomial j evaluated at final time (t=1)
    D = np.zeros(polydeg + 1)

    # B[j] = integral of Lagrange polynomial j from 0 to 1
    B = np.zeros(polydeg + 1)

    # For each collocation point, construct Lagrange polynomial and calculate
    # integration coefficients.
    for j in range(polydeg + 1):
        # Construct Lagrange polynomial: 1 at tau_root[j], 0 at tau_root[r] for r != j
        p = np.poly1d([1])
        for r in range(polydeg + 1):
            if r != j:
                p *= np.poly1d([1, -tau_root[r]]) / (tau_root[j] - tau_root[r])

        # Evaluate polynomial at t=1 for continuity constraints
        D[j] = p(1.0)

        # Get time derivative coefficients for collocation constraints
        pder = np.polyder(p)
        for r in range(polydeg + 1):
            C[j, r] = pder(tau_root[r])

        # Get integral coefficients for cost function quadrature
        pint = np.polyint(p)
        B[j] = pint(1.0)

    return C, D, B


def xdot(
    x: Symbolic,
    u: Symbolic,
    *,
    fuelflow: Any,
    dT: Symbolic,
    wind: Any = None,
) -> ca.MX:
    """State derivatives for the equations of motion. Pure.

    Args:
        x: State vector [xp (m), yp (m), h (m), mass (kg), ts (s)].
        u: Control vector [mach, vs (m/s), heading (rad)].
        fuelflow: openap.casadi.FuelFlow instance.
        dT: ISA temperature offset (K).
        wind: optional PolyWind instance.

    Returns:
        ca.MX: State derivatives [dx, dy, dh, dm, dt].
    """
    xp, yp, h, m, ts = x[0], x[1], x[2], x[3], x[4]  # type: ignore[index]  # Symbolic includes float but x is always array-like at runtime
    mach, vs, psi = u[0], u[1], u[2]  # type: ignore[index]

    v = oc.aero.mach2tas(mach, h, dT=dT)
    gamma = ca.arctan2(vs, v)

    dx = v * ca.sin(psi) * ca.cos(gamma)
    if wind is not None:
        dx += wind.calc_u(xp, yp, h, ts)

    dy = v * ca.cos(psi) * ca.cos(gamma)
    if wind is not None:
        dy += wind.calc_v(xp, yp, h, ts)

    dh = vs

    dm = -fuelflow.enroute(m, v / kts, h / ft, vs / fpm, dT=dT)

    dt = 1

    return ca.vertcat(dx, dy, dh, dm, dt)  # type: ignore[return-value]  # casadi stubs widen vertcat to DM|Unknown


def great_circle_init(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
    *,
    n_nodes: int,
    mass_init: float,
    aircraft: dict[str, Any],
    proj: Any,
    flight: Any = None,
) -> np.ndarray:
    """Build the great-circle state guess (or interpolate from a flight DataFrame).

    Returns np.ndarray of shape (n_nodes+1, 5) with columns [xp, yp, h, mass, ts].
    """
    m_guess = mass_init * np.ones(n_nodes + 1)
    ts_guess = np.linspace(0, 6 * 3600, n_nodes + 1)

    if flight is None:
        h_cr = aircraft["cruise"]["height"]
        xp_0, yp_0 = proj(lon1, lat1)
        xp_f, yp_f = proj(lon2, lat2)
        xp_guess = np.linspace(xp_0, xp_f, n_nodes + 1)
        yp_guess = np.linspace(yp_0, yp_f, n_nodes + 1)
        h_guess = h_cr * np.ones(n_nodes + 1)
    else:
        xp_guess, yp_guess = proj(flight.longitude, flight.latitude)
        h_guess = flight.altitude * ft
        if "mass" in flight:
            m_guess = flight.mass

        if "ts" in flight:
            ts_guess = flight.ts
        elif "timestamp" in flight:
            ts_guess = (flight.timestamp - flight.timestamp.min()).dt.total_seconds()

    return np.vstack([xp_guess, yp_guess, h_guess, m_guess, ts_guess]).T
