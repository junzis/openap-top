"""CompleteFlight — full-flight (takeoff to landing) trajectory optimizer."""

from __future__ import annotations

import warnings
from math import pi
from typing import TYPE_CHECKING, Any, Callable

import casadi as ca
import openap.casadi as oc
from openap.aero import fpm, ft, kts

import numpy as np
import pandas as pd

from .base import Base
from ._types import LatLon

try:
    from . import tools
except ImportError:
    warnings.warn("cfgrib and sklearn are required for wind integration")

if TYPE_CHECKING:
    from ._options import TrajectoryResult


class CompleteFlight(Base):
    """Complete flight (takeoff to landing) trajectory optimizer."""

    def __init__(
        self,
        actype: str,
        origin: str | LatLon,
        destination: str | LatLon,
        m0: float = 0.85,
        engine: str | None = None,
        use_synonym: bool = False,
        dT: float = 0.0,
    ) -> None:
        super().__init__(
            actype,
            origin,
            destination,
            m0=m0,
            engine=engine,
            use_synonym=use_synonym,
            dT=dT,
        )

    def init_conditions(self, **kwargs: Any) -> None:
        """Initialize direct collocation bounds and guesses."""

        # Convert lat/lon to Cartesian coordinates.
        xp_0, yp_0 = self.proj(self.lon1, self.lat1)
        xp_f, yp_f = self.proj(self.lon2, self.lat2)
        x_min, x_max, y_min, y_max = self._compute_bbox()

        ts_min = 0
        ts_max = max(5, self.range / 1000 / 500) * 3600

        h_max = kwargs.get("h_max", self.aircraft["limits"]["ceiling"])
        h_min = 100 * ft

        psi = self._compute_bearing_psi()

        # Initial conditions - Lower upper bounds
        self.x_0_lb = [xp_0, yp_0, h_min, self.mass_init, ts_min]
        self.x_0_ub = [xp_0, yp_0, h_min, self.mass_init, ts_min]

        # Final conditions - Lower and upper bounds
        self.x_f_lb = [xp_f, yp_f, h_min, self.oew * 0.5, ts_min]
        self.x_f_ub = [xp_f, yp_f, h_min, self.mass_init, ts_max]

        # States - Lower and upper bounds
        self.x_lb = [x_min, y_min, h_min, self.oew * 0.5, ts_min]
        self.x_ub = [x_max, y_max, h_max, self.mass_init, ts_max]

        # Control init - lower and upper bounds
        self.u_0_lb = [0.1, 500 * fpm, psi]
        self.u_0_ub = [0.3, 2500 * fpm, psi]

        # Control final - lower and upper bounds
        self.u_f_lb = [0.1, -1500 * fpm, psi]
        self.u_f_ub = [0.3, -300 * fpm, psi]

        # Control - Lower and upper bound
        self.u_lb = [0.1, -2500 * fpm, psi - pi / 2]
        self.u_ub = [self.mach_max, 2500 * fpm, psi + pi / 2]

        # Initial guess for the states
        self.x_guess = self.initial_guess()

        # Control - guesses
        self.u_guess = [0.6, 1000 * fpm, psi]

    def trajectory(
        self,
        objective: str | Callable = "fuel",
        *,
        max_fuel: float | None = None,
        return_failed: bool = False,
        initial_guess: pd.DataFrame | None = None,
        remove_cruise: bool = False,
        interpolant: Any = None,
        n_dim: int = 3,
        time_dependent: bool = False,
        auto_rescale_objective: bool = False,
        exact_hessian: bool = False,
        result_object: bool = False,
    ) -> pd.DataFrame | TrajectoryResult:
        """Compute the optimal complete flight trajectory.

        Args:
            objective: Optimization objective. Default "fuel".
            max_fuel: Maximum fuel constraint (kg).
            return_failed: Return result even if optimization fails.
            initial_guess: DataFrame to use as initial guess.
            remove_cruise: Unused for complete flight (accepted for API symmetry).
            interpolant: CasADi grid-cost interpolant (optional).
            n_dim: Interpolant input dimension (3 or 4). Default 3.
            time_dependent: Grid cost is time-dependent. Default False.
            auto_rescale_objective: Rescale objective to O(1). Default False.
            exact_hessian: Force IPOPT exact Hessian. Default False.
            result_object: If True, return a TrajectoryResult.

        Returns:
            pd.DataFrame (or TrajectoryResult if result_object=True).
        """
        _kwargs = {
            "initial_guess": initial_guess,
            "interpolant": interpolant,
            "n_dim": n_dim,
            "time_dependent": time_dependent,
            "auto_rescale_objective": auto_rescale_objective,
            "exact_hessian": exact_hessian,
        }
        self.init_conditions(**_kwargs)

        if initial_guess is not None:
            self.x_guess = self.initial_guess(initial_guess)

        customized_max_fuel = max_fuel

        X, U = self._build_opti(objective, ts_final_guess=7200, **_kwargs)
        opti = self._opti

        # --- Phase-specific constraints ---

        # Constrain altitude during cruise for long flights
        if self.range > 1500_000:
            dd = self.range / (self.nodes + 1)
            max_climb_range = 500_000
            max_descent_range = 300_000
            idx_toc = int(max_climb_range / dd)
            idx_tod = int((self.range - max_descent_range) / dd)

            for k in range(idx_toc, idx_tod):
                # Limit vertical rate during cruise
                opti.subject_to(opti.bounded(-500 * fpm, U[k][1], 500 * fpm))  # type: ignore[arg-type]  # CasADi stubs wrong: bounded(float, expr, float) is valid
                # Minimum cruise alt FL150
                opti.subject_to(X[k][2] >= 15000 * ft)

            for k in range(0, idx_toc):
                opti.subject_to(U[k][1] >= 0)

            for k in range(idx_tod, self.nodes):
                opti.subject_to(U[k][1] <= 0)

        # Force and energy constraints
        for k in range(self.nodes):
            S = self.aircraft["wing"]["area"]
            mass = X[k][3]
            v = oc.aero.mach2tas(U[k][0], X[k][2], dT=self.dT)
            tas = v / kts
            alt = X[k][2] / ft
            rho = oc.aero.density(X[k][2], dT=self.dT)
            thrust_max = self.thrust.cruise(tas, alt, dT=self.dT)
            drag = self.drag.clean(mass, tas, alt, dT=self.dT)

            # max_thrust > drag (5% margin)
            opti.subject_to(thrust_max * 0.95 >= drag)

            # max lift * 80% > weight
            cd0 = self.drag.polar["clean"]["cd0"]
            ck = self.drag.polar["clean"]["k"]
            drag_max = thrust_max * 0.9
            cd_max = drag_max / (0.5 * rho * v**2 * S + 1e-10)
            cl_max = ca.sqrt(ca.fmax(1e-10, (cd_max - cd0) / ck))
            L_max = cl_max * 0.5 * rho * v**2 * S
            opti.subject_to(L_max * 0.8 >= mass * oc.aero.g0)

            # Excess energy > change in potential energy
            excess_energy = (thrust_max - drag) * v - mass * oc.aero.g0 * U[k][1]
            opti.subject_to(excess_energy >= 0)

        # ts and dt consistency
        for k in range(self.nodes - 1):
            opti.subject_to(opti.bounded(-1, X[k + 1][4] - X[k][4] - self.dt, 1))  # type: ignore[arg-type]  # CasADi stubs wrong

        # Smooth Mach number change
        for k in range(self.nodes - 1):
            opti.subject_to(opti.bounded(-0.2, U[k + 1][0] - U[k][0], 0.2))  # type: ignore[arg-type]  # CasADi stubs wrong

        # Smooth vertical rate change
        for k in range(self.nodes - 1):
            opti.subject_to(opti.bounded(-500 * fpm, U[k + 1][1] - U[k][1], 500 * fpm))  # type: ignore[arg-type]  # CasADi stubs wrong

        # Smooth heading change
        for k in range(self.nodes - 1):
            opti.subject_to(
                opti.bounded(-15 * pi / 180, U[k + 1][2] - U[k][2], 15 * pi / 180)  # type: ignore[arg-type]  # CasADi stubs wrong
            )

        # Fuel constraint
        opti.subject_to(opti.bounded(0, X[0][3] - X[-1][3], self.fuel_max))  # type: ignore[arg-type]  # CasADi stubs wrong

        if customized_max_fuel is not None:
            opti.subject_to(X[0][3] - X[-1][3] <= customized_max_fuel)

        # --- Solve ---
        df = self._solve(X, U, **_kwargs)
        df_copy = df.copy()

        if not self._last_solution.stats()["success"]:
            warnings.warn("flight might be infeasible.")

        if df.altitude.max() < 5000:
            warnings.warn("max altitude < 5000 ft, optimization seems to have failed.")
            df = None

        if df is not None:
            final_mass = df.mass.iloc[-1]
            if final_mass < self.oew:
                warnings.warn("final mass condition violated (smaller than OEW).")
                df = None
            if final_mass > self.mlw:
                warnings.warn("final mass condition violated (larger than MLW).")
                df = None

        if return_failed:
            df = df_copy

        if result_object:
            return self._make_result(df)
        return df  # type: ignore[return-value]  # df may be None on failed solves; callers handle this
