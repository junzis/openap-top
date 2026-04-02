import warnings
from math import pi

import casadi as ca
import numpy as np
import openap.casadi as oc
import pandas as pd
from openap.aero import fpm, ft, kts

from .base import Base


class Cruise(Base):
    """Cruise phase trajectory optimizer."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.fix_mach = False
        self.fix_alt = False
        self.fix_track = False
        self.allow_descent = False

    def fix_mach_number(self):
        """Constrain Mach number to be constant during cruise."""
        self.fix_mach = True

    def fix_cruise_altitude(self):
        """Constrain altitude to be constant (no climb/descent)."""
        self.fix_alt = True

    def fix_track_angle(self):
        """Constrain heading to be constant (great circle track)."""
        self.fix_track = True

    def allow_cruise_descent(self):
        """Allow descending during cruise (step descent)."""
        self.allow_descent = True

    def init_conditions(self, **kwargs):
        """Initialize direct collocation bounds and guesses."""

        # Convert lat/lon to cartisian coordinates.
        xp_0, yp_0 = self.proj(self.lon1, self.lat1)
        xp_f, yp_f = self.proj(self.lon2, self.lat2)
        x_min = min(xp_0, xp_f) - 10_000
        x_max = max(xp_0, xp_f) + 10_000
        y_min = min(yp_0, yp_f) - 10_000
        y_max = max(yp_0, yp_f) + 10_000

        ts_min = 0
        ts_max = max(5, self.range / 1000 / 500) * 3600

        h_max = kwargs.get("h_max", self.aircraft["limits"]["ceiling"])
        h_min = kwargs.get("h_min", 15_000 * ft)

        hdg = oc.geo.bearing(self.lat1, self.lon1, self.lat2, self.lon2)
        psi = hdg * pi / 180

        # Initial conditions - Lower upper bounds
        self.x_0_lb = [xp_0, yp_0, h_min, self.mass_init, ts_min]
        self.x_0_ub = [xp_0, yp_0, h_max, self.mass_init, ts_min]

        # Final conditions - Lower and upper bounds
        self.x_f_lb = [xp_f, yp_f, h_min, self.oew, ts_min]
        self.x_f_ub = [xp_f, yp_f, h_max, self.mass_init, ts_max]

        # States - Lower and upper bounds
        self.x_lb = [x_min, y_min, h_min, self.oew, ts_min]
        self.x_ub = [x_max, y_max, h_max, self.mass_init, ts_max]

        # Control init - lower and upper bounds
        self.u_0_lb = [0.5, -500 * fpm, psi - pi / 4]
        self.u_0_ub = [self.mach_max, 500 * fpm, psi + pi / 4]

        # Control final - lower and upper bounds
        self.u_f_lb = [0.5, -500 * fpm, psi - pi / 4]
        self.u_f_ub = [self.mach_max, 500 * fpm, psi + pi / 4]

        # Control - Lower and upper bound
        self.u_lb = [0.5, -500 * fpm, psi - pi / 2]
        self.u_ub = [self.mach_max, 500 * fpm, psi + pi / 2]

        # Initial guess - states
        self.x_guess = self.initial_guess()

        # Initial guess - controls
        self.u_guess = [0.7, 0, psi]

    def trajectory(self, objective="fuel", **kwargs) -> pd.DataFrame:
        """Compute the optimal cruise trajectory.

        Args:
            objective: Optimization objective. Default "fuel".
            **kwargs:
                max_fuel: Maximum fuel constraint (kg).
                initial_guess: DataFrame to use as initial guess.
                return_failed: Return result even if optimization fails.

        Returns:
            pd.DataFrame: Optimized trajectory.
        """
        self.init_conditions(**kwargs)

        initial_guess = kwargs.get("initial_guess", None)
        if initial_guess is not None:
            self.x_guess = self.initial_guess(initial_guess)

        customized_max_fuel = kwargs.get("max_fuel", None)
        return_failed = kwargs.get("return_failed", False)

        X, U = self._build_opti(
            objective, ts_final_guess=self.range * 1000 / 200, **kwargs
        )
        opti = self._opti

        # --- Phase-specific constraints ---

        # Aircraft performance constraints
        for k in range(self.nodes):
            S = self.aircraft["wing"]["area"]
            mass = X[k][3]
            v = oc.aero.mach2tas(U[k][0], X[k][2], dT=self.dT)
            tas = v / kts
            alt = X[k][2] / ft
            rho = oc.aero.density(X[k][2], dT=self.dT)
            thrust_max = self.thrust.cruise(tas, alt, dT=self.dT)

            # max_thrust * 95% > drag
            opti.subject_to(
                thrust_max * 0.95 >= self.drag.clean(mass, tas, alt, dT=self.dT)
            )

            # max lift * 80% > weight
            cd0 = self.drag.polar["clean"]["cd0"]
            ck = self.drag.polar["clean"]["k"]
            drag_max = thrust_max * 0.9
            cd_max = drag_max / (0.5 * rho * v**2 * S + 1e-10)
            cl_max = ca.sqrt(ca.fmax(1e-10, (cd_max - cd0) / ck))
            L_max = cl_max * 0.5 * rho * v**2 * S
            opti.subject_to(L_max * 0.8 >= mass * oc.aero.g0)

        # ts and dt consistency
        for k in range(self.nodes - 1):
            opti.subject_to(opti.bounded(-1, X[k + 1][4] - X[k][4] - self.dt, 1))

        # Smooth heading change
        for k in range(self.nodes - 1):
            opti.subject_to(
                opti.bounded(-15 * pi / 180, U[k + 1][2] - U[k][2], 15 * pi / 180)
            )

        # Optional constraints
        if self.fix_mach:
            for k in range(self.nodes - 1):
                opti.subject_to(U[k + 1][0] == U[k][0])

        if self.fix_alt:
            for k in range(self.nodes):
                opti.subject_to(U[k][1] == 0)

        if self.fix_track:
            for k in range(self.nodes - 1):
                opti.subject_to(U[k + 1][2] == U[k][2])

        if not self.allow_descent:
            for k in range(self.nodes):
                opti.subject_to(U[k][1] >= 0)

        # Fuel constraint
        opti.subject_to(opti.bounded(0, X[0][3] - X[-1][3], self.fuel_max))

        if customized_max_fuel is not None:
            opti.subject_to(X[0][3] - X[-1][3] <= customized_max_fuel)

        # --- Solve ---
        df = self._solve(X, U, **kwargs)
        df_copy = df.copy()

        if not self.solver.stats()["success"]:
            warnings.warn("flight might be infeasible.")

        if df.altitude.max() < 5000:
            warnings.warn("max altitude < 5000 ft, optimization seems to have failed.")
            df = None

        if df is not None:
            final_mass = df.mass.iloc[-1]
            if final_mass < self.oew:
                warnings.warn("final mass condition violated (smaller than OEW).")
                df = None

        if return_failed:
            return df_copy

        return df
