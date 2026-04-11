from math import pi

import casadi as ca
import openap.casadi as oc
from openap.aero import fpm, ft, kts

import numpy as np
import pandas as pd

from .base import Base
from .cruise import Cruise


class Descent(Base):
    """Descent phase trajectory optimizer."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cruise = Cruise(*args, **kwargs)

    def init_conditions(self, df_cruise, alt_start=None):
        """Initialize direct collocation bounds and guesses.

        Args:
            df_cruise: Cruise trajectory DataFrame.
            alt_start: Start altitude in feet. If provided, used instead of df_cruise.
        """

        h_min = 100 * ft
        od_bearing = oc.geo.bearing(self.lat1, self.lon1, self.lat2, self.lon2)
        od_psi = od_bearing * pi / 180

        xp_0, yp_0 = self.proj(self.lon1, self.lat1)
        xp_f, yp_f = self.proj(self.lon2, self.lat2)

        x_min = min(xp_0, xp_f) - 10_000
        x_max = max(xp_0, xp_f) + 10_000
        y_min = min(yp_0, yp_f) - 10_000
        y_max = max(yp_0, yp_f) + 10_000
        ts_min = 0
        ts_max = 6 * 3600

        mass_oew = self.oew
        mass_tod = df_cruise.mass.iloc[0]
        cruise_mach = df_cruise.mach.max()

        if alt_start is not None:
            h_start = alt_start * ft
            if h_start > df_cruise.h.iloc[0]:
                print(
                    "The given alt_start is beyond performance limit, "
                    f"we will use {df_cruise.h.iloc[0] / ft}"
                )
                h_start = df_cruise.h.iloc[0]
        else:
            h_start = df_cruise.h.iloc[0]

        # Initial conditions - Lower and upper bounds
        self.x_0_lb = [xp_0 - 1000, yp_0 - 1000, h_start - 100, mass_tod, ts_min]
        self.x_0_ub = [xp_0 + 1000, yp_0 + 1000, h_start + 100, mass_tod, ts_min]

        # Final conditions - Lower and upper bounds
        self.x_f_lb = [xp_f, yp_f, h_min, mass_oew, ts_min]
        self.x_f_ub = [xp_f, yp_f, h_min, mass_tod, ts_max]

        # States - Lower and upper bounds
        self.x_lb = [x_min, y_min, h_min, mass_oew, ts_min]
        self.x_ub = [x_max, y_max, h_start + 100, mass_tod, ts_max]

        # States - guesses
        # dist = h_tod / np.tan(np.radians(3))  # 3 deg
        # xp_guess = xp_f - np.linspace(dist * np.sin(od_psi), 0, self.nodes + 1)
        # yp_guess = yp_f - np.linspace(dist * np.cos(od_psi), 0, self.nodes + 1)
        xp_guess = np.linspace(xp_0, xp_f, self.nodes + 1)
        yp_guess = np.linspace(yp_0, yp_f, self.nodes + 1)
        h_guess = np.linspace(h_start, h_min, self.nodes + 1)
        m_guess = mass_tod * np.ones(self.nodes + 1)
        ts_guess = np.linspace(0, 6 * 3600, self.nodes + 1)
        self.x_guess = np.vstack([xp_guess, yp_guess, h_guess, m_guess, ts_guess]).T

        # Control init - lower and upper bounds
        self.u_0_lb = [cruise_mach - 0.1, -2500 * fpm, od_psi - pi / 4]
        self.u_0_ub = [cruise_mach + 0.1, 0 * fpm, od_psi + pi / 4]

        # Control final - lower and upper bounds
        self.u_f_lb = [0.1, -2500 * fpm, od_psi - pi / 4]
        self.u_f_ub = [0.3, 0 * fpm, od_psi + pi / 4]

        # Control - Lower and upper bound
        self.u_lb = [0.1, -2500 * fpm, od_psi - pi / 2]
        self.u_ub = [cruise_mach, 0 * fpm, od_psi + pi / 2]

        # Control - guesses
        self.u_guess = [0.7, 0 * fpm, od_psi]

    def trajectory(self, objective="fuel", df_cruise=None, **kwargs) -> pd.DataFrame:
        """Compute the optimal descent trajectory.

        Args:
            objective: Optimization objective. Default "fuel".
            df_cruise: Cruise trajectory for initial altitude/mach.
                If None, computed automatically.
            **kwargs:
                alt_start: Start of descent altitude in feet.
                remove_cruise: Remove level-off points. Default True.

        Returns:
            pd.DataFrame: Optimized descent trajectory.
        """
        alt_start = kwargs.get("alt_start", None)

        if df_cruise is None:
            if self.debug:
                print("Finding the preliminary optimal cruise parameters...")
            df_cruise = self.cruise.trajectory(objective)

        if self.debug:
            print("Calculating optimal descent trajectory...")

        self.init_conditions(df_cruise, alt_start=alt_start)

        X, U = self._build_opti(objective, ts_final_guess=3600, **kwargs)
        opti = self._opti

        # --- Phase-specific constraints ---

        # Constrain time and dt
        for k in range(1, self.nodes):
            opti.subject_to(opti.bounded(-1, X[k][4] - X[k - 1][4] - self.dt, 1))

        # Smooth Mach number changes
        for k in range(1, self.nodes):
            opti.subject_to(opti.bounded(-0.1, U[k][0] - U[k - 1][0], 0.1))

        # Smooth vertical rate changes
        for k in range(1, self.nodes):
            opti.subject_to(
                opti.bounded(-1000 * fpm, U[k][1] - U[k - 1][1], 1000 * fpm)
            )

        # Smooth heading changes
        for k in range(1, self.nodes):
            opti.subject_to(
                opti.bounded(-5 * pi / 180, U[k][2] - U[k - 1][2], 5 * pi / 180)
            )

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

        # --- Solve ---
        df = self._solve(X, U, **kwargs)

        remove_cruise = kwargs.get("remove_cruise", True)
        if remove_cruise:
            df = df.query("vertical_rate < -100")

        return df
