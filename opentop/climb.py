from math import pi

import casadi as ca

import numpy as np
import openap.casadi as oc
import pandas as pd
from openap.aero import fpm, ft, kts

from .base import Base
from .cruise import Cruise


class Climb(Base):
    """Climb phase trajectory optimizer."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cruise = Cruise(*args, **kwargs)

    def init_conditions(self, df_cruise, alt_stop=None):
        """Initialize direct collocation bounds and guesses.

        Args:
            df_cruise: Cruise trajectory DataFrame.
            alt_stop: Stop altitude in feet. If provided, used instead of df_cruise.
        """

        # Convert lat/lon to cartisian coordinates.
        xp_0, yp_0 = self.proj(self.lon1, self.lat1)
        xp_f, yp_f = self.proj(self.lon2, self.lat2)
        x_min = min(xp_0, xp_f) - 10_000
        x_max = max(xp_0, xp_f) + 10_000
        y_min = min(yp_0, yp_f) - 10_000
        y_max = max(yp_0, yp_f) + 10_000
        od_bearing = oc.geo.bearing(self.lat1, self.lon1, self.lat2, self.lon2)
        od_psi = od_bearing * pi / 180

        mass_0 = self.mass_init
        mass_oew = self.aircraft["limits"]["OEW"]
        h_min = 100 * ft

        if alt_stop is not None:
            h_toc = alt_stop * ft
            if h_toc > df_cruise.h.iloc[0]:
                print(
                    "The given alt_stop is beyond performance limit, "
                    f"we will use {df_cruise.h.iloc[0] / ft:.0f}"
                )
                h_toc = df_cruise.h.iloc[0]
        else:
            h_toc = df_cruise.h.iloc[0]

        cruise_mach = df_cruise.mach.iloc[0]
        self.traj_range = self.wrap.climb_range()["maximum"] * 1000 * 1.5

        # Initial conditions - Lower and upper bounds
        self.x_0_lb = self.x_0_ub = [xp_0, yp_0, h_min, mass_0, 0]

        # Final conditions - Lower and upper bounds
        self.x_f_lb = [x_min, y_min, h_toc, mass_oew, 0]
        self.x_f_ub = [x_max, y_max, h_toc + 1000, mass_0, 6 * 3600]

        # States - Lower and upper bounds
        self.x_lb = [x_min, y_min, h_min, mass_oew, 0]
        self.x_ub = [x_max, y_max, h_toc, mass_0, 24 * 3600]

        # States - guesses
        xp_guess = xp_0 + np.linspace(
            0, self.traj_range * np.sin(od_psi), self.nodes + 1
        )
        yp_guess = yp_0 + np.linspace(
            0, self.traj_range * np.cos(od_psi), self.nodes + 1
        )
        h_guess = np.linspace(h_min, h_toc, self.nodes + 1)
        m_guess = mass_0 * np.ones(self.nodes + 1)
        ts_guess = np.linspace(0, 6 * 3600, self.nodes + 1)
        self.x_guess = np.vstack([xp_guess, yp_guess, h_guess, m_guess, ts_guess]).T

        # Control init - lower and upper bounds
        self.u_0_lb = [0.1, 0 * fpm, -pi]
        self.u_0_ub = [0.4, 2500 * fpm, 3 * pi]

        # Control final - lower and upper bounds
        self.u_f_lb = [cruise_mach, 0, -pi]
        self.u_f_ub = [cruise_mach, 0, 3 * pi]

        # Control - Lower and upper bound
        self.u_lb = [0.1, 0 * fpm, -pi]
        self.u_ub = [cruise_mach, 2500 * fpm, 3 * pi]

        # Control - guesses
        self.u_guess = [0.2, 1500 * fpm, od_psi]

    def trajectory(
        self,
        objective="fuel",
        df_cruise=None,
        *,
        alt_stop=None,
        remove_cruise=True,
        initial_guess=None,
        interpolant=None,
        n_dim=3,
        time_dependent=False,
        auto_rescale_objective=False,
        exact_hessian=False,
        result_object=False,
    ) -> pd.DataFrame:
        """Compute the optimal climb trajectory.

        Args:
            objective: Optimization objective. Default "fuel".
            df_cruise: Cruise trajectory for target altitude/mach. If None,
                computed automatically.
            alt_stop: Top of climb altitude in feet.
            remove_cruise: Remove level-off points. Default True.
            initial_guess: Unused by Climb (accepted for API symmetry).
            interpolant: CasADi grid-cost interpolant (optional).
            n_dim: Interpolant input dimension (3 or 4). Default 3.
            time_dependent: Grid cost is time-dependent. Default False.
            auto_rescale_objective: Rescale objective to O(1). Default False.
            exact_hessian: Force IPOPT exact Hessian. Default False.
            result_object: If True, return a TrajectoryResult.

        Returns:
            pd.DataFrame (or TrajectoryResult if result_object=True).
        """
        if df_cruise is None:
            if self.debug:
                print("Finding the preliminary optimal cruise parameters...")
            df_cruise = self.cruise.trajectory(objective)

        if self.debug:
            print("Calculating optimal climbing trajectory...")

        self.init_conditions(df_cruise, alt_stop=alt_stop)

        _kwargs = {
            "initial_guess": initial_guess,
            "interpolant": interpolant,
            "n_dim": n_dim,
            "time_dependent": time_dependent,
            "auto_rescale_objective": auto_rescale_objective,
            "exact_hessian": exact_hessian,
        }

        X, U = self._build_opti(objective, ts_final_guess=3600, **_kwargs)
        opti = self._opti

        # --- Phase-specific constraints ---

        # Smooth Mach number changes
        for k in range(1, self.nodes):
            opti.subject_to(opti.bounded(-0.05, U[k][0] - U[k - 1][0], 0.05))

        # Total energy model
        for k in range(self.nodes - 1):
            hk = X[k][2]
            hk1 = X[k + 1][2]
            vk = oc.aero.mach2tas(U[k][0], hk, dT=self.dT)
            vk1 = oc.aero.mach2tas(U[k + 1][0], hk1, dT=self.dT)
            dvdt = (vk1 - vk) / self.dt
            dhdt = (hk1 - hk) / self.dt
            thrust_max = self.thrust.climb(0, hk / ft, 0, dT=self.dT)
            drag = self.drag.clean(X[k][3], vk / kts, hk / ft, dT=self.dT)
            opti.subject_to(
                (thrust_max - drag) / X[k][3] - oc.aero.g0 / vk * dhdt - dvdt >= 0
            )

        # Constrain time and dt
        for k in range(1, self.nodes):
            opti.subject_to(opti.bounded(-1, X[k][4] - X[k - 1][4] - self.dt, 1))

        # Smooth vertical rate changes
        for k in range(1, self.nodes):
            opti.subject_to(opti.bounded(-500 * fpm, U[k][1] - U[k - 1][1], 500 * fpm))

        # Smooth heading changes
        for k in range(1, self.nodes - 1):
            opti.subject_to(
                opti.bounded(-5 * pi / 180, U[k][2] - U[k - 1][2], 5 * pi / 180)
            )

        # Final position should be along the cruise trajectory
        if df_cruise is not None:
            xp_1, yp_1 = df_cruise.x.iloc[0], df_cruise.y.iloc[0]
            xp_2, yp_2 = df_cruise.x.iloc[1], df_cruise.y.iloc[1]
            opti.subject_to(
                (yp_2 - yp_1) * (X[-1][0] - xp_1) == (xp_2 - xp_1) * (X[-1][1] - yp_1)
            )

        # Fixed range
        xp_0, yp_0 = self.proj(self.lon1, self.lat1)
        climb_range = ca.sqrt((X[-1][0] - xp_0) ** 2 + (X[-1][1] - yp_0) ** 2)
        opti.subject_to(climb_range == self.traj_range)

        # --- Solve ---
        df = self._solve(X, U, **_kwargs)

        if remove_cruise:
            df = df.query("vertical_rate > 100")

        if result_object:
            return self._make_result(df)
        return df
