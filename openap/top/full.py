import warnings
from math import pi

import casadi as ca
import openap.casadi as oc
from openap.aero import fpm, ft, kts

import numpy as np
import openap
import pandas as pd

from .base import Base
from .climb import Climb
from .cruise import Cruise
from .descent import Descent

try:
    from . import tools
except Exception:
    warnings.warn("cfgrib and sklearn are required for wind integration")


class CompleteFlight(Base):
    """
    A class to represent a complete flight trajectory optimization.

    Methods
    -------
    __init__(*args, **kwargs)
        Initializes the CompleteFlight object with given arguments.

    init_conditions()
        Initializes the direct collocation bounds and guesses for the optimization problem.

    trajectory(objective="fuel", return_failed=False, **kwargs) -> pd.DataFrame
        Calculates the complete global optimal trajectory based on the given objective.
        Parameters:
            objective (str): The objective of the optimization, default is "fuel".
            return_failed (bool): If True, returns the failed trajectory if optimization fails.
            **kwargs: Additional keyword arguments for the optimization model.
        Returns:
            pd.DataFrame: The optimized trajectory as a pandas DataFrame.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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
        h_min = 100 * ft

        hdg = oc.geo.bearing(self.lat1, self.lon1, self.lat2, self.lon2)
        psi = hdg * pi / 180

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

    def trajectory(self, objective="fuel", **kwargs) -> pd.DataFrame:
        """
        Computes the optimal trajectory for the aircraft based on the given objective.

        Parameters:
        - objective (str): The objective of the optimization, default is "fuel".
        - **kwargs: Additional keyword arguments.
            - max_fuel (float): Customized maximum fuel constraint.
            - initial_guess (pd.DataFrame): Initial guess for the trajectory.
            - return_failed (bool): If True, returns the DataFrame even if the
                optimization fails. Default is False.

        Returns:
        - pd.DataFrame: A DataFrame containing the optimized trajectory.
        """
        self.init_conditions(**kwargs)

        initial_guess = kwargs.get("initial_guess", None)
        if initial_guess is not None:
            self.x_guess = self.initial_guess(initial_guess)

        customized_max_fuel = kwargs.get("max_fuel", None)
        return_failed = kwargs.get("return_failed", False)

        X, U = self._build_opti(objective, ts_final_guess=7200, **kwargs)
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
                opti.subject_to(opti.bounded(-500 * fpm, U[k][1], 500 * fpm))
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
            opti.subject_to(opti.bounded(-1, X[k + 1][4] - X[k][4] - self.dt, 1))

        # Smooth Mach number change
        for k in range(self.nodes - 1):
            opti.subject_to(opti.bounded(-0.2, U[k + 1][0] - U[k][0], 0.2))

        # Smooth vertical rate change
        for k in range(self.nodes - 1):
            opti.subject_to(opti.bounded(-500 * fpm, U[k + 1][1] - U[k][1], 500 * fpm))

        # Smooth heading change
        for k in range(self.nodes - 1):
            opti.subject_to(
                opti.bounded(-15 * pi / 180, U[k + 1][2] - U[k][2], 15 * pi / 180)
            )

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
            if final_mass > self.mlw:
                warnings.warn("final mass condition violated (larger than MLW).")
                df = None

        if return_failed:
            return df_copy

        return df


class MultiPhase(Base):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cruise = Cruise(*args, **kwargs)
        self.climb = Climb(*args, **kwargs)
        self.descent = Descent(*args, **kwargs)

    def enable_wind(self, windfield: pd.DataFrame):
        w = tools.PolyWind(
            windfield, self.proj, self.lat1, self.lon1, self.lat2, self.lon2
        )
        self.cruise.wind = w
        self.climb.wind = w
        self.descent.wind = w

    def change_engine(self, engtype):
        self.cruise.engtype = engtype
        self.cruise.engine = oc.prop.engine(engtype)
        self.cruise.thrust = oc.Thrust(self.actype, engtype)
        self.cruise.fuelflow = oc.FuelFlow(self.actype, engtype, polydeg=2)
        self.cruise.emission = oc.Emission(self.actype, engtype)

        self.climb.engtype = engtype
        self.climb.engine = oc.prop.engine(engtype)
        self.climb.thrust = oc.Thrust(self.actype, engtype)
        self.climb.fuelflow = oc.FuelFlow(self.actype, engtype, polydeg=2)
        self.climb.emission = oc.Emission(self.actype, engtype)

        self.descent.engtype = engtype
        self.descent.engine = oc.prop.engine(engtype)
        self.descent.thrust = oc.Thrust(self.actype, engtype)
        self.descent.fuelflow = oc.FuelFlow(self.actype, engtype, polydeg=2)
        self.descent.emission = oc.Emission(self.actype, engtype)

    def trajectory(self, objective="fuel", **kwargs) -> pd.DataFrame:
        """
        Calculate the optimal trajectory including climb, cruise, and descent phases.

        Parameters:
        objective (str or tuple): The optimization objective for the trajectory.
            It can be a string or a tuple of strings specifying the objective for
            climb, cruise, and descent respectively. Default is "fuel".
        **kwargs: Additional keyword arguments.

        Returns:
        pd.DataFrame: A DataFrame containing the combined trajectory data.

        The DataFrame includes columns for mass, latitude, longitude, true airspeed,
            and timestamp (ts) among others.

        The method performs the following steps:
        1. Calculate the preliminary optimal cruise trajectory parameters.
        2. Calculate the optimal climb trajectory.
        3. Update the cruise parameters based on the climb results and recalculate the cruise trajectory.
        4. Calculate the optimal descent trajectory.
        5. Determine the top of descent (TOD) point.
        6. Adjust the timestamps for the cruise and descent phases.
        7. Concatenate the climb, cruise, and descent data into a single DataFrame.

        If the `debug` attribute is set to True, debug information will be printed.
        """
        if isinstance(objective, str):
            obj_cl = obj_cr = obj_de = objective
        else:
            obj_cl, obj_cr, obj_de = objective

        if self.debug:
            print("Finding the preliminary optimal cruise trajectory parameters...")

        dfcr = self.cruise.trajectory(obj_cr, **kwargs)

        # climb
        if self.debug:
            print("Finding optimal climb trajectory...")

        dfcl = self.climb.trajectory(obj_cl, dfcr, **kwargs)

        # cruise
        if self.debug:
            print("Finding optimal cruise trajectory...")

        self.cruise.mass_init = dfcl.mass.iloc[-1]
        self.cruise.lat1 = dfcl.latitude.iloc[-1]
        self.cruise.lon1 = dfcl.longitude.iloc[-1]
        dfcr = self.cruise.trajectory(obj_cr, **kwargs)

        # descent
        if self.debug:
            print("Finding optimal descent trajectory...")

        self.descent.mass_init = dfcr.mass.iloc[-1]
        dfde = self.descent.trajectory(obj_de, dfcr, **kwargs)

        # find top of descent
        dbrg = np.array(
            openap.aero.bearing(
                dfde.latitude.iloc[0],
                dfde.longitude.iloc[0],
                dfcr.latitude,
                dfcr.longitude,
            )
        )
        ddbrg = np.abs((dbrg[1:] - dbrg[:-1]).round())
        idx = np.where(ddbrg > 90)[0]
        idx_tod = idx[0] if len(idx) > 0 else -1

        dfcr = dfcr.iloc[: idx_tod + 1]

        # time at top of climb
        dfcr.ts = dfcl.ts.iloc[-1] + dfcr.ts

        # time at top of descent, considering the distant between last point in cruise and tod

        x1, y1 = self.proj(dfcr.longitude.iloc[-1], dfcr.latitude.iloc[-1])
        x2, y2 = self.proj(dfde.longitude.iloc[0], dfde.latitude.iloc[0])

        d = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        v = dfcr.tas.iloc[-1] * kts
        dt = np.round(d / v)
        dfde.ts = dfcr.ts.iloc[-1] + dt + dfde.ts

        df_full = pd.concat([dfcl, dfcr, dfde], ignore_index=True)

        return df_full

    def get_solver_stats(self):
        """Get solver statistics for all phases.

        Returns:
            dict: Solver statistics for climb, cruise, and descent phases.
        """
        return {
            "climb": self.climb.solver.stats(),
            "cruise": self.cruise.solver.stats(),
            "descent": self.descent.solver.stats(),
        }
