import warnings
from collections.abc import Iterable
from math import pi

import casadi as ca
import numpy as np
import openap
import openap.casadi as oc
import pandas as pd
from openap.extra.aero import fpm, ft, kts

from .base import Base
from .climb import Climb
from .cruise import Cruise
from .descent import Descent

try:
    from . import tools
except:
    RuntimeWarning("cfgrib and sklearn are required for wind integration")


class CompleteFlight(Base):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def init_conditions(self):
        """Initialize direct collocation bounds and guesses."""

        # Convert lat/lon to cartisian coordinates.
        xp_0, yp_0 = self.proj(self.lon1, self.lat1)
        xp_f, yp_f = self.proj(self.lon2, self.lat2)
        x_min = min(xp_0, xp_f) - 10_000
        x_max = max(xp_0, xp_f) + 10_000
        y_min = min(yp_0, yp_f) - 10_000
        y_max = max(yp_0, yp_f) + 10_000
        ts_min = 0
        ts_max = 6 * 3600

        mach_max = self.aircraft["limits"]["MMO"]
        mass_oew = self.aircraft["limits"]["OEW"]
        h_max = self.aircraft["limits"]["ceiling"]
        h_min = 1500 * ft
        hdg = oc.aero.bearing(self.lat1, self.lon1, self.lat2, self.lon2)
        psi = hdg * pi / 180
        mass_init = self.initial_mass

        # Initial conditions - Lower upper bounds
        self.x_0_lb = [xp_0, yp_0, h_min, mass_init, ts_min]
        self.x_0_ub = [xp_0, yp_0, h_min, mass_init, ts_min]

        # Final conditions - Lower and upper bounds
        self.x_f_lb = [xp_f, yp_f, h_min, mass_oew * 0.5, ts_min]
        self.x_f_ub = [xp_f, yp_f, h_min, mass_init, ts_max]

        # States - Lower and upper bounds
        self.x_lb = [x_min, y_min, h_min, mass_oew * 0.5, ts_min]
        self.x_ub = [x_max, y_max, h_max, mass_init, ts_max]

        # Initial guess - states
        xp_guess = np.linspace(xp_0, xp_f, self.nodes + 1)
        yp_guess = np.linspace(yp_0, yp_f, self.nodes + 1)
        h_guess = h_max * np.ones(self.nodes + 1)
        m_guess = mass_init * np.ones(self.nodes + 1)
        ts_guess = np.linspace(0, 6 * 3600, self.nodes + 1)
        self.x_guess = np.vstack([xp_guess, yp_guess, h_guess, m_guess, ts_guess]).T

        # Control init - lower and upper bounds
        self.u_0_lb = [0.1, 500 * fpm, psi]
        self.u_0_ub = [0.3, 2500 * fpm, psi]

        # Control final - lower and upper bounds
        self.u_f_lb = [0.1, -1500 * fpm, psi]
        self.u_f_ub = [0.3, -300 * fpm, psi]

        # Control - Lower and upper bound
        self.u_lb = [0.1, -2500 * fpm, -pi]
        self.u_ub = [mach_max, 2500 * fpm, 3 * pi]

        # Control - guesses
        self.u_guess = [0.6, 1000 * fpm, psi]

    def trajectory(
        self, objective="fuel", return_failed=False, **kwargs
    ) -> pd.DataFrame:
        if self.debug:
            print("Calculating complete global optimal trajectory...")
            ipopt_print = 5
            print_time = 1
        else:
            ipopt_print = 0
            print_time = 0

        self.init_conditions()
        self.init_model(objective, **kwargs)

        C, D, B = self.collocation_coeff()

        # Start with an empty NLP
        w = []  # Containing all the states & controls generated
        w0 = []  # Containing the initial guess for w
        lbw = []  # Lower bound constraints on the w variable
        ubw = []  # Upper bound constraints on the w variable
        J = 0  # Objective function
        g = []  # Constraint function
        lbg = []  # Constraint lb value
        ubg = []  # Constraint ub value

        # For plotting x and u given w
        X = []
        U = []

        # Apply initial conditions
        # Create Xk such that it is the same length as x
        nstates = self.x.shape[0]
        Xk = ca.MX.sym("X0", nstates, self.x.shape[1])
        w.append(Xk)
        lbw.append(self.x_0_lb)
        ubw.append(self.x_0_ub)
        w0.append(self.x_guess[0])
        X.append(Xk)

        # Formulate the NLP
        for k in range(self.nodes):
            # New NLP variable for the control
            Uk = ca.MX.sym("U_" + str(k), self.u.shape[0])
            U.append(Uk)
            w.append(Uk)
            if k == 0:
                lbw.append(self.u_0_lb)
                ubw.append(self.u_0_ub)
                w0.append(self.u_guess)
            elif k == self.nodes - 1:
                lbw.append(self.u_f_lb)
                ubw.append(self.u_f_ub)
                w0.append(self.u_guess)
            else:
                lbw.append(self.u_lb)
                ubw.append(self.u_ub)
                w0.append(self.u_guess)

            # State at collocation points
            Xc = []
            for j in range(self.polydeg):
                Xkj = ca.MX.sym("X_" + str(k) + "_" + str(j), nstates)
                Xc.append(Xkj)
                w.append(Xkj)
                lbw.append(self.x_lb)
                ubw.append(self.x_ub)
                w0.append(self.x_guess[k])

            # Loop over collocation points
            Xk_end = D[0] * Xk
            for j in range(1, self.polydeg + 1):
                # Expression for the state derivative at the collocation point
                xpc = C[0, j] * Xk
                for r in range(self.polydeg):
                    xpc = xpc + C[r + 1, j] * Xc[r]

                # Append collocation equations
                fj, qj = self.func_dynamics(Xc[j - 1], Uk)
                g.append(self.dt * fj - xpc)
                lbg.append([0] * nstates)
                ubg.append([0] * nstates)

                # Add contribution to the end state
                Xk_end = Xk_end + D[j] * Xc[j - 1]

                # Add contribution to quadrature function
                # J = J + B[j] * qj * dt
                J = J + B[j] * qj

            # New NLP variable for state at end of interval
            Xk = ca.MX.sym("X_" + str(k + 1), nstates)
            w.append(Xk)
            X.append(Xk)

            if k < self.nodes - 1:
                lbw.append(self.x_lb)
                ubw.append(self.x_ub)
            else:
                # Final conditions
                lbw.append(self.x_f_lb)
                ubw.append(self.x_f_ub)

            w0.append(self.x_guess[k])

            # Add equality constraint
            g.append(Xk_end - Xk)
            lbg.append([0] * nstates)
            ubg.append([0] * nstates)

        w.append(self.ts_final)
        lbw.append([0])
        ubw.append([ca.inf])
        w0.append([7200])

        # constrain altitude during cruise for long cruise flights
        if self.range > 1500_000:
            dd = self.range / (self.nodes + 1)
            max_climb_range = 500_000
            max_descent_range = 300_000
            idx_toc = int(max_climb_range / dd)
            idx_tod = int((self.range - max_descent_range) / dd)

            for k in range(idx_toc, idx_tod):
                # minimum avoid large changes in altitude
                g.append(U[k][1])
                lbg.append([-500 * fpm])
                ubg.append([500 * fpm])

                # minimum cruise alt FL150
                g.append(X[k][2])
                lbg.append([15000 * ft])
                ubg.append([ca.inf])

            for k in range(0, idx_toc):
                g.append(U[k][1])
                lbg.append([0])
                ubg.append([ca.inf])

            for k in range(idx_tod, self.nodes):
                g.append(U[k][1])
                lbg.append([-ca.inf])
                ubg.append([0])

        # aircraft performance constraints
        for k in range(1, self.nodes):

            S = self.aircraft["wing"]["area"]
            mass = X[k][3]
            v = oc.aero.mach2tas(U[k][0], X[k][2])
            tas = v / kts
            alt = X[k][2] / ft
            rho = oc.aero.density(X[k][2])
            thrust_max = self.thrust.cruise(tas, alt)

            # max_thrust * 95% > drag (5% margin)
            g.append(thrust_max * 0.95 - self.drag.clean(mass, tas, alt))
            lbg.append([0])
            ubg.append([ca.inf])

            # max lift * 80% > weight (20% margin)
            drag_max = thrust_max * 0.9
            cd_max = drag_max / (0.5 * rho * v**2 * S)
            cd0 = self.drag.polar["clean"]["cd0"]
            ck = self.drag.polar["clean"]["k"]
            cl_max = ca.sqrt((cd_max - cd0) / ck)
            L_max = cl_max * 0.5 * rho * v**2 * S
            g.append(L_max * 0.8 - mass * oc.aero.g0)
            lbg.append([0])
            ubg.append([ca.inf])

        # ts and dt should be consistent
        for k in range(1, self.nodes):
            g.append(X[k][4] - X[k - 1][4] - self.dt)
            lbg.append([-1])
            ubg.append([1])

        # smooth Mach number change
        for k in range(1, self.nodes):
            g.append(U[k][0] - U[k - 1][0])
            lbg.append([-0.2])
            ubg.append([0.2])  # to be tunned

        # smooth vertical rate change
        for k in range(1, self.nodes):
            g.append(U[k][1] - U[k - 1][1])
            lbg.append([-500 * fpm])
            ubg.append([500 * fpm])  # to be tunned

        # smooth heading change
        for k in range(1, self.nodes):
            g.append(U[k][2] - U[k - 1][2])
            lbg.append([-15 * pi / 180])
            ubg.append([15 * pi / 180])

        # Concatenate vectors
        w = ca.vertcat(*w)
        g = ca.vertcat(*g)
        X = ca.horzcat(*X)
        U = ca.horzcat(*U)
        w0 = np.concatenate(w0)
        lbw = np.concatenate(lbw)
        ubw = np.concatenate(ubw)
        lbg = np.concatenate(lbg)
        ubg = np.concatenate(ubg)

        # Create an NLP solver
        nlp = {"f": J, "x": w, "g": g}

        opts = {
            "print_time": print_time,
            "ipopt.print_level": ipopt_print,
            "ipopt.sb": "yes",
            "ipopt.max_iter": self.ipopt_max_iter,
        }
        self.solver = ca.nlpsol("solver", "ipopt", nlp, opts)
        self.solution = self.solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)

        if not self.solver.stats()["success"]:
            warnings.warn("optimization failed")
            return None

        # final timestep
        ts_final = self.solution["x"][-1].full()[0][0]

        # Function to get x and u from w
        output = ca.Function("output", [w], [X, U], ["w"], ["x", "u"])
        x_opt, u_opt = output(self.solution["x"])

        df = self.to_trajectory(ts_final, x_opt, u_opt)
        df_copy = df.copy()

        # check if the optimizer has failed due to too short flight distance
        if df.altitude.max() < 5000:
            warnings.warn("optimization seems to have failed.")
            return None

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

        self.cruise.initial_mass = dfcl.mass.iloc[-1]
        self.cruise.lat1 = dfcl.latitude.iloc[-1]
        self.cruise.lon1 = dfcl.longitude.iloc[-1]
        dfcr = self.cruise.trajectory(obj_cr, **kwargs)

        # descent
        if self.debug:
            print("Finding optimal descent trajectory...")

        self.descent.initial_mass = dfcr.mass.iloc[-1]
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
