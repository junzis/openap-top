from collections.abc import Iterable

import casadi as ca
import numpy as np
import pandas as pd
import openap
import openap.casadi as oc

from openap.extra.aero import ft, kts, fpm
from math import pi

from .base import Base
from .cruise import Cruise
from .climb import Climb
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

        mach_max = self.aircraft["limits"]["MMO"]
        mass_oew = self.aircraft["limits"]["OEW"]
        h_max = self.aircraft["limits"]["ceiling"]
        h_min = 1500 * ft
        hdg = oc.aero.bearing(self.lat1, self.lon1, self.lat2, self.lon2)
        psi = hdg * pi / 180
        mass_init = self.initial_mass

        # Initial conditions - Lower upper bounds
        self.x_0_lb = [xp_0, yp_0, h_min, mass_init]
        self.x_0_ub = [xp_0, yp_0, h_min, mass_init]

        # Final conditions - Lower and upper bounds
        self.x_f_lb = [xp_f, yp_f, h_min, mass_oew * 0.5]
        self.x_f_ub = [xp_f, yp_f, h_min, mass_init]

        # States - Lower and upper bounds
        self.x_lb = [x_min, y_min, h_min, mass_oew * 0.5]
        self.x_ub = [x_max, y_max, h_max, mass_init]

        # Initial guess - states
        xp_g = np.linspace(xp_0, xp_f, self.nodes + 1)
        yp_g = np.linspace(yp_0, yp_f, self.nodes + 1)
        h_g = h_max * np.ones(self.nodes + 1)
        m_g = mass_init * np.ones(self.nodes + 1)
        self.x_guess = np.vstack([xp_g, yp_g, h_g, m_g]).T

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
                fj, qj = self.f(Xc[j - 1], Uk)
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
            max_descent_range = 500_000
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

        # total energy model
        for k in range(self.nodes - 1):
            hk = X[k][2]
            hk1 = X[k + 1][2]
            vs = U[k][1]
            vk = oc.aero.mach2tas(U[k][0], hk)
            vk1 = oc.aero.mach2tas(U[k + 1][0], hk1)
            dvdt = (vk1 - vk) / self.dt
            dhdt = (hk1 - hk) / self.dt
            thrust_max = self.thrust.climb(0, hk / ft, 0)
            drag = self.drag.clean(X[k][3], vk / kts, hk / ft)
            g.append((thrust_max - drag) / X[k][3] - oc.aero.g0 / vk * dhdt - dvdt)
            lbg.append([0])
            ubg.append([ca.inf])

        # aircraft performance constraints
        for k in range(1, self.nodes):
            # max_thrust > drag
            v = oc.aero.mach2tas(U[k][0], X[k][2])
            tas = v / kts
            alt = X[k][2] / ft
            g.append(self.thrust.cruise(0, alt) - self.drag.clean(X[k][3], tas, alt))
            lbg.append([0])
            ubg.append([ca.inf])

            # max lift > weight
            rho = oc.aero.density(X[k][2])
            S = self.aircraft["wing"]["area"]
            g.append(1.4 * 0.5 * rho * v**2 * S - X[k][3] * oc.aero.g0)
            lbg.append([0])
            ubg.append([ca.inf])

        # final mass larger than OEW
        # g.append(X[-1][3] - self.oew)
        # lbg.append([0])
        # ubg.append([self.mlw])

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
            RuntimeWarning("optimization failed")
            return None
        
        # final timestep
        ts_final = self.solution["x"][-1].full()[0][0]

        # Function to get x and u from w
        output = ca.Function("output", [w], [X, U], ["w"], ["x", "u"])
        x_opt, u_opt = output(self.solution["x"])

        df = self.to_trajectory(ts_final, x_opt, u_opt)

        # check if the optimizer has failed due to too short flight distance
        if df.alt.max() < 5000:
            RuntimeWarning("optimization seems to have failed.")
            return None
        
        if df.mass.iloc[-1] < self.oew or df.mass.iloc[-1] > self.mlw:
            RuntimeWarning("final mass condition violated.")
            return None

        # check final mass, which should be larger than OEW, and smaller than MLW
        if df is not None:
            final_mass = df.mass.iloc[-1]
            if final_mass < self.oew or final_mass > self.mlw:
                RuntimeWarning(
                    "optimization failed, final mass smaller than OEW or larger than MLW."
                )

                if not return_failed:
                    df = None

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
        self.cruise.lat1 = dfcl.lat.iloc[-1]
        self.cruise.lon1 = dfcl.lon.iloc[-1]
        dfcr = self.cruise.trajectory(obj_cr, **kwargs)

        # descent
        if self.debug:
            print("Finding optimal descent trajectory...")

        self.descent.initial_mass = dfcr.mass.iloc[-1]
        dfde = self.descent.trajectory(obj_de, dfcr, **kwargs)

        # find top of descent
        dbrg = np.array(
            openap.aero.bearing(dfde.lat.iloc[0], dfde.lon.iloc[0], dfcr.lat, dfcr.lon)
        )
        ddbrg = np.abs((dbrg[1:] - dbrg[:-1]).round())
        idx = np.where(ddbrg > 90)[0]
        idx_tod = idx[0] if len(idx) > 0 else -1

        dfcr = dfcr.iloc[: idx_tod + 1]

        # time at top of climb
        dfcr.ts = dfcl.ts.iloc[-1] + dfcr.ts

        # time at top of descent, considering the distant between last point in cruise and tod

        x1, y1 = self.proj(dfcr.lon.iloc[-1], dfcr.lat.iloc[-1])
        x2, y2 = self.proj(dfde.lon.iloc[0], dfde.lat.iloc[0])

        d = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        v = dfcr.tas.iloc[-1] * kts
        dt = np.round(d / v)
        dfde.ts = dfcr.ts.iloc[-1] + dt + dfde.ts

        df_full = pd.concat([dfcl, dfcr, dfde], ignore_index=True)

        return df_full
