from math import pi

import casadi as ca

import numpy as np
import openap
import openap.casadi as oc
import pandas as pd
from openap.extra.aero import fpm, ft, kts

from .base import Base
from .cruise import Cruise


class Descent(Base):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.curise = Cruise(*args, **kwargs)

    def init_conditions(self, df_cruise):
        """Initialize direct collocation bounds and guesses."""

        h_min = 100 * ft
        h_tod = df_cruise.h.iloc[-1]
        psi_tod = df_cruise.heading.iloc[-1] * pi / 180
        od_bearing = oc.aero.bearing(self.lat1, self.lon1, self.lat2, self.lon2)
        od_psi = od_bearing * pi / 180

        xp_0, yp_0 = self.proj(self.lon1, self.lat1)
        xp_f, yp_f = self.proj(self.lon2, self.lat2)

        x_min = min(xp_0, xp_f) - 10_000
        x_max = max(xp_0, xp_f) + 10_000
        y_min = min(yp_0, yp_f) - 10_000
        y_max = max(yp_0, yp_f) + 10_000
        ts_min = 0
        ts_max = 6 * 3600

        mass_tod = df_cruise.mass.iloc[-1]
        mass_oew = self.oew
        cruise_mach = df_cruise.mach.iloc[-1]

        # Initial conditions - Lower and upper bounds
        self.x_0_lb = [x_min, y_min, h_tod, mass_tod, ts_min]
        self.x_0_ub = [x_max, y_max, h_tod, mass_tod, ts_min]

        # Final conditions - Lower and upper bounds
        self.x_f_lb = [xp_f, yp_f, h_min, mass_oew, ts_min]
        self.x_f_ub = [xp_f, yp_f, h_min, mass_tod, ts_max]

        # States - Lower and upper bounds
        self.x_lb = [x_min, y_min, h_min, mass_oew, ts_min]
        self.x_ub = [x_max, y_max, h_tod, mass_tod, ts_max]

        # States - guesses
        dist = h_tod / np.tan(np.radians(3))  # 3 deg
        xp_guess = xp_f - np.linspace(dist * np.sin(od_psi), 0, self.nodes + 1)
        yp_guess = yp_f - np.linspace(dist * np.cos(od_psi), 0, self.nodes + 1)
        h_guess = np.linspace(h_tod, h_min, self.nodes + 1)
        m_guess = mass_tod * np.ones(self.nodes + 1)
        ts_guess = np.linspace(0, 6 * 3600, self.nodes + 1)
        self.x_guess = np.vstack([xp_guess, yp_guess, h_guess, m_guess, ts_guess]).T

        # Control init - lower and upper bounds
        self.u_0_lb = [cruise_mach, -2000 * fpm, psi_tod]
        self.u_0_ub = [cruise_mach, 0, psi_tod]

        # Control final - lower and upper bounds
        self.u_f_lb = [0.1, -1000 * fpm, od_psi]
        self.u_f_ub = [0.3, 0 * fpm, od_psi]

        # Control - Lower and upper bound
        self.u_lb = [0.1, -2000 * fpm, -pi]
        self.u_ub = [cruise_mach, 0, 3 * pi]

        # Control - guesses
        self.u_guess = [0.7, -1500 * fpm, psi_tod]

    def trajectory(self, objective="fuel", df_cruise=None, **kwargs) -> pd.DataFrame:
        if df_cruise is None:
            if self.debug:
                print("Finding the preliminary optimal cruise trajectory parameters...")
            df_cruise = self.curise.trajectory(objective)

        if self.debug:
            print("Calculating optimal descent trajectory...")

        self.init_conditions(df_cruise)
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
                # normal boundary conditions
                lbw.append(self.x_lb)
                ubw.append(self.x_ub)
            else:
                # Final bounday conditions
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
        w0.append([3600])

        # constrain time and dt
        for k in range(1, self.nodes):
            g.append(X[k][4] - X[k - 1][4] - self.dt)
            lbg.append([-1])
            ubg.append([1])

        # smooth Mach number changes
        for k in range(1, self.nodes):
            g.append(U[k][0] - U[k - 1][0])
            lbg.append([-0.1])
            ubg.append([0.1])  # to be tunned

        # smooth vertical rate changes
        for k in range(1, self.nodes):
            g.append(U[k][1] - U[k - 1][1])
            lbg.append([-1000 * fpm])
            ubg.append([1000 * fpm])  # to be tunned

        # smooth heading changes
        for k in range(1, self.nodes):
            g.append(U[k][2] - U[k - 1][2])
            lbg.append([-5 * pi / 180])
            ubg.append([5 * pi / 180])

        # # flight path angle
        # for k in range(0, self.nodes):
        #     v = oc.aero.mach2tas(U[k][0], X[k][2],dT=self.dT)
        #     vs = U[k][1]
        #     g.append(vs / v)
        #     lbg.append([np.tan(np.radians(-4))])
        #     ubg.append([np.tan(np.radians(-2))])

        # first position should be along the cruise trajectory
        xp_1, yp_1 = self.proj(
            df_cruise.longitude.iloc[-1], df_cruise.latitude.iloc[-1]
        )
        xp_2, yp_2 = self.proj(
            df_cruise.longitude.iloc[-2], df_cruise.latitude.iloc[-2]
        )

        # xp_1, yp_1 = df_cruise.x.iloc[-1], df_cruise.y.iloc[-1]
        # xp_2, yp_2 = df_cruise.x.iloc[-2], df_cruise.y.iloc[-2]
        g.append((yp_1 - yp_2) / (xp_1 - xp_2) - (yp_1 - X[0][1]) / (xp_1 - X[0][0]))
        lbg.append([0])
        ubg.append([0])

        # force equilibrium
        for k in range(self.nodes - 1):
            vs = U[k][1]
            h = X[k][2]
            v = oc.aero.mach2tas(U[k][0], h, dT=self.dT)
            gamma = ca.arctan2(vs, v)
            thrust_idle = self.thrust.descent_idle(v / kts, h / ft, dT=self.dT)
            drag = self.drag.clean(X[k][3], v / kts, h / ft, dT=self.dT)
            g.append(thrust_idle - X[k][3] * 9.8 * ca.sin(gamma) - drag)
            lbg.append([-ca.inf])
            ubg.append([0])

        # total energy model
        for k in range(self.nodes - 1):
            hk = X[k][2]
            hk1 = X[k + 1][2]
            vs = U[k][1]
            vk = oc.aero.mach2tas(U[k][0], hk, dT=self.dT)
            vk1 = oc.aero.mach2tas(U[k + 1][0], hk1, dT=self.dT)
            pa = ca.arctan2(vs, vk) * 180 / pi
            dvdt = (vk1 - vk) / self.dt
            dhdt = (hk1 - hk) / self.dt
            thrust_idle = self.thrust.descent_idle(vk / kts, hk / ft, dT=self.dT)
            drag = self.drag.clean(X[k][3], vk / kts, hk / ft, pa, dT=self.dT)
            g.append((thrust_idle - drag) / X[k][3] - oc.aero.g0 / vk * dhdt - dvdt)
            lbg.append([0])
            ubg.append([ca.inf])

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

        self.solver = ca.nlpsol("solver", "ipopt", nlp, self.solver_options)

        self.solution = self.solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)

        # final timestep
        ts_final = self.solution["x"][-1].full()[0][0]

        # Function to get x and u from w
        output = ca.Function("output", [w], [X, U], ["w"], ["x", "u"])
        x_opt, u_opt = output(self.solution["x"])

        df = self.to_trajectory(ts_final, x_opt, u_opt)

        return df
