from math import pi

import casadi as ca

import numpy as np
import openap.casadi as oc
import pandas as pd
from openap.extra.aero import fpm, ft, kts

from .base import Base
from .cruise import Cruise


class Climb(Base):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cruise = Cruise(*args, **kwargs)

    def init_conditions(self, **kwargs):
        """Initialize direct collocation bounds and guesses."""

        # Convert lat/lon to cartisian coordinates.
        xp_0, yp_0 = self.proj(self.lon1, self.lat1)
        xp_f, yp_f = self.proj(self.lon2, self.lat2)
        x_min = min(xp_0, xp_f) - 10_000
        x_max = max(xp_0, xp_f) + 10_000
        y_min = min(yp_0, yp_f) - 10_000
        y_max = max(yp_0, yp_f) + 10_000

        h_end = kwargs.get("h_end", self.aircraft["cruise"]["height"])
        mach_end = kwargs.get("mach_end", self.aircraft["cruise"]["mach"])
        trk_end = kwargs.get(
            "trk_end", oc.aero.bearing(self.lat1, self.lon1, self.lat2, self.lon2)
        )

        self.range = np.sqrt((xp_0 - xp_f) ** 2 + (yp_0 - yp_f) ** 2)
        h_end_min = self.range * np.sin(3 * pi / 180)
        h_end_max = self.range * np.sin(10 * pi / 180)

        psi_end = trk_end * pi / 180

        mass_0 = self.mass_init
        mass_oew = self.aircraft["oew"]
        mach_max = self.aircraft["mmo"]
        h_min = 100 * ft
        h_max = self.aircraft["ceiling"]

        # Initial conditions - Lower and upper bounds
        self.x_0_lb = self.x_0_ub = [xp_0, yp_0, h_min, mass_0, 0]

        # Final conditions - Lower and upper bounds
        self.x_f_lb = [xp_f, yp_f, h_end_min, mass_oew, 0]
        self.x_f_ub = [xp_f, yp_f, h_end_max, mass_0, 3600]

        # States - Lower and upper bounds
        self.x_lb = [x_min, y_min, h_min, mass_oew, 0]
        self.x_ub = [x_max, y_max, h_max, mass_0, 3600]

        # States - guesses
        xp_guess = xp_0 + np.linspace(0, self.range * np.sin(psi_end), self.nodes + 1)
        yp_guess = yp_0 + np.linspace(0, self.range * np.cos(psi_end), self.nodes + 1)
        h_guess = np.linspace(h_min, h_end, self.nodes + 1)
        m_guess = mass_0 * np.ones(self.nodes + 1)
        ts_guess = np.linspace(0, 3600, self.nodes + 1)
        self.x_guess = np.vstack([xp_guess, yp_guess, h_guess, m_guess, ts_guess]).T

        # Control init - lower and upper bounds
        self.u_0_lb = [0.2, 0 * fpm, psi_end - pi / 2]
        self.u_0_ub = [0.3, 2500 * fpm, psi_end + pi / 2]

        # Control final - lower and upper bounds
        self.u_f_lb = [0.3, 0, psi_end - pi / 2]
        self.u_f_ub = [mach_max, 2500 * fpm, psi_end + pi / 2]

        # Control - Lower and upper bound
        self.u_lb = [0.2, 0 * fpm, psi_end - pi / 2]
        self.u_ub = [mach_max, 2500 * fpm, psi_end + pi / 2]

        # Control - guesses
        self.u_guess = [0.25, 2000 * fpm, psi_end]

    def trajectory(self, objective="fuel", df_cruise=None, **kwargs) -> pd.DataFrame:
        df_cruise = kwargs.get("df_cruise", None)
        h_end = kwargs.get("h_end", None)
        mach_end = kwargs.get("mach_end", None)
        trk_end = kwargs.get("trk_end", None)

        customized_max_fuel = kwargs.get("max_fuel", None)

        # if df_cruise is None and h_end is None and mach_end is None:
        #     if self.debug:
        #         print("Finding the preliminary optimal cruise trajectory parameters...")
        #     df_cruise = self.cruise.trajectory(objective)

        # if df_cruise is not None:
        #     h_end = df_cruise.h.iloc[0]
        #     mach_end = df_cruise.mach.iloc[0]
        #     trk_end = df_cruise.heading.iloc[0]

        # assert h_end is not None and mach_end is not None

        self.init_conditions(**kwargs)

        if self.debug:
            print("Calculating optimal climbing trajectory...")

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

        # constrain time and dt
        for k in range(1, self.nodes):
            g.append(X[k][4] - X[k - 1][4] - self.dt)
            lbg.append([-1])
            ubg.append([1])

        # smooth Mach number changes
        for k in range(1, self.nodes):
            g.append(U[k][0] - U[k - 1][0])
            lbg.append([-0.1])
            ubg.append([0.1])

        # smooth vertical rate changes
        for k in range(1, self.nodes):
            g.append(U[k][1] - U[k - 1][1])
            lbg.append([-500 * fpm])
            ubg.append([500 * fpm])

        # smooth heading changes
        for k in range(1, self.nodes):
            g.append(U[k][2] - U[k - 1][2])
            lbg.append([-5 * pi / 180])
            ubg.append([5 * pi / 180])

        # final position should be along the cruise trajectory
        if df_cruise is not None:
            xp_1, yp_1 = df_cruise.x.iloc[0], df_cruise.y.iloc[0]
            xp_2, yp_2 = df_cruise.x.iloc[1], df_cruise.y.iloc[1]
            g.append(
                (yp_2 - yp_1) / (xp_2 - xp_1) - (X[-1][1] - yp_1) / (X[-1][0] - xp_1)
            )
            lbg.append([0])
            ubg.append([0])

        # # fixed range
        # xp_0, yp_0 = self.proj(self.lon1, self.lat1)
        # g.append(ca.sqrt((X[-1][0] - xp_0) ** 2 + (X[-1][1] - yp_0) ** 2))
        # lbg.append([self.traj_range])
        # ubg.append([self.traj_range])

        if customized_max_fuel is not None:
            g.append(X[0][3] - X[-1][3] - customized_max_fuel)
            lbg.append([-ca.inf])
            ubg.append([0])

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
