import casadi as ca
import numpy as np
import pandas as pd
import openap
import openap.casadi as oc

from openap.extra.aero import ft, kts, fpm
from math import pi

from .base import Base


class Cruise(Base):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.fix_mach = False
        self.fix_alt = False
        self.fix_track = False
        self.allow_descent = False

    def fix_mach_number(self):
        self.fix_mach = True

    def fix_cruise_altitude(self):
        self.fix_alt = True

    def fix_track_angle(self):
        self.fix_track = True

    def allow_cruise_descent(self):
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

        mach_max = self.aircraft["limits"]["MMO"]
        mass_toc = self.initial_mass
        mass_oew = self.aircraft["limits"]["OEW"]
        h_max = kwargs.get("h_max", self.aircraft["limits"]["ceiling"])
        h_min = kwargs.get("h_min", 15_000 * ft)

        # Initial conditions - Lower upper bounds
        self.x_0_lb = [xp_0, yp_0, h_min, mass_toc]
        self.x_0_ub = [xp_0, yp_0, h_max, mass_toc]

        # Final conditions - Lower and upper bounds
        self.x_f_lb = [xp_f, yp_f, h_min, mass_oew]
        self.x_f_ub = [xp_f, yp_f, h_max, mass_toc]

        # States - Lower and upper bounds
        self.x_lb = [x_min, y_min, h_min, mass_oew]
        self.x_ub = [x_max, y_max, h_max, mass_toc]

        # Control - Lower and upper bound
        self.u_lb = [0.5, -500 * fpm, -pi]
        self.u_ub = [mach_max, 500 * fpm, 3 * pi]

        # Initial guess - states
        xp_g = np.linspace(xp_0, xp_f, self.nodes + 1)
        yp_g = np.linspace(yp_0, yp_f, self.nodes + 1)
        h_g = h_max * np.ones(self.nodes + 1)
        m_g = mass_toc * np.ones(self.nodes + 1)
        self.x_guess = np.vstack([xp_g, yp_g, h_g, m_g]).T

        # Initial guess - controls
        hdg = oc.aero.bearing(self.lat1, self.lon1, self.lat2, self.lon2)
        self.u_guess = [0.7, 0, hdg * pi / 180]

    def trajectory(self, objective="fuel", **kwargs) -> pd.DataFrame:
        if self.debug:
            print("Calculating optimal cruise trajectory...")
            ipopt_print = 5
            print_time = 1
        else:
            ipopt_print = 0
            print_time = 0

        self.init_conditions(**kwargs)
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
            w.append(Uk)
            lbw.append(self.u_lb)
            ubw.append(self.u_ub)
            w0.append(self.u_guess)
            U.append(Uk)

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

            # lbw.append(x_lb)
            # ubw.append(x_ub)

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
        w0.append([self.range * 1000 / 200])

        # aircraft performane constraints
        for k in range(1, self.nodes):
            # max_thrust > drag
            v = oc.aero.mach2tas(U[k][0], X[k][2])
            tas = v / kts
            alt = X[k][2] / ft

            # max thrut at given tas, important
            Tmax = self.thrust.cruise(tas, alt)

            g.append(Tmax - self.drag.clean(X[k][3], tas, alt))
            lbg.append([0])
            ubg.append([ca.inf])

            # max lift > weight
            rho = oc.aero.density(X[k][2])
            S = self.aircraft["wing"]["area"]
            g.append(1.4 * 0.5 * rho * v**2 * S - X[k][3] * oc.aero.g0)
            lbg.append([0])
            ubg.append([ca.inf])

        # smooth Mach number change
        for k in range(1, self.nodes):
            g.append(U[k][0] - U[k - 1][0])
            lbg.append([-0.02])
            ubg.append([0.02])  # to be tunned

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

        # optional constraints
        if self.fix_mach:
            for k in range(1, self.nodes):
                g.append(U[k][0] - U[k - 1][0])
                lbg.append([0])
                ubg.append([0])

        if self.fix_alt:
            for k in range(1, self.nodes + 1):
                g.append(X[k][2] - X[k - 1][2])
                lbg.append([0])
                ubg.append([0])

        if self.fix_track:
            for k in range(1, self.nodes):
                g.append(U[k][2] - U[k - 1][2])
                lbg.append([0])
                ubg.append([0])

        if not self.allow_descent:
            for k in range(0, self.nodes):
                g.append(U[k][1])
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

        opts = {
            "ipopt.print_level": ipopt_print,
            "ipopt.sb": "yes",
            "print_time": print_time,
            "ipopt.max_iter": self.ipopt_max_iter,
        }
        self.solver = ca.nlpsol("solver", "ipopt", nlp, opts)
        self.solution = self.solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)

        # final timestep
        ts_final = self.solution["x"][-1].full()[0][0]

        # Function to get x and u from w
        output = ca.Function("output", [w], [X, U], ["w"], ["x", "u"])
        x_opt, u_opt = output(self.solution["x"])

        df = self.to_trajectory(ts_final, x_opt, u_opt)

        return df
