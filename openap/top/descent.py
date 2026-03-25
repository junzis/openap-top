from math import pi

import casadi as ca
import openap.casadi as oc
from openap.aero import fpm, ft, kts

import numpy as np
import pandas as pd

from .base import Base
from .cruise import Cruise


class Descent(Base):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.curise = Cruise(*args, **kwargs)

    def init_conditions(self, df_cruise, **kwargs):
        """Initialize direct collocation bounds and guesses.

        Args:
            df_cruise: Cruise trajectory DataFrame.
            alt_start: Start altitude in feet. If provided, used instead of df_cruise.
        """
        waypoints = kwargs.get("waypoints", None)

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

        alt_start = kwargs.get("alt_start", None)
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
        self.x_0_lb = [xp_0, yp_0, h_start - 100, mass_tod, ts_min]
        self.x_0_ub = [xp_0, yp_0, h_start + 100, mass_tod, ts_min]

        # Final conditions - Lower and upper bounds
        self.x_f_lb = [xp_f, yp_f, h_min, mass_oew, ts_min]
        self.x_f_ub = [xp_f, yp_f, h_min, mass_tod, ts_max]

        # States - Lower and upper bounds
        self.x_lb = [x_min, y_min, h_min, mass_oew, ts_min]
        self.x_ub = [x_max, y_max, h_start + 100, mass_tod, ts_max]

        # States - guesses
        xp_guess = np.linspace(xp_0, xp_f, self.nodes + 1)
        yp_guess = np.linspace(yp_0, yp_f, self.nodes + 1)
        h_guess = np.linspace(h_start, h_min, self.nodes + 1)
        m_guess = mass_tod * np.ones(self.nodes + 1)
        ts_guess = np.linspace(0, 6 * 3600, self.nodes + 1)
        self.x_guess = np.vstack([xp_guess, yp_guess, h_guess, m_guess, ts_guess]).T
        dt_min = 5
        dt_max = ts_max

        # Control init - lower and upper bounds
        self.u_0_lb = [cruise_mach - 0.1, -2500 * fpm, od_psi - pi / 4, dt_min]
        self.u_0_ub = [cruise_mach + 0.1, 200 * fpm, od_psi + pi / 4, dt_max]

        if waypoints is None:
            # Control final - lower and upper bounds (-- dt final should be small)
            self.u_f_lb = [0.1, -2500 * fpm, od_psi - pi / 4, dt_min]
            self.u_f_ub = [0.3, 200 * fpm, od_psi + pi / 4, dt_max]
        else:
            # Control final - lower and upper bounds
            self.u_f_lb = [0.1, -2500 * fpm, od_psi - pi / 4, dt_min]
            self.u_f_ub = [0.3, 200 * fpm, od_psi + pi / 4, 50]
        # Control - Lower and upper bound
        self.u_lb = [0.1, -2500 * fpm, od_psi - pi / 2, dt_min]
        self.u_ub = [cruise_mach, 200 * fpm, od_psi + pi / 2, dt_max]

        # Control - guesses
        self.u_guess = [0.7, 0 * fpm, od_psi, 100]

    def trajectory(self, objective="fuel", df_cruise=None, **kwargs) -> pd.DataFrame:

        alt_start = kwargs.get("alt_start", None)
        waypoints = kwargs.get("waypoints", None)
        uniform_nodes = kwargs.get("uniform_nodes", False)
        if df_cruise is None:
            if self.debug:
                print("Finding the preliminary optimal cruise parameters...")
            df_cruise = self.curise.trajectory(objective)

        if self.debug:
            print("Calculating optimal descent trajectory...")

        self.init_conditions(df_cruise, alt_start=alt_start)
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
                g.append(U[k][3] * fj - xpc)
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

        # # difference between dts
        # for k in range(self.nodes - 1):
        #     g.append(ca.fabs(U[k][3] - U[k + 1][3]))
        #     lbg.append([20])
        #     ubg.append([500])

        if waypoints is not None:
            for wp in waypoints:
                # wpx, wpy = self.proj(wp[1], wp[0])
                dist_min = 21_000_000
                for k in range(self.nodes):
                    lon_k, lat_k = self.proj(
                        X[k][0], X[k][1], inverse=True, symbolic=True
                    )
                    dist_min = ca.fmin(
                        oc.geo.distance(lat_k, lon_k, wp[0], wp[1]), dist_min
                    )
                g.append(dist_min)
                lbg.append([0])
                ubg.append([2000])
            # ts and dt should be consistent
            if uniform_nodes:
                for k in range(self.nodes - 1):
                    g.append(X[k + 1][4] - X[k][4] - U[k + 1][3])
                    lbg.append([-20])
                    ubg.append([20])
            else:
                for k in range(self.nodes - 1):
                    g.append(X[k + 1][4] - X[k][4] - U[k + 1][3])
                    lbg.append([-200])
                    ubg.append([200])
        else:
            for k in range(self.nodes - 1):
                g.append(X[k + 1][4] - X[k][4] - U[k + 1][3])
                lbg.append([-0])
                ubg.append([0])

        # t_final is the sum of dts
        sum_t = 0
        for k in range(self.nodes):
            sum_t = sum_t + U[k][3]
        g.append(sum_t - self.ts_final)
        lbg.append([-1])
        ubg.append([1])

        # # smooth Mach number changes
        # for k in range(1, self.nodes):
        #     g.append(U[k][0] - U[k - 1][0])
        #     lbg.append([-0.2])
        #     ubg.append([0.2])  # to be tunned

        # cas constraint
        for k in range(self.nodes):
            cas = oc.aero.mach2cas(U[k][0], X[k][2], dT=self.dT)
            cas_max = self.aircraft["vmo"] * kts
            g.append(cas)
            lbg.append([0])
            ubg.append([cas_max])

        # smooth cas change
        for k in range(1, self.nodes - 1):
            cask = oc.aero.mach2cas(U[k][0], X[k][2], dT=self.dT)
            cask1 = oc.aero.mach2cas(U[k + 1][0], X[k + 1][2], dT=self.dT)
            g.append((cask1 - cask) / U[k][3])
            lbg.append([-1])  # per second
            ubg.append([1])  # per second

        # smooth vertical rate change
        for k in range(self.nodes - 1):
            g.append((U[k + 1][1] - U[k][1]) / U[k][3])
            lbg.append([-5 * fpm])  # per second
            ubg.append([5 * fpm])  # per second

        # smooth heading change
        for k in range(self.nodes - 1):
            g.append((U[k + 1][2] - U[k][2]) / U[k][3])
            lbg.append([-0.5 * pi / 180])  # per second
            ubg.append([0.5 * pi / 180])  # per second

        if waypoints is not None:
            if uniform_nodes:
                # More nodes at lower altitudes
                for k in range(self.nodes - 1):
                    g.append((X[k + 1][2] - X[k][2]))
                    lbg.append([-2_000])
                    ubg.append([2_000])
            else:
                # More nodes at lower altitudes
                for k in range(self.nodes - 1):
                    g.append((X[k + 1][2] - X[k][2]))
                    lbg.append([-1_000])
                    ubg.append([1_000])

        # force and energy constraint
        for k in range(self.nodes):
            S = self.aircraft["wing"]["area"]
            cd0 = self.drag.polar["clean"]["cd0"]
            ck = self.drag.polar["clean"]["k"]
            mass = X[k][3]
            v = oc.aero.mach2tas(U[k][0], X[k][2], dT=self.dT)
            tas = v / kts
            alt = X[k][2] / ft
            rho = oc.aero.density(X[k][2], dT=self.dT)
            thrust_max = self.thrust.cruise(tas, alt, dT=self.dT)
            drag = self.drag.clean(mass, tas, alt, dT=self.dT)

            # max_thrust > drag (5% margin)
            g.append(thrust_max * 0.95 - drag)
            lbg.append([0])
            ubg.append([ca.inf])

            # max lift * 80% > weight (20% margin)
            drag_max = thrust_max * 0.9
            cd_max = drag_max / (0.5 * rho * v**2 * S + 1e-10)
            cd0 = self.drag.polar["clean"]["cd0"]
            ck = self.drag.polar["clean"]["k"]
            cl_max = ca.sqrt(ca.fmax(1e-10, (cd_max - cd0) / ck))
            L_max = cl_max * 0.5 * rho * v**2 * S
            g.append(L_max * 0.8 - mass * oc.aero.g0)
            lbg.append([0])
            ubg.append([ca.inf])

            # excess energy > change potential energy
            excess_energy = (thrust_max - drag) * v - mass * oc.aero.g0 * U[k][1]
            g.append(excess_energy)
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

        df = self.to_trajectory(ts_final, x_opt, u_opt, **kwargs)

        remove_cruise = kwargs.get("remove_cruise", True)
        if remove_cruise:
            df = df.query("vertical_rate < -100")

        return df
