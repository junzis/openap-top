import warnings
from math import pi

import casadi as ca
import numpy as np
import openap.casadi as oc
import pandas as pd
from openap.aero import fpm, ft, kts

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

        ts_min = 0
        ts_max = max(5, self.range / 1000 / 500) * 3600

        h_max = kwargs.get("h_max", self.aircraft["limits"]["ceiling"])
        h_min = kwargs.get("h_min", 15_000 * ft)

        hdg = oc.geo.bearing(self.lat1, self.lon1, self.lat2, self.lon2)
        psi = hdg * pi / 180

        # --- Scaling ---
        scaling = kwargs.get("scaling", False)
        if scaling:
            self.set_scaling(
                scale_x=max(abs(x_max - x_min) / 2, 1.0),
                scale_y=max(abs(y_max - y_min) / 2, 1.0),
                scale_h=max(h_max, 1.0),
                scale_m=max(self.mass_init, 1.0),
                scale_t=max(ts_max, 1.0),
                scale_mach=1.0,
                scale_vs=2500 * fpm,
                scale_psi=np.pi,
                scale_force=50000.0,
                scale_energy=1e6,
                scale_obj=1.0,
            )
        else:
            self.reset_scaling()

        sx = self.scale_x
        sy = self.scale_y
        sh = self.scale_h
        sm = self.scale_m
        st = self.scale_t
        smach = self.scale_mach
        svs = self.scale_vs
        spsi = self.scale_psi

        # Initial conditions - Lower upper bounds
        self.x_0_lb = [xp_0/sx, yp_0/sy, h_min/sh, self.mass_init/sm, ts_min/st]
        self.x_0_ub = [xp_0/sx, yp_0/sy, h_max/sh, self.mass_init/sm, ts_min/st]

        # Final conditions - Lower and upper bounds
        self.x_f_lb = [xp_f/sx, yp_f/sy, h_min/sh, self.oew/sm, ts_min/st]
        self.x_f_ub = [xp_f/sx, yp_f/sy, h_max/sh, self.mass_init/sm, ts_max/st]

        # States - Lower and upper bounds
        self.x_lb = [x_min/sx, y_min/sy, h_min/sh, self.oew/sm, ts_min/st]
        self.x_ub = [x_max/sx, y_max/sy, h_max/sh, self.mass_init/sm, ts_max/st]

        # Control init - lower and upper bounds
        self.u_0_lb = [0.5/smach, (-500*fpm)/svs, (psi - pi/4)/spsi]
        self.u_0_ub = [self.mach_max/smach, (500*fpm)/svs, (psi + pi/4)/spsi]

        # Control final - lower and upper bounds
        self.u_f_lb = [0.5/smach, (-500*fpm)/svs, (psi - pi/4)/spsi]
        self.u_f_ub = [self.mach_max/smach, (500*fpm)/svs, (psi + pi/4)/spsi]

        # Control - Lower and upper bound
        self.u_lb = [0.5/smach, (-500*fpm)/svs, (psi - pi/2)/spsi]
        self.u_ub = [self.mach_max/smach, (500*fpm)/svs, (psi + pi/2)/spsi]

        # Initial guess - states (compute in physical units, then scale)
        self.x_guess = self.initial_guess()
        for i in range(len(self.x_guess)):
            self.x_guess[i] = self.scale_state(self.x_guess[i])

        # Initial guess - controls
        self.u_guess = [0.7/smach, 0/svs, psi/spsi]

    def trajectory(self, objective="fuel", **kwargs) -> pd.DataFrame:
        """
        Computes the optimal trajectory for the aircraft based on the given objective.

        Parameters:
        - objective (str): The objective of the optimization, default is "fuel".
        - **kwargs: Additional keyword arguments.
            - max_fuel (float): Customized maximum fuel constraint.
            - initial_guess (pd.DataFrame): Initial guess for the trajectory. This is
                usually a exsiting flight trajectory.
            - return_failed (bool): If True, returns the DataFrame even if the
                optimization fails. Default is False.

        Returns:
        - pd.DataFrame: A DataFrame containing the optimized trajectory.

        Note:
        - The function uses CasADi for symbolic computation and optimization.
        - The constraints and bounds are defined based on the aircraft's performance
            and operational limits.
        """

        # arguments passed init_condition to overwright h_min and h_max
        self.init_conditions(**kwargs)

        self.init_model(objective, **kwargs)

        customized_max_fuel = kwargs.get("max_fuel", None)

        scaling = kwargs.get("scaling", False)

        initial_guess = kwargs.get("initial_guess", None)
        if initial_guess is not None:
            self.x_guess = self.initial_guess(initial_guess)
            for i in range(len(self.x_guess)):
                self.x_guess[i] = self.scale_state(self.x_guess[i])

        return_failed = kwargs.get("return_failed", False)

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
            elif k == self.nodes - 1:
                lbw.append(self.u_f_lb)
                ubw.append(self.u_f_ub)
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
                if scaling:
                    fj, qj = self.scaled_dynamics(Xc[j - 1], Uk)
                else:
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
        w0.append([self.range * 1000 / 200 / self.scale_t])

        # aircraft performance constraints
        for k in range(self.nodes):
            S = self.aircraft["wing"]["area"]
            # Unscale state/control for physical constraint evaluation
            mass = X[k][3] * self.scale_m
            h = X[k][2] * self.scale_h
            mach = U[k][0] * self.scale_mach
            v = oc.aero.mach2tas(mach, h, dT=self.dT)
            tas = v / kts
            alt = h / ft
            rho = oc.aero.density(h, dT=self.dT)
            thrust_max = self.thrust.cruise(tas, alt, dT=self.dT)
            drag = self.drag.clean(mass, tas, alt, dT=self.dT)

            # max_thrust * 95% > drag (5% margin)
            g.append((thrust_max * 0.95 - drag) / self.scale_force)
            lbg.append([0])
            ubg.append([ca.inf])

            # max lift * 80% > weight (20% margin)
            drag_max = thrust_max * 0.9
            cd_max = drag_max / (0.5 * rho * v**2 * S + 1e-10)
            cd0 = self.drag.polar["clean"]["cd0"]
            ck = self.drag.polar["clean"]["k"]
            cl_max = ca.sqrt(ca.fmax(1e-10, (cd_max - cd0) / ck))
            L_max = cl_max * 0.5 * rho * v**2 * S
            g.append((L_max * 0.8 - mass * oc.aero.g0) / self.scale_force)
            lbg.append([0])
            ubg.append([ca.inf])

        # ts and dt should be consistent
        for k in range(self.nodes - 1):
            g.append(X[k + 1][4] - X[k][4] - self.dt)
            lbg.append([-1.0 / self.scale_t])
            ubg.append([1.0 / self.scale_t])

        # # smooth Mach number change
        # for k in range(self.nodes - 1):
        #     g.append(U[k + 1][0] - U[k][0])
        #     lbg.append([-0.2])
        #     ubg.append([0.2])  # to be tunned

        # # smooth vertical rate change
        # for k in range(self.nodes - 1):
        #     g.append(U[k + 1][1] - U[k][1])
        #     lbg.append([-500 * fpm])
        #     ubg.append([500 * fpm])  # to be tunned

        # smooth heading change
        for k in range(self.nodes - 1):
            g.append(U[k + 1][2] - U[k][2])
            lbg.append([(-15 * pi / 180) / self.scale_psi])
            ubg.append([(15 * pi / 180) / self.scale_psi])

        # optional constraints
        if self.fix_mach:
            for k in range(self.nodes - 1):
                g.append(U[k + 1][0] - U[k][0])
                lbg.append([0])
                ubg.append([0])

        if self.fix_alt:
            for k in range(self.nodes):
                g.append(U[k][1])
                lbg.append([0])
                ubg.append([0])

        if self.fix_track:
            for k in range(self.nodes - 1):
                g.append(U[k + 1][2] - U[k][2])
                lbg.append([0])
                ubg.append([0])

        if not self.allow_descent:
            for k in range(self.nodes):
                g.append(U[k][1])
                lbg.append([0])
                ubg.append([ca.inf])

        # add fuel constraint (in scaled units)
        fuel_consumed = X[0][3] - X[-1][3]
        g.append(fuel_consumed)
        lbg.append([0])
        ubg.append([self.fuel_max / self.scale_m])

        if customized_max_fuel is not None:
            g.append(fuel_consumed - customized_max_fuel / self.scale_m)
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
        ts_final = self.solution["x"][-1].full()[0][0] * self.scale_t

        # Function to get x and u from w
        output = ca.Function("output", [w], [X, U], ["w"], ["x", "u"])
        x_opt, u_opt = output(self.solution["x"])

        # Unscale solution back to physical units
        x_np = x_opt.full()
        u_np = u_opt.full()
        x_np[0, :] *= self.scale_x
        x_np[1, :] *= self.scale_y
        x_np[2, :] *= self.scale_h
        x_np[3, :] *= self.scale_m
        x_np[4, :] *= self.scale_t
        u_np[0, :] *= self.scale_mach
        u_np[1, :] *= self.scale_vs
        u_np[2, :] *= self.scale_psi
        x_opt = ca.DM(x_np)
        u_opt = ca.DM(u_np)

        df = self.to_trajectory(ts_final, x_opt, u_opt, **kwargs)

        df_copy = df.copy()

        # check if the optimizer has failed
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
