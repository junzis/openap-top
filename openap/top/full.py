import warnings
from math import pi
from pathlib import Path
import casadi as ca
import numpy as np
import openap
import openap.casadi as oc
import pandas as pd
from mpmath.libmp.libmpf import h_mask
from openap.extra.aero import fpm, ft, kts

from .base import Base
from .climb import Climb
from .cruise import Cruise
from .descent import Descent

try:
    from . import tools
except Exception:
    RuntimeWarning("cfgrib and sklearn are required for wind integration")


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
        ts_max = max(5, self.range / 1000 / 500) * 3600  # max time of 5 hours

        h_max = kwargs.get("h_max", self.aircraft["limits"]["ceiling"])
        h_min = 100 * ft
        # h_max = 11000*ft + 2000
        # h_min = 11000*ft - 2000
        hdg = oc.aero.bearing(self.lat1, self.lon1, self.lat2, self.lon2)
        psi = hdg * pi / 180

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

        # Store unscaled values for reference
        self.x_0_unscaled = [xp_0, yp_0, h_min, self.mass_init, ts_min]
        self.x_f_unscaled = [xp_f, yp_f, h_min, None, None]  # mass and time vary

        # ============================================================
        # SCALED INITIAL CONDITIONS
        # ============================================================
        self.x_0_lb = [
            xp_0 / self.scale_x,
            yp_0 / self.scale_y,
            h_min / self.scale_h,
            self.mass_init / self.scale_m,
            ts_min / self.scale_t
        ]
        self.x_0_ub = self.x_0_lb.copy()

        # ============================================================
        # SCALED FINAL CONDITIONS
        # ============================================================
        self.x_f_lb = [
            xp_f / self.scale_x,
            yp_f / self.scale_y,
            h_min / self.scale_h,
            (self.oew * 0.5) / self.scale_m,
            ts_min / self.scale_t
        ]
        self.x_f_ub = [
            xp_f / self.scale_x,
            yp_f / self.scale_y,
            h_min / self.scale_h,
            self.mass_init / self.scale_m,
            ts_max / self.scale_t
        ]

        # ============================================================
        # SCALED STATE BOUNDS
        # ============================================================
        self.x_lb = [
            x_min / self.scale_x,
            y_min / self.scale_y,
            h_min / self.scale_h,
            (self.oew * 0.5) / self.scale_m,
            ts_min / self.scale_t
        ]
        self.x_ub = [
            x_max / self.scale_x,
            y_max / self.scale_y,
            h_max / self.scale_h,
            self.mass_init / self.scale_m,
            ts_max / self.scale_t
        ]

        # ============================================================
        # SCALED CONTROL BOUNDS
        # ============================================================
        self.u_0_lb = [
            0.1 / self.scale_mach,
            (500 * fpm) / self.scale_vs,
            psi / self.scale_psi
        ]
        self.u_0_ub = [
            0.3 / self.scale_mach,
            (2500 * fpm) / self.scale_vs,
            psi / self.scale_psi
        ]

        self.u_f_lb = [
            0.1 / self.scale_mach,
            (-1500 * fpm) / self.scale_vs,
            psi / self.scale_psi
        ]
        self.u_f_ub = [
            0.3 / self.scale_mach,
            (-300 * fpm) / self.scale_vs,
            psi / self.scale_psi
        ]

        self.u_lb = [
            0.1 / self.scale_mach,
            (-2500 * fpm) / self.scale_vs,
            (psi - pi / 2) / self.scale_psi
        ]
        self.u_ub = [
            self.mach_max / self.scale_mach,
            (2500 * fpm) / self.scale_vs,
            (psi + pi / 2) / self.scale_psi
        ]

        # ============================================================
        # SCALED INITIAL GUESS
        # ============================================================
        self.x_guess = self.initial_guess()
        # Scale the guess
        for i in range(len(self.x_guess)):
            self.x_guess[i][0] /= self.scale_x
            self.x_guess[i][1] /= self.scale_y
            self.x_guess[i][2] /= self.scale_h
            self.x_guess[i][3] /= self.scale_m
            self.x_guess[i][4] /= self.scale_t

        self.u_guess = [
            0.6 / self.scale_mach,
            (1000 * fpm) / self.scale_vs,
            psi / self.scale_psi
        ]

    def unscale_state(self, x_scaled):
        """Unscale state variables back to physical units."""
        x_unscaled = x_scaled.copy()
        x_unscaled[0] *= self.scale_x
        x_unscaled[1] *= self.scale_y
        x_unscaled[2] *= self.scale_h
        x_unscaled[3] *= self.scale_m
        x_unscaled[4] *= self.scale_t
        return x_unscaled

    def unscale_control(self, u_scaled):
        """Unscale control variables back to physical units."""
        u_unscaled = u_scaled.copy()
        u_unscaled[0] *= self.scale_mach
        u_unscaled[1] *= self.scale_vs
        u_unscaled[2] *= self.scale_psi
        return u_unscaled

    def scaled_dynamics(self, x_scaled, u_scaled):
        """
        Compute dynamics with scaled variables.
        Returns scaled derivatives and scaled objective contribution.
        """
        # Unscale for dynamics computation
        x_unscaled = ca.vertcat(
            x_scaled[0] * self.scale_x,
            x_scaled[1] * self.scale_y,
            x_scaled[2] * self.scale_h,
            x_scaled[3] * self.scale_m,
            x_scaled[4] * self.scale_t
        )
        u_unscaled = ca.vertcat(
            u_scaled[0] * self.scale_mach,
            u_scaled[1] * self.scale_vs,
            u_scaled[2] * self.scale_psi
        )

        # Compute unscaled dynamics
        f_unscaled, q_unscaled = self.func_dynamics(x_unscaled, u_unscaled)

        # Scale derivatives: dx_scaled/dt = (dx/dt) / scale
        # But dt is also scaled, so: dx_scaled/dt_scaled = (dx/dt) * (scale_t / scale_x)
        f_scaled = ca.vertcat(
            f_unscaled[0] * (self.scale_t / self.scale_x),
            f_unscaled[1] * (self.scale_t / self.scale_y),
            f_unscaled[2] * (self.scale_t / self.scale_h),
            f_unscaled[3] * (self.scale_t / self.scale_m),
            f_unscaled[4]  # dt/dt = 1 (no scaling needed)
        )

        # Scale objective (if needed)
        q_scaled = q_unscaled * self.scale_t / self.scale_obj

        return f_scaled, q_scaled

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

        # arguments passed init_condition to overwright h_max
        self.init_conditions(**kwargs)

        self.init_model(objective, **kwargs)

        customized_max_fuel = kwargs.get("max_fuel", None)
        initial_guess = kwargs.get("initial_guess", None)
        scaling = kwargs.get("scaling")
        return_failed = kwargs.get("return_failed", False)

        if initial_guess is not None:
            self.x_guess = self.initial_guess(initial_guess)
            # Scale the guess if using scaling
            for i in range(len(self.x_guess)):
                self.x_guess[i][0] /= self.scale_x
                self.x_guess[i][1] /= self.scale_y
                self.x_guess[i][2] /= self.scale_h
                self.x_guess[i][3] /= self.scale_m
                self.x_guess[i][4] /= self.scale_t

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
                # J = J + B[j] * qj * self.dt
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
                lbg.append([(-500 * fpm) / self.scale_vs])
                ubg.append([(500 * fpm) / self.scale_vs])

                g.append(X[k][2])
                lbg.append([(15000 * ft) / self.scale_h])
                ubg.append([ca.inf])

            for k in range(0, idx_toc):
                g.append(U[k][1])
                lbg.append([0])
                ubg.append([ca.inf])

            for k in range(idx_tod, self.nodes):
                g.append(U[k][1])
                lbg.append([-ca.inf])
                ubg.append([0])

        # force and energy constraint
        for k in range(self.nodes):
            # Unscale variables for constraint evaluation
            mass = X[k][3] * self.scale_m
            h = X[k][2] * self.scale_h
            mach = U[k][0] * self.scale_mach
            vs = U[k][1] * self.scale_vs

            S = self.aircraft["wing"]["area"]
            cd0 = self.drag.polar["clean"]["cd0"]
            ck = self.drag.polar["clean"]["k"]

            v = oc.aero.mach2tas(mach, h, dT=self.dT)
            tas = v / kts
            alt = h / ft
            rho = oc.aero.density(h, dT=self.dT)
            thrust_max = self.thrust.cruise(tas, alt, dT=self.dT)
            drag = self.drag.clean(mass, tas, alt, dT=self.dT)

            # max_thrust > drag (5% margin)
            g.append((thrust_max * 0.95 - drag) / self.scale_force)
            lbg.append([0])
            ubg.append([ca.inf])

            # max lift * 80% > weight (20% margin)
            drag_max = thrust_max * 0.9
            cd_max = drag_max / (0.5 * rho * v ** 2 * S + 1e-10)
            cl_max = ca.sqrt(ca.fmax(1e-10, (cd_max - cd0) / ck))
            L_max = cl_max * 0.5 * rho * v ** 2 * S
            g.append((L_max * 0.8 - mass * oc.aero.g0) / self.scale_force)
            lbg.append([0])
            ubg.append([ca.inf])

            # excess energy > change potential energy
            excess_energy = (thrust_max - drag) * v - mass * oc.aero.g0 * vs
            g.append(excess_energy / self.scale_energy)
            lbg.append([0])
            ubg.append([ca.inf])

        # ts and dt should be consistent
        for k in range(self.nodes - 1):
            g.append(X[k + 1][4] - X[k][4] - self.dt)
            lbg.append([-1])
            ubg.append([1])

        # smooth Mach number change
        for k in range(self.nodes - 1):
            g.append(U[k + 1][0] - U[k][0])
            lbg.append([-0.2 / self.scale_mach])
            ubg.append([0.2 / self.scale_mach])

        # smooth vertical rate change
        for k in range(self.nodes - 1):
            g.append(U[k + 1][1] - U[k][1])
            lbg.append([(-500 * fpm) / self.scale_vs])
            ubg.append([(500 * fpm) / self.scale_vs])

        # smooth heading change
        for k in range(self.nodes - 1):
            g.append(U[k + 1][2] - U[k][2])
            lbg.append([(-15 * pi / 180) / self.scale_psi])
            ubg.append([(15 * pi / 180) / self.scale_psi])

        fuel_consumed_scaled = (X[0][3] - X[-1][3])  # Already in scaled units
        fuel_max_scaled = self.fuel_max / self.scale_m

        g.append(fuel_consumed_scaled)
        lbg.append([0])
        ubg.append([fuel_max_scaled])

        if customized_max_fuel is not None:
            custom_fuel_scaled = customized_max_fuel / self.scale_m
            g.append(fuel_consumed_scaled - custom_fuel_scaled)
            lbg.append([-ca.inf])
            ubg.append([0])

        # # add fuel constraint
        # g.append(X[0][3] - X[-1][3])
        # lbg.append([0])
        # ubg.append([self.fuel_max])
        #
        # if customized_max_fuel is not None:
        #     g.append(X[0][3] - X[-1][3] - customized_max_fuel)
        #     lbg.append([-ca.inf])
        #     ubg.append([0])

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
        ts_final *= self.scale_t
        # Function to get x and u from w
        output = ca.Function("output", [w], [X, U], ["w"], ["x", "u"])
        x_opt, u_opt = output(self.solution["x"])

        # Unscale solution if scaling was used
        x_opt_unscaled = x_opt.full()
        u_opt_unscaled = u_opt.full()

        x_opt_unscaled[0, :] *= self.scale_x
        x_opt_unscaled[1, :] *= self.scale_y
        x_opt_unscaled[2, :] *= self.scale_h
        x_opt_unscaled[3, :] *= self.scale_m
        x_opt_unscaled[4, :] *= self.scale_t

        u_opt_unscaled[0, :] *= self.scale_mach
        u_opt_unscaled[1, :] *= self.scale_vs
        u_opt_unscaled[2, :] *= self.scale_psi

        x_opt = ca.DM(x_opt_unscaled)
        u_opt = ca.DM(u_opt_unscaled)

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
