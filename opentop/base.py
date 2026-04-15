import time
import warnings
from typing import Callable, ClassVar, Union

import casadi as ca
import openap.casadi as oc
from openap.aero import fpm, ft, kts

import numpy as np
import openap
import pandas as pd

try:
    from . import tools
except ImportError:
    warnings.warn("cfgrib and sklearn are required for wind integration")

from . import _dynamics, _objectives, _trajectory


def _perturb_guess(df, lateral_km, altitude_ft, proj):
    """Perturb a canonical initial-guess DataFrame.

    Args:
        df: DataFrame with columns longitude, latitude, altitude (ft),
            and typically mass, ts. Other columns are preserved.
        lateral_km: Peak perpendicular offset in km, applied as a sinusoidal
            bulge so origin and destination are unchanged. Sign chooses
            direction; magnitude gives peak deviation.
        altitude_ft: Constant altitude offset in feet applied to every row.
        proj: The optimizer's projection callable; signature
            (lon, lat) -> (xp, yp) and (xp, yp, inverse=True) -> (lon, lat).

    Returns:
        pd.DataFrame: A copy of df with lateral and altitude shifts applied.
        The input DataFrame is not modified.
    """
    out = df.copy()
    # Altitude: constant offset at every node.
    out["altitude"] = df["altitude"] + altitude_ft

    # Lateral: sinusoidal bulge perpendicular to the origin->destination
    # projected vector. sin(0) = sin(pi) = 0 preserves endpoints.
    if lateral_km != 0.0:   # exact-zero fast path; any non-zero value
                             # including sub-mm random draws runs the full
                             # projection so output is consistent
        lon = df["longitude"].values
        lat = df["latitude"].values
        xp, yp = proj(lon, lat)
        xp = np.asarray(xp, dtype=float)
        yp = np.asarray(yp, dtype=float)
        dx = xp[-1] - xp[0]
        dy = yp[-1] - yp[0]
        length = float(np.hypot(dx, dy))
        if length > 0.0:
            # Unit perpendicular: 90° rotation of the (dx, dy) direction.
            perp_x = -dy / length
            perp_y = dx / length
            n = len(xp)
            progress = np.linspace(0.0, 1.0, n)
            offset_m = lateral_km * 1000.0 * np.sin(np.pi * progress)
            xp_new = xp + offset_m * perp_x
            yp_new = yp + offset_m * perp_y
            lon_new, lat_new = proj(xp_new, yp_new, inverse=True)
            lon_new = np.asarray(lon_new, dtype=float)
            lat_new = np.asarray(lat_new, dtype=float)
            # Restore endpoints exactly. Mathematically sin(0) = sin(pi) = 0,
            # but IEEE 754 doubles give sin(np.pi) ~= 1.2e-16. Multiplied by
            # lateral_km * 1000 that residual is ~1e-16 km (IEEE 754 only,
            # since base.py::proj uses np.pi); endpoint clamping defends
            # against potential pyproj round-trip error for non-identity
            # projections. Zeroing offset_m[0] and offset_m[-1] before the
            # call does NOT help because the projection's round-trip error is
            # independent of the offset magnitude — the endpoints must be
            # clamped here.
            lon_new[0] = lon[0]
            lat_new[0] = lat[0]
            lon_new[-1] = lon[-1]
            lat_new[-1] = lat[-1]
            out["longitude"] = lon_new
            out["latitude"] = lat_new
    return out


def _rank_candidates(candidates):
    """Sort candidates best-first: feasibility before objective.

    Uses a stable sort. Ties preserve start-index order (insertion order).

    Args:
        candidates: list of candidate dicts with 'success' (bool) and
            'objective' (float) keys.

    Returns:
        list: new list sorted in best-first order.
    """
    return sorted(candidates, key=lambda c: (not c["success"], c["objective"]))


class Base:
    def __init__(
        self,
        actype: str,
        origin: Union[str, tuple],
        destination: Union[str, tuple],
        m0: float = 0.8,
        dT: float = 0.0,
        use_synonym=False,
        engine: str = None,
    ):
        """OpenAP trajectory optimizer.

        Args:
            actype (str): ICAO aircraft type code
            origin (Union[str, tuple]): ICAO or IATA code of airport, or tuple (lat, lon)
            destination (Union[str, tuple]): ICAO or IATA code of airport, or tuple (lat, lon)
            m0 (float, optional): Takeoff mass factor. Defaults to 0.8 (of MTOW).
            dT (float, optional): Temperature shift from standard ISA. Default = 0
            engine (str, optional): Engine type. Defaults to aircraft's default engine.
        """
        if isinstance(origin, str):
            ap1 = openap.nav.airport(origin)
            self.lat1, self.lon1 = ap1["lat"], ap1["lon"]
        else:
            self.lat1, self.lon1 = origin

        if isinstance(destination, str):
            ap2 = openap.nav.airport(destination)
            self.lat2, self.lon2 = ap2["lat"], ap2["lon"]
        else:
            self.lat2, self.lon2 = destination

        self.actype = actype
        self.aircraft = oc.prop.aircraft(self.actype, use_synonym=use_synonym)
        self.engtype = engine or self.aircraft["engine"]["default"]
        self.engine = oc.prop.engine(self.engtype)

        self.mass_init = m0 * self.aircraft["mtow"]
        self.oew = self.aircraft["oew"]
        self.mlw = self.aircraft["mlw"]
        self.fuel_max = self.aircraft["mfc"]
        self.mach_max = self.aircraft["mmo"]
        self.dT = dT

        self.use_synonym = use_synonym
        force_engine = engine is not None

        self.thrust = oc.Thrust(
            actype,
            self.engtype,
            use_synonym=self.use_synonym,
            force_engine=force_engine,
        )
        self.wrap = openap.WRAP(actype, use_synonym=self.use_synonym)
        self.drag = oc.Drag(actype, wave_drag=True, use_synonym=self.use_synonym)
        self.fuelflow = oc.FuelFlow(
            actype,
            self.engtype,
            wave_drag=True,
            use_synonym=self.use_synonym,
            force_engine=force_engine,
        )
        self.emission = oc.Emission(
            actype,
            self.engtype,
            use_synonym=self.use_synonym,
        )

        self.wind = None

        # Check cruise range
        self.range = oc.geo.distance(self.lat1, self.lon1, self.lat2, self.lon2)
        max_range = self.wrap.cruise_range()["maximum"] * 1.2
        if self.range > max_range * 1000:
            warnings.warn("The destination is likely out of maximum cruise range.")

        self.debug = False
        self.setup()

    def proj(self, lon, lat, inverse=False, symbolic=False):
        """Project between lon/lat and local cartesian coordinates.

        Uses azimuthal equidistant projection centered between origin
        and destination.

        Args:
            lon: Longitude (forward) or x coordinate (inverse).
            lat: Latitude (forward) or y coordinate (inverse).
            inverse: If True, convert (x, y) back to (lon, lat).
            symbolic: If True, use CasADi symbolic math.

        Returns:
            tuple: (x, y) in meters, or (lon, lat) if inverse.
        """
        lat0 = (self.lat1 + self.lat2) / 2
        lon0 = (self.lon1 + self.lon2) / 2

        if symbolic:
            geo, trig = oc.geo, ca
        else:
            geo, trig = openap.aero, np

        if not inverse:
            bearings = geo.bearing(lat0, lon0, lat, lon) / 180 * np.pi
            distances = geo.distance(lat0, lon0, lat, lon)
            return distances * trig.sin(bearings), distances * trig.cos(bearings)
        else:
            x, y = lon, lat
            distances = trig.sqrt(x**2 + y**2)
            bearing = trig.arctan2(x, y) * 180 / np.pi
            lat, lon = geo.latlon(lat0, lon0, distances, bearing)
            return lon, lat

    def initial_guess(self, flight: pd.DataFrame = None):
        """Generate initial guess for the optimizer.

        Args:
            flight: Existing trajectory to use as initial guess.
                If None, uses straight-line interpolation at cruise altitude.

        Returns:
            np.ndarray: Array of shape (nodes+1, 5) with columns
                [xp, yp, h, mass, ts].
        """
        return _dynamics.great_circle_init(
            self.lat1, self.lon1, self.lat2, self.lon2,
            n_nodes=self.nodes,
            mass_init=self.mass_init,
            aircraft=self.aircraft,
            proj=self.proj,
            flight=flight,
        )

    def enable_wind(self, windfield: pd.DataFrame):
        """Enable wind field integration using polynomial regression model.

        Args:
            windfield: DataFrame with columns [longitude, latitude, h, ts, u, v].
        """
        self.wind = tools.PolyWind(
            windfield, self.proj, self.lat1, self.lon1, self.lat2, self.lon2
        )

    def collocation_coeff(self):
        """Compute Legendre collocation coefficients.

        Returns:
            tuple: (C, D, B) where C is the derivative matrix,
                D is the continuity vector, B is the quadrature vector.
        """
        return _dynamics.collocation_coeff(self.polydeg)

    def xdot(self, x, u) -> ca.MX:
        """State derivatives for the equations of motion.

        Args:
            x: State vector [xp (m), yp (m), h (m), mass (kg), ts (s)].
            u: Control vector [mach, vs (m/s), heading (rad)].

        Returns:
            ca.MX: State derivatives [dx, dy, dh, dm, dt].
        """
        return _dynamics.xdot(
            x, u,
            fuelflow=self.fuelflow,
            dT=self.dT,
            wind=self.wind,
        )

    def setup(
        self,
        nodes: int | None = None,
        polydeg: int = 3,
        debug=False,
        max_nodes: int = 120,
        max_iter: int = 3000,
        tol: float = 1e-6,
        acceptable_tol: float = 1e-4,
        ipopt_kwargs=None,
    ):
        """Configure the optimizer discretization and solver settings.

        Args:
            nodes: Number of collocation intervals. Default auto-computed
                from distance (~1 per 50 km, clamped to [20, max_nodes]).
            polydeg: Collocation polynomial degree. Default 3 (Legendre).
            debug: Print solver output. Default False.
            max_nodes: Upper limit for auto-computed nodes. Default 120.
            max_iter: IPOPT maximum iterations. Default 3000.
            tol: IPOPT convergence tolerance. Default 1e-6.
            acceptable_tol: IPOPT acceptable tolerance. Default 1e-4.
            ipopt_kwargs: Additional IPOPT options as {key: value}.
                Keys are passed as "ipopt.{key}".
        """
        if ipopt_kwargs is None:
            ipopt_kwargs = {}
        if nodes is not None:
            self.nodes = nodes
        else:
            self.nodes = int(self.range / 50_000)  # node every 50km

        self.nodes = max(20, self.nodes)
        self.nodes = min(max_nodes, self.nodes)

        self.polydeg = polydeg

        self.debug = debug

        if debug:
            print("Calculating optimal trajectory...")
            ipopt_print = 5
            print_time = 1
        else:
            ipopt_print = 0
            print_time = 0

        self.solver_options = {
            # Convert Opti bounded() constraints to IPOPT variable bounds (lbx/ubx)
            "detect_simple_bounds": True,
            "print_time": print_time,
            "calc_lam_p": False,
            "ipopt.print_level": ipopt_print,
            "ipopt.sb": "yes",
            "ipopt.max_iter": max_iter,
            "ipopt.fixed_variable_treatment": "relax_bounds",
            "ipopt.tol": tol,
            "ipopt.acceptable_tol": acceptable_tol,
            "ipopt.mu_strategy": "adaptive",
        }

        for key, value in ipopt_kwargs.items():
            self.solver_options[f"ipopt.{key}"] = value

    def init_model(self, objective, **kwargs):
        """Build the symbolic dynamics function for the given objective.

        Creates self.x (states), self.u (controls), and self.func_dynamics.
        Must be called after self.ts_final and self.dt are set.

        Args:
            objective: Objective name (str), "ci:N" format, or callable(x, u, dt).
            **kwargs: Passed to the objective function.
        """
        # Model variables
        xp = ca.MX.sym("xp")
        yp = ca.MX.sym("yp")
        h = ca.MX.sym("h")
        m = ca.MX.sym("m")
        ts = ca.MX.sym("ts")

        mach = ca.MX.sym("mach")
        vs = ca.MX.sym("vs")
        psi = ca.MX.sym("psi")

        self.x = ca.vertcat(xp, yp, h, m, ts)
        self.u = ca.vertcat(mach, vs, psi)

        # self.ts_final and self.dt are set by _build_opti() before this call

        # Handle objective function. User callables expect `self.obj_*`
        # helpers (see tests/test_full_flight.py), so they run as-is. String
        # specs go through the pure-function registry with model context
        # injected here.
        if isinstance(objective, Callable):
            self.objective = objective
            L = self.objective(self.x, self.u, self.dt, **kwargs)
        else:
            resolved = _objectives.resolve_objective(objective)
            ctx = self._objective_ctx()
            ctx.update(kwargs)
            self.objective = lambda x, u, dt, **kw: resolved(
                x, u, dt, **{**ctx, **kw}
            )
            L = resolved(self.x, self.u, self.dt, **ctx)

        # Continuous time dynamics
        self.func_dynamics = ca.Function(
            "f",
            [self.x, self.u],
            [self.xdot(self.x, self.u), L],
            ["x", "u"],
            ["xdot", "L"],
            {"allow_free": True},
        )

    def _build_opti(self, objective, ts_final_guess, **kwargs):
        """Build CasADi Opti problem with direct collocation structure.

        Creates the Opti instance, free final time variable, calls init_model,
        and builds the collocation equations, variable bounds, and initial guesses.

        Must be called after init_conditions() which sets the bound attributes.

        Args:
            objective: Objective function name or callable.
            ts_final_guess: Initial guess for total flight time (seconds).
            **kwargs: Passed through to init_model().

        Returns:
            tuple: (X, U) where X is list of state MX vars at each node boundary
                   (length nodes+1), U is list of control MX vars (length nodes).
        """
        self._opti = ca.Opti()

        # Grid-cost interpolants (bspline) need IPOPT's exact Hessian for
        # numerical stability. For string objective specs this is driven by
        # the registry's `requires_exact_hessian` flag; for user-supplied
        # callables (which might internally call obj_grid_cost) the same
        # effect is triggered by the presence of an `interpolant` kwarg.
        needs_exact_hessian = False
        if not isinstance(objective, Callable):
            try:
                resolved = _objectives.resolve_objective(objective)
                needs_exact_hessian = getattr(
                    resolved, "requires_exact_hessian", False
                )
            except (ValueError, TypeError):
                # Defer the error until init_model so the message matches the
                # existing dispatch path.
                pass
        if kwargs.get("interpolant") is not None:
            needs_exact_hessian = True
        if needs_exact_hessian:
            self.solver_options["ipopt.hessian_approximation"] = "exact"

        # Free final time — must be set before init_model
        self.ts_final = self._opti.variable()
        self._opti.subject_to(self.ts_final >= 0)
        self._opti.set_initial(self.ts_final, ts_final_guess)
        self.dt = self.ts_final / self.nodes

        # Build dynamics function (captures self.dt with free ts_final)
        self.init_model(objective, **kwargs)

        C, D, B = self.collocation_coeff()
        nstates = self.x.shape[0]

        X = []  # States at node boundaries (length: nodes + 1)
        U = []  # Controls at each node (length: nodes)
        J = 0  # Objective accumulator

        # Initial state
        Xk = self._opti.variable(nstates)
        self._opti.subject_to(self._opti.bounded(self.x_0_lb, Xk, self.x_0_ub))
        self._opti.set_initial(Xk, self.x_guess[0])
        X.append(Xk)

        for k in range(self.nodes):
            # Control variable
            Uk = self._opti.variable(self.u.shape[0])
            U.append(Uk)

            if k == 0:
                u_lb, u_ub = self.u_0_lb, self.u_0_ub
            elif k == self.nodes - 1:
                u_lb, u_ub = self.u_f_lb, self.u_f_ub
            else:
                u_lb, u_ub = self.u_lb, self.u_ub

            self._opti.subject_to(self._opti.bounded(u_lb, Uk, u_ub))
            self._opti.set_initial(Uk, self.u_guess)

            # Collocation points within this interval
            Xc = []
            for j in range(self.polydeg):
                Xkj = self._opti.variable(nstates)
                Xc.append(Xkj)
                self._opti.subject_to(self._opti.bounded(self.x_lb, Xkj, self.x_ub))
                self._opti.set_initial(Xkj, self.x_guess[k])

            # Collocation equations and quadrature
            Xk_end = D[0] * Xk
            for j in range(1, self.polydeg + 1):
                xpc = C[0, j] * Xk
                for r in range(self.polydeg):
                    xpc = xpc + C[r + 1, j] * Xc[r]

                fj, qj = self.func_dynamics(Xc[j - 1], Uk)
                self._opti.subject_to(self.dt * fj == xpc)

                Xk_end = Xk_end + D[j] * Xc[j - 1]
                J = J + B[j] * qj

            # State at end of interval
            Xk = self._opti.variable(nstates)
            X.append(Xk)

            if k < self.nodes - 1:
                x_lb, x_ub = self.x_lb, self.x_ub
            else:
                x_lb, x_ub = self.x_f_lb, self.x_f_ub

            self._opti.subject_to(self._opti.bounded(x_lb, Xk, x_ub))
            self._opti.set_initial(Xk, self.x_guess[k])

            # Continuity constraint
            self._opti.subject_to(Xk_end == Xk)

        # Optional: rescale the objective by its value at the initial guess
        # so IPOPT sees f(x0) ≈ 1. Important for objectives whose natural
        # magnitude is far from O(1) — for example climate-metric objectives
        # combining contrail ATR20 (~1e-12 K/s) and CO2 (~7e-15 K/kg·s),
        # where IPOPT's default termination tolerances would otherwise be
        # satisfied at any feasible point. The physical objective_value is
        # multiplied back in _solve so callers see the unscaled number.
        self._objective_rescale = 1.0
        if kwargs.get("auto_rescale_objective", False):
            x_init = self._opti.debug.value(self._opti.x, self._opti.initial())
            j_at_init = ca.Function("j_at_init", [self._opti.x], [J])
            f0 = float(j_at_init(x_init))
            # Only skip rescaling if f0 is essentially zero to avoid
            # divide-by-zero. Otherwise rescale by abs(f0) in either
            # direction — crucial for climate-metric objectives where
            # the natural magnitude is far below 1.
            if abs(f0) > 1e-30:
                self._objective_rescale = abs(f0)
                J = J / self._objective_rescale

        self._opti.minimize(J)

        return X, U

    def _solve(self, X, U, **kwargs):
        """Solve the Opti NLP and extract trajectory DataFrame.

        Args:
            X: List of state MX variables from _build_opti.
            U: List of control MX variables from _build_opti.
            **kwargs: Passed through to to_trajectory().

        Returns:
            pd.DataFrame: Trajectory DataFrame.
        """
        self._opti.solver("ipopt", self.solver_options)

        try:
            sol = self._opti.solve()
        except RuntimeError as e:
            if self.debug:
                warnings.warn(f"Solver failed: {e}")
            sol = self._opti.debug

        self.solver = sol
        # Undo auto_rescale_objective so callers always see the physical value.
        self.objective_value = float(sol.value(self._opti.f)) * self._objective_rescale

        ts_final_val = float(sol.value(self.ts_final))
        x_opt = sol.value(ca.horzcat(*X))
        u_opt = sol.value(ca.horzcat(*U))

        return self.to_trajectory(ts_final_val, x_opt, u_opt, **kwargs)

    def _calc_emission(self, x, u, symbolic=True):
        """Compute emission species from state and control vectors.

        Returns:
            tuple: (co2, h2o, sox, soot, nox) emission rates.
        """
        xp, yp, h, m = x[0], x[1], x[2], x[3]
        mach, vs, psi = u[0], u[1], u[2]

        if symbolic:
            fuelflow = self.fuelflow
            emission = self.emission
            v = oc.aero.mach2tas(mach, h, dT=self.dT)
        else:
            fuelflow = openap.FuelFlow(
                self.actype, self.engtype, polydeg=2, use_synonym=self.use_synonym
            )
            emission = openap.Emission(
                self.actype, self.engtype, use_synonym=self.use_synonym
            )
            v = openap.aero.mach2tas(mach, h, dT=self.dT)

        ff = fuelflow.enroute(m, v / kts, h / ft, vs / fpm, dT=self.dT)
        co2 = emission.co2(ff)
        h2o = emission.h2o(ff)
        sox = emission.sox(ff)
        soot = emission.soot(ff)
        nox = emission.nox(ff, v / kts, h / ft, dT=self.dT)

        return co2, h2o, sox, soot, nox

    # Preserved for back-compat on `Base._CLIMATE_COEFF`; authoritative copy
    # lives in opentop._objectives.
    _CLIMATE_COEFF: ClassVar[dict] = _objectives._CLIMATE_COEFF

    def _objective_ctx(self):
        """Build the model context dict pure objectives need as kwargs."""
        return {
            "fuelflow": self.fuelflow,
            "dT": self.dT,
            "actype": self.actype,
            "engtype": self.engtype,
            "use_synonym": self.use_synonym,
            "proj": self.proj,
            "calc_emission": self._calc_emission,
        }

    def obj_fuel(self, x, u, dt, symbolic=True, **kwargs):
        """Fuel burn objective: fuelflow * dt. Delegates to _objectives."""
        ctx = self._objective_ctx()
        ctx.update(kwargs)
        return _objectives.obj_fuel(x, u, dt, symbolic=symbolic, **ctx)

    def obj_time(self, x, u, dt, **kwargs):
        """Minimum time objective. Delegates to _objectives."""
        return _objectives.obj_time(x, u, dt, **kwargs)

    def obj_ci(self, x, u, dt, ci, time_price=25, fuel_price=0.8, **kwargs):
        """Cost index objective blending time and fuel costs.

        Args:
            ci: Cost index (0-100). 0 = fuel only, 100 = time only.
            time_price: Cost of time in EUR/min. Default 25.
            fuel_price: Cost of fuel in EUR/L. Default 0.8.
        """
        ctx = self._objective_ctx()
        ctx.update(kwargs)
        return _objectives.obj_ci(
            x, u, dt,
            ci=ci, time_price=time_price, fuel_price=fuel_price,
            **ctx,
        )

    def _obj_climate(self, x, u, dt, metric, **kwargs):
        """Climate impact objective using GWP/GTP metric coefficients."""
        ctx = self._objective_ctx()
        ctx.update(kwargs)
        return _objectives.obj_climate(x, u, dt, metric=metric, **ctx)

    def obj_grid_cost(self, x, u, dt, **kwargs):
        """Grid-based cost objective using a CasADi interpolant.

        Args:
            **kwargs:
                interpolant: CasADi interpolant function.
                symbolic: Use symbolic computation. Default True.
                n_dim: Input dimension, 3 (lon,lat,h) or 4 (+ts). Default 3.
                time_dependent: Multiply cost by dt. Default True.
        """
        return _objectives.obj_grid_cost(
            x, u, dt, proj=self.proj, **kwargs
        )

    def to_trajectory(self, ts_final, x_opt, u_opt, **kwargs):
        """Convert optimization results to a trajectory DataFrame.

        Args:
            ts_final: Final timestamp
            x_opt: Optimized states
            u_opt: Optimized controls
            **kwargs: Additional arguments including:
                - interpolant: Grid cost interpolant function
                - time_dependent: Whether grid cost is time dependent (default True)
                - n_dim: Dimension of grid cost, 3 or 4 (auto-detected from
                  interpolant if not provided)

        Returns:
            pd.DataFrame: Trajectory with columns including fuel_cost and grid_cost
        """
        # Historically to_trajectory accepted arbitrary kwargs (via
        # kwargs.get) and silently ignored unknowns; preserve that contract
        # by passing only the recognised ones through to the pure helper.
        df, X, U, dt = _trajectory.to_dataframe(
            ts_final,
            x_opt,
            u_opt,
            proj=self.proj,
            nodes=self.nodes,
            dT=self.dT,
            wind=self.wind,
            actype=self.actype,
            engtype=self.engtype,
            use_synonym=self.use_synonym,
            interpolant=kwargs.get("interpolant", None),
            time_dependent=kwargs.get("time_dependent", True),
            n_dim=kwargs.get("n_dim"),
        )
        # Preserve historical side effects for downstream callers/tests.
        self.X = X
        self.U = U
        self.dt = dt
        return df

    def multi_start_trajectory(
        self,
        *,
        n_starts=5,
        lateral_jitter_km=100.0,
        altitude_jitter_ft=3000.0,
        seed=None,
        **trajectory_kwargs,
    ):
        """Run N solves from different initial guesses and return the best.

        Start 0 uses the user's ``initial_guess=`` (if given) or the
        optimizer's default great-circle guess. Starts 1..N-1 apply random
        lateral and altitude perturbations to a canonical basis DataFrame:
        if ``initial_guess=`` was supplied, that DataFrame is the basis;
        otherwise start 0's solved trajectory is the basis.

        Args:
            n_starts: Total number of solves (>= 1). n_starts=1 matches a
                plain ``trajectory()`` call.
            lateral_jitter_km: Peak lateral perpendicular offset, in km.
                Perturbations sample uniformly from [-jitter, +jitter]. The
                default is tuned for routes of roughly 500-2000 km; tighten
                for short hops, widen for trans-oceanic flights.
            altitude_jitter_ft: Peak altitude offset, in feet. Default
                covers about one flight-level band — appropriate for most
                cruise-altitude exploration.
            seed: RNG seed for reproducible perturbations. None = OS entropy.
            **trajectory_kwargs: Forwarded to ``trajectory()``.

        Returns:
            (trajectory, candidates):
                trajectory: the winning ``pd.DataFrame`` (feasible + lowest
                    objective, falling back to lowest objective among stalled
                    runs if none converged).
                candidates: list of per-start dicts, best-first ordered.
                    Each dict has keys: start_index, objective, fuel,
                    grid_cost, success, return_status, iters, perturbation,
                    wall_time_s, trajectory.

        Raises:
            ValueError: if n_starts < 1 or seed is an invalid type.
            Exceptions from underlying trajectory() or _perturb_guess calls
            propagate as-is; the loop does not swallow errors from individual
            starts. An IPOPT failure inside trajectory() returns a
            ``success=False`` candidate rather than raising.
        """
        if n_starts < 1:
            raise ValueError(f"n_starts must be >= 1, got {n_starts}")
        if seed is not None and (not isinstance(seed, int) or seed < 0):
            raise ValueError(
                f"seed must be a non-negative integer or None, got {seed!r}"
            )

        has_interpolant = trajectory_kwargs.get("interpolant") is not None

        def _make_candidate(index, df, lat_km, alt_ft, wall_time_s):
            stats = self.solver.stats() if hasattr(self, "solver") else {}
            grid = (
                float(df["grid_cost"].sum(skipna=True))
                if has_interpolant
                else float("nan")
            )
            return {
                "start_index": index,
                "objective": float(getattr(self, "objective_value", float("nan"))),
                "fuel": float(df["mass"].iloc[0] - df["mass"].iloc[-1]),
                "grid_cost": grid,
                "success": bool(stats.get("success")),
                "return_status": str(stats.get("return_status", "")),
                "iters": int(stats.get("iter_count", 0)),
                "perturbation": {"lateral_km": float(lat_km),
                                  "altitude_ft": float(alt_ft)},
                "wall_time_s": float(wall_time_s),
                "trajectory": df,
            }

        candidates = []

        # Start 0: canonical (no perturbation).
        t0 = time.perf_counter()
        df_0 = self.trajectory(**trajectory_kwargs)
        wall_0 = time.perf_counter() - t0
        candidates.append(_make_candidate(0, df_0, 0.0, 0.0, wall_0))

        if n_starts > 1:
            rng = np.random.default_rng(seed)
            # Canonical DataFrame to perturb: prefer the user-provided
            # initial_guess if given, else the DataFrame produced by start 0.
            canonical_df = trajectory_kwargs.get("initial_guess")
            if canonical_df is None:
                canonical_df = df_0

            for i in range(1, n_starts):
                lat_km = float(rng.uniform(-lateral_jitter_km, lateral_jitter_km))
                alt_ft = float(rng.uniform(-altitude_jitter_ft, altitude_jitter_ft))
                perturbed = _perturb_guess(
                    canonical_df, lat_km, alt_ft, proj=self.proj
                )
                ts = time.perf_counter()
                df_i = self.trajectory(
                    **{**trajectory_kwargs, "initial_guess": perturbed}
                )
                wall = time.perf_counter() - ts
                candidates.append(_make_candidate(i, df_i, lat_km, alt_ft, wall))

        candidates = _rank_candidates(candidates)
        return candidates[0]["trajectory"], candidates
