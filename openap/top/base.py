import warnings
from math import pi
from typing import Callable, Union

import casadi as ca
import numpy as np
import openap
import openap.casadi as oc
import pandas as pd
from openap.extra.aero import fpm, ft, kts
from pyproj import Proj

try:
    from . import tools
except:
    RuntimeWarning("cfgrib and sklearn are required for wind integration")


class Base:
    def __init__(
        self,
        actype: str,
        origin: Union[str, tuple],
        destination: Union[str, tuple],
        m0: float = 0.8,
        use_synonym=False,
    ):
        """OpenAP trajectory optimizer.

        Args:
            actype (str): ICAO aircraft type code
            origin (Union[str, tuple]): ICAO or IATA code of airport, or tuple (lat, lon)
            destination (Union[str, tuple]): ICAO or IATA code of airport, or tuple (lat, lon)
            m0 (float, optional): Takeoff mass factor. Defaults to 0.8 (of MTOW).
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
        self.engtype = self.aircraft["engine"]["default"]
        self.engine = oc.prop.engine(self.aircraft["engine"]["default"])

        self.initial_mass = m0 * self.aircraft["limits"]["MTOW"]
        self.oew = self.aircraft["limits"]["OEW"]
        self.mlw = self.aircraft["limits"]["MLW"]

        self.use_synonym = use_synonym

        self.thrust = oc.Thrust(actype, use_synonym=self.use_synonym)
        self.wrap = openap.WRAP(actype, use_synonym=self.use_synonym)
        self.drag = oc.Drag(actype, wave_drag=True, use_synonym=self.use_synonym)
        self.fuelflow = oc.FuelFlow(
            actype, wave_drag=True, use_synonym=self.use_synonym
        )
        self.emission = oc.Emission(actype, use_synonym=self.use_synonym)

        # self.proj = Proj(
        #     proj="lcc",
        #     ellps="WGS84",
        #     lat_1=min(self.lat1, self.lat2),
        #     lat_2=max(self.lat1, self.lat2),
        #     lat_0=(self.lat1 + self.lat2) / 2,
        #     lon_0=(self.lon1 + self.lon2) / 2,
        # )

        self.wind = None

        # Check cruise range
        self.range = oc.aero.distance(self.lat1, self.lon1, self.lat2, self.lon2)
        max_range = self.wrap.cruise_range()["maximum"] * 1.2
        if self.range > max_range * 1000:
            warnings.warn(f"The destination is likely out of maximum cruise range.")

        self.debug = False

        self.setup_dc()

    def proj(self, lon, lat, inverse=False, symbolic=False):
        lat0 = (self.lat1 + self.lat2) / 2
        lon0 = (self.lon1 + self.lon2) / 2

        if not inverse:
            if symbolic:
                bearings = oc.aero.bearing(lat0, lon0, lat, lon) / 180 * 3.14159
                distances = oc.aero.distance(lat0, lon0, lat, lon)
                x = distances * ca.sin(bearings)
                y = distances * ca.cos(bearings)
            else:
                bearings = openap.aero.bearing(lat0, lon0, lat, lon) / 180 * 3.14159
                distances = openap.aero.distance(lat0, lon0, lat, lon)
                x = distances * np.sin(bearings)
                y = distances * np.cos(bearings)

            return x, y
        else:
            x, y = lon, lat
            if symbolic:
                distances = ca.sqrt(x**2 + y**2)
                bearing = ca.arctan2(x, y) * 180 / 3.14159
                lat, lon = oc.aero.latlon(lat0, lon0, distances, bearing)
            else:
                distances = np.sqrt(x**2 + y**2)
                bearing = np.arctan2(x, y) * 180 / 3.14159
                lat, lon = openap.aero.latlon(lat0, lon0, distances, bearing)

            return lon, lat

    def enable_wind(self, windfield: pd.DataFrame):
        self.wind = tools.PolyWind(
            windfield, self.proj, self.lat1, self.lon1, self.lat2, self.lon2
        )

    def change_engine(self, engtype):
        self.engtype = engtype
        self.engine = oc.prop.engine(engtype)
        self.thrust = oc.Thrust(
            self.actype,
            engtype,
            use_synonym=self.use_synonym,
            force_engine=True,
        )
        self.fuelflow = oc.FuelFlow(
            self.actype,
            engtype,
            wave_drag=True,
            use_synonym=self.use_synonym,
            force_engine=True,
        )
        self.emission = oc.Emission(self.actype, engtype, use_synonym=self.use_synonym)

    def collocation_coeff(self):
        # Get collocation points
        tau_root = np.append(0, ca.collocation_points(self.polydeg, "legendre"))

        # Coefficients of the collocation equation
        C = np.zeros((self.polydeg + 1, self.polydeg + 1))

        # Coefficients of the continuity equation
        D = np.zeros(self.polydeg + 1)

        # Coefficients of the quadrature function
        B = np.zeros(self.polydeg + 1)

        # Construct polynomial basis
        for j in range(self.polydeg + 1):
            # Construct Lagrange polynomials to get the polynomial basis at the collocation point
            p = np.poly1d([1])
            for r in range(self.polydeg + 1):
                if r != j:
                    p *= np.poly1d([1, -tau_root[r]]) / (tau_root[j] - tau_root[r])

            # Evaluate the polynomial at the final time to get the coefficients of the continuity equation
            D[j] = p(1.0)

            # Evaluate the time derivative of the polynomial at all collocation points to get the coefficients of the continuity equation
            pder = np.polyder(p)
            for r in range(self.polydeg + 1):
                C[j, r] = pder(tau_root[r])

            # Evaluate the integral of the polynomial to get the coefficients of the quadrature function
            pint = np.polyint(p)
            B[j] = pint(1.0)

        return C, D, B

    def xdot(self, x, u) -> ca.MX:
        """Ordinary differential equation for cruising

        Args:
            x (ca.MX): States [x position (m), y position (m), height (m), mass (kg)]
            u (ca.MX): Controls [mach number, vertical speed (m/s), heading (rad)]

        Returns:
            ca.MX: State direvatives
        """
        xp, yp, h, m, ts = x[0], x[1], x[2], x[3], x[4]
        mach, vs, psi = u[0], u[1], u[2]

        v = oc.aero.mach2tas(mach, h)
        gamma = ca.arctan2(vs, v)

        dx = v * ca.sin(psi) * ca.cos(gamma)
        if self.wind is not None:
            dx += self.wind.calc_u(xp, yp, h)

        dy = v * ca.cos(psi) * ca.cos(gamma)
        if self.wind is not None:
            dy += self.wind.calc_v(xp, yp, h)

        dh = vs

        dm = -self.fuelflow.enroute(m, v / kts, h / ft, vs / fpm, limit=False)

        dt = 1

        return ca.vertcat(dx, dy, dh, dm, dt)

    def setup_dc(self, nodes=40, polydeg=3, max_iteration=1000):
        self.nodes = nodes
        self.polydeg = polydeg
        self.ipopt_max_iter = max_iteration

    def init_model(self, objective, **kwargs):
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

        self.ts_final = ca.MX.sym("ts_final")

        # Control discretization
        self.dt = self.ts_final / self.nodes

        # Handel objective function
        if isinstance(objective, Callable):
            function = objective
        elif objective.lower().startswith("ci:"):
            ci = int(objective[3:])
            kwargs["ci"] = ci
            function = self.obj_ci
        else:
            function = getattr(self, f"obj_{objective}")

        L = function(self.x, self.u, self.dt, **kwargs)

        # scale objective based on initial guess
        x0 = self.x_guess.T
        u0 = self.u_guess
        dt0 = self.range / 200 / self.nodes
        cost = np.sum(function(x0, u0, dt0, symbolic=False, **kwargs))
        L = L / cost * 1e3

        # Continuous time dynamics
        self.func_dynamics = ca.Function(
            "f",
            [self.x, self.u],
            [self.xdot(self.x, self.u), L],
            ["x", "u"],
            ["xdot", "L"],
            {"allow_free": True},
        )

    def _calc_emission(self, x, u, symbolic=True):
        xp, yp, h, m = x[0], x[1], x[2], x[3]
        mach, vs, psi = u[0], u[1], u[2]

        if symbolic:
            fuelflow = self.fuelflow
            emission = self.emission
            v = oc.aero.mach2tas(mach, h)
        else:
            fuelflow = openap.FuelFlow(
                self.actype, self.engtype, polydeg=2, use_synonym=self.use_synonym
            )
            emission = openap.Emission(
                self.actype, self.engtype, use_synonym=self.use_synonym
            )
            v = openap.aero.mach2tas(mach, h)

        ff = fuelflow.enroute(m, v / kts, h / ft, vs / fpm, limit=False)
        co2 = emission.co2(ff)
        h2o = emission.h2o(ff)
        sox = emission.sox(ff)
        soot = emission.soot(ff)
        nox = emission.nox(ff, v / kts, h / ft)

        return co2, h2o, sox, soot, nox

    def obj_fuel(self, x, u, dt, symbolic=True, **kwargs):
        xp, yp, h, m, ts = x[0], x[1], x[2], x[3], x[4]
        mach, vs, psi = u[0], u[1], u[2]

        if symbolic:
            fuelflow = self.fuelflow
            v = oc.aero.mach2tas(mach, h)
        else:
            fuelflow = openap.FuelFlow(
                self.actype,
                self.engtype,
                use_synonym=self.use_synonym,
                force_engine=True,
            )
            v = openap.aero.mach2tas(mach, h)

        ff = fuelflow.enroute(m, v / kts, h / ft, vs / fpm, limit=False)
        return ff * dt

    def obj_time(self, x, u, dt, **kwargs):
        return dt

    def obj_ci(self, x, u, dt, ci, **kwargs):
        mach, vs, psi = u[0], u[1], u[2]

        fuel = self.obj_fuel(x, u, dt, **kwargs)

        # time cost 25 eur/min
        time_cost = (dt / 60) * 25

        # fuel cost 0.8 eur/L, Jet A density 0.82
        fuel_cost = fuel * (0.8 / 0.82)

        obj = ci / 100 * time_cost + (1 - ci / 100) * fuel_cost
        return obj

    def obj_gwp20(self, x, u, dt, **kwargs):
        co2, h2o, sox, soot, nox = self._calc_emission(x, u, **kwargs)
        cost = co2 + 0.22 * h2o + 619 * nox - 832 * sox + 4288 * soot
        # cost = cost * 1e-3
        return cost * dt

    def obj_gwp50(self, x, u, dt, **kwargs):
        co2, h2o, sox, soot, nox = self._calc_emission(x, u, **kwargs)
        cost = co2 + 0.1 * h2o + 205 * nox - 392 * sox + 2018 * soot
        # cost = cost * 1e-3
        return cost * dt

    def obj_gwp100(self, x, u, dt, **kwargs):
        co2, h2o, sox, soot, nox = self._calc_emission(x, u, **kwargs)
        cost = co2 + 0.06 * h2o + 114 * nox - 226 * sox + 1166 * soot
        # cost = cost * 1e-3
        return cost * dt

    def obj_gtp20(self, x, u, dt, **kwargs):
        co2, h2o, sox, soot, nox = self._calc_emission(x, u, **kwargs)
        cost = co2 + 0.07 * h2o - 222 * nox - 241 * sox + 1245 * soot
        # cost = cost * 1e-3
        return cost * dt

    def obj_gtp50(self, x, u, dt, **kwargs):
        co2, h2o, sox, soot, nox = self._calc_emission(x, u, **kwargs)
        cost = co2 + 0.01 * h2o - 69 * nox - 38 * sox + 195 * soot
        # cost = cost * 1e-3
        return cost * dt

    def obj_gtp100(self, x, u, dt, **kwargs):
        co2, h2o, sox, soot, nox = self._calc_emission(x, u, **kwargs)
        cost = co2 + 0.008 * h2o + 13 * nox - 31 * sox + 161 * soot
        # cost = cost * 1e-3
        return cost * dt

    def obj_grid_cost(self, x, u, dt, **kwargs):
        xp, yp, h, m, ts = x[0], x[1], x[2], x[3], x[4]
        mach, vs, psi = u[0], u[1], u[2]

        interpolant = kwargs.get("interpolant", None)
        symbolic = kwargs.get("symbolic", True)

        lon, lat = self.proj(xp, yp, inverse=True, symbolic=symbolic)

        if symbolic:
            cost = interpolant(ca.vertcat(lon, lat, h, ts))
        else:
            cost = interpolant(np.array([lon, lat, h, ts])).full()[0]

        return cost

    def obj_combo(self, x, u, dt, obj1, obj2, ratio=0.5, **kwargs):
        if isinstance(obj1, str):
            obj1 = getattr(self, f"obj_{obj1}")

        if isinstance(obj2, str):
            obj2 = getattr(self, f"obj_{obj2}")

        x0 = self.x_guess.T
        u0 = self.u_guess
        dt0 = self.range / 200 / self.nodes

        kwargs_ = kwargs.copy()
        kwargs_["symbolic"] = False

        n1 = obj1(x0, u0, dt0, **kwargs_).sum()
        n2 = obj2(x0, u0, dt0, **kwargs_).sum()

        c1 = obj1(x, u, dt, **kwargs)
        c2 = obj2(x, u, dt, **kwargs)

        return ratio * c1 / n1 + (1 - ratio) * c2 / n2

    def to_trajectory(self, ts_final, x_opt, u_opt, climate_metrics=False):
        X = x_opt.full()
        U = u_opt.full()

        U2 = U[:, -2:-1]
        U1 = U[:, -1:]
        Uf = U1 + (U1 - U2)
        U = np.append(U, Uf, axis=1)
        n = self.nodes + 1

        self.X = X
        self.U = U
        self.dt = ts_final / n

        xp, yp, h, mass, ts = X
        mach, vs, psi = U
        lon, lat = self.proj(xp, yp, inverse=True)
        ts_ = np.linspace(0, ts_final, n).round()

        df = (
            pd.DataFrame()
            .assign(ts=ts_)
            .assign(x=xp)
            .assign(y=yp)
            .assign(h=h)
            .assign(latitude=lat)
            .assign(longitude=lon)
            .assign(altitude=(h / ft).round())
            .assign(mach=mach.round(4))
            .assign(tas=(openap.aero.mach2tas(mach, h) / kts).round(2))
            .assign(vertical_rate=(vs / fpm).round())
            .assign(heading=(np.rad2deg(psi) % 360).round(2))
            .assign(mass=mass.round())
        )

        if self.wind:
            wu = self.wind.calc_u(xp, yp, h)
            wv = self.wind.calc_v(xp, yp, h)
            df = df.assign(wu=wu, wv=wv)

        func_objs = []
        for func in dir(self):
            if "obj_fuel" in func:
                func_objs.append(func)

            if climate_metrics and (("obj_gwp" in func) or ("obj_gtp" in func)):
                func_objs.append(func)

        dt = ts_final / self.nodes

        for fo in func_objs:
            df[fo.replace("obj_", "")] = getattr(self, fo)(X, U, dt, symbolic=False)

        return df
