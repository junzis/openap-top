"""Joint separation-constrained optimization of multiple flight phases."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from math import atan2, cos, degrees, isfinite, radians, sin, sqrt
from types import MappingProxyType
from typing import Any, cast

import casadi as ca

import numpy as np
import pandas as pd

from ._multi_start import _perturb_guess
from ._separation import (
    PairSeparationReport,
    SeparationConfig,
    interpolate_collocation_state,
    numeric_collocation_state,
    separation_metric,
)
from ._transcription import AircraftTranscription
from .cruise import Cruise
from .descent import Descent


@dataclass(frozen=True, slots=True)
class FlightSpec:
    """One flight participating in a joint optimization.

    ``options`` apply to both the independent warm start and joint solve.
    ``joint_options`` override them only while building the shared NLP.
    """

    id: str
    optimizer: Cruise | Descent
    start_time: float = 0.0
    objective: str | Callable[..., Any] = "fuel"
    weight: float = 1.0
    initial_guess: pd.DataFrame | None = None
    options: Mapping[str, Any] = field(default_factory=dict)
    joint_options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id.strip():
            raise ValueError("flight id must be a non-empty string")
        if not isinstance(self.optimizer, (Cruise, Descent)):
            raise TypeError(
                "MultiAircraft currently supports Cruise and Descent optimizers only"
            )
        if not isfinite(self.start_time):
            raise ValueError("start_time must be finite")
        if not isfinite(self.weight) or self.weight <= 0:
            raise ValueError("weight must be finite and positive")
        object.__setattr__(self, "options", MappingProxyType(dict(self.options)))
        object.__setattr__(
            self, "joint_options", MappingProxyType(dict(self.joint_options))
        )


@dataclass(frozen=True, slots=True)
class MultiAircraftResult:
    """Structured result of a shared multi-aircraft solve."""

    trajectories: dict[str, pd.DataFrame]
    success: bool
    solver_success: bool
    separation_success: bool
    status: str
    objective: float
    aircraft_objectives: dict[str, float]
    weights: dict[str, float]
    pair_reports: tuple[PairSeparationReport, ...]
    stats: dict[str, Any]
    nlp_variables: int
    nlp_constraints: int
    separation_constraints: int
    refinement_rounds: int
    build_time_s: float
    solve_time_s: float
    verification_time_s: float


def _common_projection_center(flights: tuple[FlightSpec, ...]) -> tuple[float, float]:
    vectors = []
    for flight in flights:
        optimizer = flight.optimizer
        for lat, lon in (
            (optimizer.lat1, optimizer.lon1),
            (optimizer.lat2, optimizer.lon2),
        ):
            lat_rad = radians(lat)
            lon_rad = radians(lon)
            vectors.append(
                (
                    cos(lat_rad) * cos(lon_rad),
                    cos(lat_rad) * sin(lon_rad),
                    sin(lat_rad),
                )
            )
    x, y, z = np.mean(np.asarray(vectors, dtype=float), axis=0)
    horizontal = sqrt(x * x + y * y)
    if horizontal < 1e-12 and abs(z) < 1e-12:
        raise ValueError("flight endpoints do not define a stable projection center")
    return degrees(atan2(z, horizontal)), degrees(atan2(y, x))


@contextmanager
def _fleet_projection(flights: tuple[FlightSpec, ...], center: tuple[float, float]):
    previous = []
    for flight in flights:
        optimizer = flight.optimizer
        previous.append((optimizer, optimizer._projection_center))
        optimizer._projection_center = center
    try:
        yield
    finally:
        for optimizer, value in previous:
            optimizer._projection_center = value


def _canonical_route_sign(trajectory: pd.DataFrame, proj: Any) -> float:
    """Orient a route consistently when choosing a lateral warm-start side."""
    xp, yp = proj(
        trajectory.longitude.iloc[[0, -1]].to_numpy(dtype=float),
        trajectory.latitude.iloc[[0, -1]].to_numpy(dtype=float),
    )
    dx = float(np.asarray(xp)[-1] - np.asarray(xp)[0])
    dy = float(np.asarray(yp)[-1] - np.asarray(yp)[0])
    dominant = dx if abs(dx) >= abs(dy) else dy
    return -1.0 if dominant < 0.0 else 1.0


class MultiAircraft:
    """Coordinate multiple phase optimizers in one shared CasADi Opti NLP."""

    def __init__(
        self,
        flights: list[FlightSpec] | tuple[FlightSpec, ...],
        separation: SeparationConfig | None = None,
        *,
        enforce_separation: bool = True,
        debug: bool = False,
        max_iter: int | None = None,
        ipopt_kwargs: Mapping[str, Any] | None = None,
        common_altitude: bool = False,
        minimum_arrival_gap_s: float | None = None,
    ) -> None:
        if not flights:
            raise ValueError("at least one flight is required")
        self.flights = tuple(flights)
        ids = [flight.id for flight in self.flights]
        if len(ids) != len(set(ids)):
            raise ValueError("flight ids must be unique")
        self.separation = separation or SeparationConfig()
        self.enforce_separation = bool(enforce_separation)
        self.debug = debug
        self.max_iter = max_iter
        self.ipopt_kwargs = dict(ipopt_kwargs or {})
        self.common_altitude = bool(common_altitude)
        if minimum_arrival_gap_s is not None and (
            not isfinite(minimum_arrival_gap_s) or minimum_arrival_gap_s <= 0
        ):
            raise ValueError("minimum_arrival_gap_s must be finite and positive")
        self.minimum_arrival_gap_s = minimum_arrival_gap_s
        self._validate_scope()

    def _validate_scope(self) -> None:
        for flight in self.flights:
            optimizer = flight.optimizer
            if self.common_altitude:
                if not isinstance(optimizer, Cruise):
                    raise ValueError(
                        "common_altitude is supported for Cruise optimizers only"
                    )
                if not optimizer.fix_alt:
                    raise ValueError(
                        "common_altitude requires fix_cruise_altitude() on every flight"
                    )
            if optimizer.wind is not None:
                raise NotImplementedError(
                    "MultiAircraft does not yet support projection-dependent "
                    "wind models"
                )
            if self.minimum_arrival_gap_s is not None and not isinstance(
                optimizer, Descent
            ):
                raise ValueError(
                    "minimum_arrival_gap_s is supported for Descent optimizers only"
                )
            options = dict(flight.options)
            if "initial_guess" in options:
                raise ValueError("set initial_guess on FlightSpec, not in options")
            if options.get("variable_timestep", False):
                raise NotImplementedError(
                    "MultiAircraft currently requires uniform timesteps"
                )
            if options.get("interpolant") is not None and options.get(
                "time_dependent", False
            ):
                raise NotImplementedError(
                    "time-dependent grid objectives need fleet absolute-time support"
                )
            unsupported = {"return_failed", "result_object"}.intersection(options)
            if unsupported:
                names = ", ".join(sorted(unsupported))
                raise ValueError(f"FlightSpec options must not contain {names}")

    def trajectory(self) -> MultiAircraftResult:
        """Solve all configured flights jointly and verify their separation."""
        center = _common_projection_center(self.flights)
        with _fleet_projection(self.flights, center):
            independent = self._independent_trajectories()
            pair_times = (
                self._candidate_times(independent)
                if self.enforce_separation and len(self.flights) > 1
                else {}
            )
            references = self._symmetry_breaking_guesses(independent, pair_times)

            final = None
            total_build = 0.0
            total_solve = 0.0
            total_verify = 0.0
            for refinement in range(self.separation.max_refinements + 1):
                build_start = time.perf_counter()
                built = self._build_problem(references, pair_times, center, refinement)
                total_build += time.perf_counter() - build_start

                solve_start = time.perf_counter()
                solution, solver_success, stats = self._solve_problem(built[0])
                total_solve += time.perf_counter() - solve_start

                extracted = self._extract_solution(built, solution)
                verify_start = time.perf_counter()
                reports, violations = self._verify(extracted)
                total_verify += time.perf_counter() - verify_start
                final = (built, solution, solver_success, stats, extracted, reports)

                separation_success = not violations
                if solver_success and separation_success:
                    break
                if not solver_success or refinement >= self.separation.max_refinements:
                    break
                for pair, times in violations.items():
                    pair_times.setdefault(pair, set()).update(times)
                solved_references = {
                    index: extracted["trajectories"][flight.id]
                    for index, flight in enumerate(self.flights)
                }
                references = self._symmetry_breaking_guesses(
                    solved_references, pair_times
                )

            assert final is not None
            built, solution, solver_success, stats, extracted, reports = final
            separation_success = (not self.enforce_separation) or all(
                (not report.overlaps)
                or report.margin >= -self.separation.verification_tolerance
                for report in reports
            )
            return self._make_result(
                built,
                solution,
                solver_success,
                separation_success,
                stats,
                extracted,
                reports,
                refinement,
                total_build,
                total_solve,
                total_verify,
            )

    def _independent_trajectories(self) -> dict[int, pd.DataFrame]:
        trajectories = {}
        for index, flight in enumerate(self.flights):
            options = dict(flight.options)
            # Fleet-time interpolation currently assumes equal interval
            # durations. Waypoint-constrained warm starts must use that same
            # uniform grid even though phase optimizers normally enable
            # variable timesteps automatically when waypoints are present.
            options["variable_timestep"] = False
            if isinstance(flight.optimizer, Cruise):
                df = flight.optimizer.trajectory(
                    objective=flight.objective,
                    initial_guess=flight.initial_guess,
                    return_failed=True,
                    **options,
                )
            else:
                # Keep all descent nodes for candidate-time detection and
                # collocation warm starts, even if the public single-phase
                # default would remove an initial level segment.
                options["remove_cruise"] = False
                df = flight.optimizer.trajectory(
                    objective=flight.objective,
                    initial_guess=flight.initial_guess,
                    **options,
                )
            if not isinstance(df, pd.DataFrame):
                raise RuntimeError(
                    f"independent warm-start solve failed for {flight.id}"
                )
            df = cast(pd.DataFrame, df)
            if df.empty:
                raise RuntimeError(
                    f"independent warm-start solve failed for {flight.id}"
                )
            trajectories[index] = df
        return trajectories

    def _candidate_times(
        self, trajectories: dict[int, pd.DataFrame]
    ) -> dict[tuple[int, int], set[float]]:
        candidate_times: dict[tuple[int, int], set[float]] = {}
        config = self.separation
        for i in range(len(self.flights)):
            for j in range(i + 1, len(self.flights)):
                first, second = trajectories[i], trajectories[j]
                start_i, start_j = (
                    self.flights[i].start_time,
                    self.flights[j].start_time,
                )
                ti = start_i + first.ts.to_numpy(dtype=float)
                tj = start_j + second.ts.to_numpy(dtype=float)
                overlap_start = max(float(ti[0]), float(tj[0]))
                overlap_end = min(float(ti[-1]), float(tj[-1]))
                if overlap_end <= overlap_start:
                    continue
                scan = np.arange(overlap_start, overlap_end, config.detect_dt_s)
                scan = np.unique(np.append(scan, overlap_end))
                xi = np.interp(scan, ti, first.x)
                yi = np.interp(scan, ti, first.y)
                hi = np.interp(scan, ti, first.h)
                xj = np.interp(scan, tj, second.x)
                yj = np.interp(scan, tj, second.y)
                hj = np.interp(scan, tj, second.h)
                metric = np.asarray(
                    separation_metric(xi - xj, yi - yj, hi - hj, config),
                    dtype=float,
                )
                near = scan[metric < config.minimum_metric * config.watch_factor**2]
                if not len(near):
                    continue
                window_start = max(
                    overlap_start, float(near.min()) - config.encounter_buffer_s
                )
                window_end = min(
                    overlap_end, float(near.max()) + config.encounter_buffer_s
                )
                times = np.arange(window_start, window_end, config.constraint_dt_s)
                times = np.unique(np.append(times, window_end))
                candidate_times[(i, j)] = {float(value) for value in times}
        return candidate_times

    def _symmetry_breaking_guesses(
        self,
        trajectories: dict[int, pd.DataFrame],
        pair_times: dict[tuple[int, int], set[float]],
    ) -> dict[int, pd.DataFrame]:
        involved = {index for pair in pair_times for index in pair}
        midpoint = (len(self.flights) - 1) / 2
        guesses = {}
        for index, flight in enumerate(self.flights):
            source = trajectories[index]
            if index not in involved or flight.initial_guess is not None:
                guesses[index] = source
                continue
            offset = index - midpoint
            fix_alt = getattr(flight.optimizer, "fix_alt", False)
            fix_track = getattr(flight.optimizer, "fix_track", False)
            if fix_alt or not fix_track:
                # Prefer a lateral symmetry break whenever lateral resolution
                # is available.  A vertical stagger can put an altitude-free
                # problem deep inside the feasible region because the
                # separation metric raises vertical distance to a high power,
                # which in turn can trap IPOPT in an over-separated local
                # solution.  The small lateral bow remains inside the conflict
                # region while providing a non-zero separation gradient.
                guesses[index] = _perturb_guess(
                    source,
                    lateral_km=offset
                    * _canonical_route_sign(source, flight.optimizer.proj),
                    altitude_ft=0.0,
                    proj=flight.optimizer.proj,
                )
            else:
                guess = source.copy()
                # Taper the vertical stagger to preserve both fixed endpoint
                # altitudes. This branch is reserved for track-fixed problems,
                # where lateral symmetry breaking is unavailable.
                progress = np.linspace(0.0, 1.0, len(source))
                guess["altitude"] = source.altitude + (
                    2_000.0 * offset * np.sin(np.pi * progress)
                )
                guesses[index] = guess
        return self._arrival_spacing_guesses(guesses)

    def _arrival_spacing_guesses(
        self, guesses: dict[int, pd.DataFrame]
    ) -> dict[int, pd.DataFrame]:
        """Retimestamp descent guesses to satisfy the requested landing order."""
        if self.minimum_arrival_gap_s is None:
            return guesses

        spaced = {}
        previous_end = None
        for index, flight in enumerate(self.flights):
            guess = guesses[index].copy()
            relative = guess.ts.to_numpy(dtype=float) - float(guess.ts.iloc[0])
            duration = float(relative[-1])
            if duration <= 0:
                raise ValueError("arrival initial guess duration must be positive")
            natural_end = flight.start_time + duration
            target_end = natural_end
            if previous_end is not None:
                target_end = max(target_end, previous_end + self.minimum_arrival_gap_s)
            target_duration = target_end - flight.start_time
            guess.loc[:, "ts"] = relative * (target_duration / duration)
            spaced[index] = guess
            previous_end = target_end
        return spaced

    def _build_problem(
        self,
        references: dict[int, pd.DataFrame],
        pair_times: dict[tuple[int, int], set[float]],
        center: tuple[float, float],
        refinement: int,
    ) -> tuple[
        ca.Opti,
        dict[int, AircraftTranscription],
        Any,
        float,
        int,
    ]:
        opti = ca.Opti()
        transcriptions = {}
        total_raw = 0
        common_altitude_state = None
        for index, flight in enumerate(self.flights):
            options = dict(flight.options)
            options.update(flight.joint_options)
            options.pop("variable_timestep", None)
            options.pop("remove_cruise", None)
            options.setdefault(
                "route_margin_m", max(50_000.0, 4 * self.separation.horizontal_m)
            )
            options["auto_rescale_objective"] = False
            transcription = flight.optimizer._add_formulation(
                opti,
                flight.objective,
                initial_guess=references[index],
                variable_timestep=False,
                name_prefix=f"aircraft_{index}_{flight.id}_r{refinement}",
                minimize=False,
                **options,
            )
            transcription.projection_center = center
            transcriptions[index] = transcription
            if self.common_altitude:
                # Every aircraft is already level through
                # fix_cruise_altitude(); equating only their initial altitude
                # states gives a shared optimized flight level without
                # redundant equalities at every collocation point.
                if common_altitude_state is None:
                    common_altitude_state = transcription.X[0][2]
                else:
                    opti.subject_to(transcription.X[0][2] == common_altitude_state)
            total_raw = total_raw + flight.weight * transcription.objective_raw

        separation_constraints = self._add_separation_constraints(
            opti, transcriptions, references, pair_times
        )
        if self.minimum_arrival_gap_s is not None:
            for index in range(len(self.flights) - 1):
                first_end = (
                    self.flights[index].start_time + transcriptions[index].ts_final
                )
                second_end = (
                    self.flights[index + 1].start_time
                    + transcriptions[index + 1].ts_final
                )
                opti.subject_to(second_end - first_end >= self.minimum_arrival_gap_s)
        x_initial = opti.debug.value(opti.x, opti.initial())
        objective_at_initial = ca.Function(
            f"fleet_objective_at_initial_{refinement}", [opti.x], [total_raw]
        )
        initial_value = float(objective_at_initial(x_initial))  # type: ignore[arg-type]
        objective_scale = abs(initial_value) if abs(initial_value) > 1e-30 else 1.0
        opti.minimize(total_raw / objective_scale)
        return (
            opti,
            transcriptions,
            total_raw,
            objective_scale,
            separation_constraints,
        )

    def _add_separation_constraints(
        self,
        opti: ca.Opti,
        transcriptions: dict[int, AircraftTranscription],
        references: dict[int, pd.DataFrame],
        pair_times: dict[tuple[int, int], set[float]],
    ) -> int:
        count = 0
        for (i, j), times in pair_times.items():
            first = transcriptions[i]
            second = transcriptions[j]
            reference_i = references[i]
            reference_j = references[j]
            duration_i = float(reference_i.ts.iloc[-1] - reference_i.ts.iloc[0])
            duration_j = float(reference_j.ts.iloc[-1] - reference_j.ts.iloc[0])
            if duration_i <= 0 or duration_j <= 0:
                continue
            for absolute_time in sorted(times):
                progress_i = (
                    (absolute_time - self.flights[i].start_time) / duration_i
                ) * first.optimizer.nodes
                progress_j = (
                    (absolute_time - self.flights[j].start_time) / duration_j
                ) * second.optimizer.nodes
                interval_i = min(
                    max(int(np.floor(progress_i)), 0), first.optimizer.nodes - 1
                )
                interval_j = min(
                    max(int(np.floor(progress_j)), 0), second.optimizer.nodes - 1
                )
                # Anchor the physical encounter time to aircraft i. This keeps
                # the expression compact while allowing aircraft j's free
                # duration to shift its progress through the encounter.
                tau_i = min(max(progress_i - interval_i, 0.0), 1.0)
                symbolic_time = (
                    self.flights[i].start_time
                    + (interval_i + tau_i) * first.ts_final / first.optimizer.nodes
                )
                tau_j = (symbolic_time - self.flights[j].start_time) / (
                    second.ts_final / second.optimizer.nodes
                ) - interval_j
                opti.subject_to(opti.bounded(0.0, tau_j, 1.0))  # type: ignore[arg-type]
                state_i = interpolate_collocation_state(
                    first.X[interval_i], first.Xc[interval_i], tau_i
                )
                state_j = interpolate_collocation_state(
                    second.X[interval_j], second.Xc[interval_j], tau_j
                )
                metric = separation_metric(
                    state_i[0] - state_j[0],
                    state_i[1] - state_j[1],
                    state_i[2] - state_j[2],
                    self.separation,
                )
                opti.subject_to(
                    metric
                    >= self.separation.minimum_metric
                    + self.separation.constraint_buffer
                )
                count += 1
        return count

    def _solver_options(self) -> dict[str, Any]:
        options = dict(self.flights[0].optimizer.solver_options)
        # The fleet Hessian grows quickly with aircraft count. L-BFGS avoids
        # the expensive exact factorization and is substantially faster for
        # the supported fuel-only formulation.
        options.setdefault("ipopt.hessian_approximation", "limited-memory")
        options["print_time"] = 1 if self.debug else 0
        options["ipopt.print_level"] = 5 if self.debug else 0
        if self.max_iter is not None:
            options["ipopt.max_iter"] = self.max_iter
        for key, value in self.ipopt_kwargs.items():
            options[f"ipopt.{key}"] = value
        if any(
            flight.optimizer.solver_options.get("ipopt.hessian_approximation")
            == "exact"
            for flight in self.flights
        ):
            options["ipopt.hessian_approximation"] = "exact"
        return options

    def _solve_problem(self, opti: ca.Opti) -> tuple[Any, bool, dict[str, Any]]:
        opti.solver("ipopt", self._solver_options())
        try:
            solution = opti.solve()
        except RuntimeError:
            solution = opti.debug
        stats = dict(solution.stats())
        return solution, bool(stats.get("success", False)), stats

    def _extract_solution(
        self,
        built: tuple[
            ca.Opti,
            dict[int, AircraftTranscription],
            Any,
            float,
            int,
        ],
        solution: Any,
    ) -> dict[str, Any]:
        _, transcriptions, total_raw, _, _ = built
        trajectories = {}
        objective_values = {}
        numeric = {}
        for index, flight in enumerate(self.flights):
            transcription = transcriptions[index]
            x_opt = np.asarray(solution.value(ca.horzcat(*transcription.X)))
            u_opt = np.asarray(solution.value(ca.horzcat(*transcription.U)))
            x_collocation = np.asarray(
                [
                    [np.asarray(solution.value(value)).ravel() for value in interval]
                    for interval in transcription.Xc
                ]
            )
            duration = float(solution.value(transcription.ts_final))
            options = dict(flight.options)
            options.update(flight.joint_options)
            df = flight.optimizer.to_trajectory(
                duration,
                x_opt,
                u_opt,
                interpolant=options.get("interpolant"),
                n_dim=options.get("n_dim"),
                time_dependent=options.get("time_dependent", False),
            )
            if isinstance(flight.optimizer, Descent) and options.get(
                "remove_cruise", False
            ):
                df = df.query("vertical_rate < -100")
            df = df.assign(absolute_ts=flight.start_time + df.ts)
            trajectories[flight.id] = df
            objective = float(solution.value(transcription.objective_raw))
            objective_values[flight.id] = objective
            flight.optimizer._last_solution = solution
            flight.optimizer.objective_value = objective
            numeric[index] = {
                "X": x_opt,
                "Xc": x_collocation,
                "duration": duration,
            }
        return {
            "trajectories": trajectories,
            "objectives": objective_values,
            "total_objective": float(solution.value(total_raw)),
            "numeric": numeric,
        }

    def _state_at(self, numeric: dict[str, Any], relative_time: float) -> np.ndarray:
        duration = numeric["duration"]
        nodes = numeric["X"].shape[1] - 1
        progress = min(max(relative_time / duration * nodes, 0.0), float(nodes))
        interval = min(int(np.floor(progress)), nodes - 1)
        tau = min(max(progress - interval, 0.0), 1.0)
        return numeric_collocation_state(
            numeric["X"][:, interval], numeric["Xc"][interval], tau
        )

    def _verify(
        self, extracted: dict[str, Any]
    ) -> tuple[tuple[PairSeparationReport, ...], dict[tuple[int, int], set[float]]]:
        reports = []
        violations: dict[tuple[int, int], set[float]] = {}
        numeric = extracted["numeric"]
        for i in range(len(self.flights)):
            for j in range(i + 1, len(self.flights)):
                start_i, start_j = (
                    self.flights[i].start_time,
                    self.flights[j].start_time,
                )
                end_i = start_i + numeric[i]["duration"]
                end_j = start_j + numeric[j]["duration"]
                overlap_start = max(start_i, start_j)
                overlap_end = min(end_i, end_j)
                if overlap_end <= overlap_start:
                    reports.append(
                        PairSeparationReport(
                            self.flights[i].id,
                            self.flights[j].id,
                            False,
                            float("inf"),
                            self.separation.minimum_metric,
                            float("inf"),
                            float("nan"),
                            float("inf"),
                            float("inf"),
                        )
                    )
                    continue
                times = np.arange(
                    overlap_start, overlap_end, self.separation.verification_dt_s
                )
                times = np.unique(np.append(times, overlap_end))
                states_i = np.asarray(
                    [self._state_at(numeric[i], value - start_i) for value in times]
                )
                states_j = np.asarray(
                    [self._state_at(numeric[j], value - start_j) for value in times]
                )
                dx = states_i[:, 0] - states_j[:, 0]
                dy = states_i[:, 1] - states_j[:, 1]
                dz = states_i[:, 2] - states_j[:, 2]
                metrics = np.asarray(
                    separation_metric(dx, dy, dz, self.separation), dtype=float
                )
                worst = int(np.argmin(metrics))
                minimum = float(metrics[worst])
                reports.append(
                    PairSeparationReport(
                        self.flights[i].id,
                        self.flights[j].id,
                        True,
                        minimum,
                        self.separation.minimum_metric,
                        minimum - self.separation.minimum_metric,
                        float(times[worst]),
                        float(np.hypot(dx[worst], dy[worst])),
                        float(abs(dz[worst])),
                    )
                )
                failed = metrics < (
                    self.separation.minimum_metric
                    - self.separation.verification_tolerance
                )
                if self.enforce_separation and np.any(failed):
                    violations[(i, j)] = {float(value) for value in times[failed]}
        return tuple(reports), violations

    def _make_result(
        self,
        built: tuple[
            ca.Opti,
            dict[int, AircraftTranscription],
            Any,
            float,
            int,
        ],
        solution: Any,
        solver_success: bool,
        separation_success: bool,
        stats: dict[str, Any],
        extracted: dict[str, Any],
        reports: tuple[PairSeparationReport, ...],
        refinement: int,
        build_time_s: float,
        solve_time_s: float,
        verification_time_s: float,
    ) -> MultiAircraftResult:
        del solution
        status = str(stats.get("return_status", ""))
        if solver_success and not separation_success:
            status = "Separation_Verification_Failed"
        return MultiAircraftResult(
            trajectories=extracted["trajectories"],
            success=solver_success and separation_success,
            solver_success=solver_success,
            separation_success=separation_success,
            status=status,
            objective=extracted["total_objective"],
            aircraft_objectives=extracted["objectives"],
            weights={flight.id: flight.weight for flight in self.flights},
            pair_reports=reports,
            stats=stats,
            nlp_variables=int(built[0].nx),
            nlp_constraints=int(built[0].ng),
            separation_constraints=built[4],
            refinement_rounds=refinement,
            build_time_s=build_time_s,
            solve_time_s=solve_time_s,
            verification_time_s=verification_time_s,
        )


__all__ = [
    "FlightSpec",
    "MultiAircraft",
    "MultiAircraftResult",
    "PairSeparationReport",
    "SeparationConfig",
]
