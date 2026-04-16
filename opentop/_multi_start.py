"""Multi-start trajectory optimization.

Pure functions operating on an optimizer instance passed as argument.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Union

import numpy as np
import pandas as pd

from ._types import ObjectiveFn


def _perturb_guess(
    df: pd.DataFrame,
    lateral_km: float,
    altitude_ft: float,
    proj: Any,
) -> pd.DataFrame:
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
    if lateral_km != 0.0:  # exact-zero fast path; any non-zero value
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


def _rank_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sort candidates best-first: feasibility before objective.

    Uses a stable sort. Ties preserve start-index order (insertion order).

    Args:
        candidates: list of candidate dicts with 'success' (bool) and
            'objective' (float) keys.

    Returns:
        list: new list sorted in best-first order.
    """
    return sorted(candidates, key=lambda c: (not c["success"], c["objective"]))


def run_multi_start(
    optimizer: Any,
    objective: Union[str, Callable[..., Any]] = "fuel",
    **kwargs: Any,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Run N solves from different initial guesses and return the best.

    Takes the optimizer instance as explicit arg; no ``self`` coupling.

    Start 0 uses the user's ``initial_guess=`` (if given) or the
    optimizer's default great-circle guess. Starts 1..N-1 apply random
    lateral and altitude perturbations to a canonical basis DataFrame:
    if ``initial_guess=`` was supplied, that DataFrame is the basis;
    otherwise start 0's solved trajectory is the basis.

    Args:
        optimizer: the ``Base`` (or subclass) instance driving the solves.
        objective: Optimization objective, forwarded to ``trajectory()``.
        **kwargs: Multi-start controls (``n_starts``, ``lateral_jitter_km``,
            ``altitude_jitter_ft``, ``seed``) plus any extra keyword arguments
            forwarded to ``trajectory()``.

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
    n_starts = kwargs.pop("n_starts", 5)
    lateral_jitter_km = kwargs.pop("lateral_jitter_km", 100.0)
    altitude_jitter_ft = kwargs.pop("altitude_jitter_ft", 3000.0)
    seed = kwargs.pop("seed", None)
    trajectory_kwargs = dict(kwargs)
    # multi_start always returns DataFrames; strip the result_object flag if passed.
    trajectory_kwargs.pop("result_object", None)
    trajectory_kwargs["objective"] = objective

    if n_starts < 1:
        raise ValueError(f"n_starts must be >= 1, got {n_starts}")
    if seed is not None and (not isinstance(seed, int) or seed < 0):
        raise ValueError(f"seed must be a non-negative integer or None, got {seed!r}")

    has_interpolant = trajectory_kwargs.get("interpolant") is not None

    def _make_candidate(index, df, lat_km, alt_ft, wall_time_s):
        stats = (
            optimizer._last_solution.stats()
            if hasattr(optimizer, "_last_solution")
            else {}
        )
        grid = (
            float(df["grid_cost"].sum(skipna=True)) if has_interpolant else float("nan")
        )
        return {
            "start_index": index,
            "objective": float(getattr(optimizer, "objective_value", float("nan"))),
            "fuel": float(df["mass"].iloc[0] - df["mass"].iloc[-1]),
            "grid_cost": grid,
            "success": bool(stats.get("success")),
            "return_status": str(stats.get("return_status", "")),
            "iters": int(stats.get("iter_count", 0)),
            "perturbation": {"lateral_km": float(lat_km), "altitude_ft": float(alt_ft)},
            "wall_time_s": float(wall_time_s),
            "trajectory": df,
        }

    candidates = []

    # Start 0: canonical (no perturbation).
    t0 = time.perf_counter()
    df_0 = optimizer.trajectory(**trajectory_kwargs)
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
                canonical_df, lat_km, alt_ft, proj=optimizer.proj
            )
            ts = time.perf_counter()
            df_i = optimizer.trajectory(
                **{**trajectory_kwargs, "initial_guess": perturbed}
            )
            wall = time.perf_counter() - ts
            candidates.append(_make_candidate(i, df_i, lat_km, alt_ft, wall))

    candidates = _rank_candidates(candidates)
    return candidates[0]["trajectory"], candidates
