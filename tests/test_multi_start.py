"""Tests for the multi-start trajectory wrapper and its helpers."""

import math

import numpy as np
import pandas as pd
import pytest

import opentop as top
from opentop._multi_start import _perturb_guess


def _make_canonical_df(n=10, origin=(52.362, 13.501), dest=(40.472, -3.563),
                       cruise_alt_ft=35000):
    """Build a synthetic canonical trajectory DataFrame (great-circle,
    constant cruise altitude) for testing _perturb_guess in isolation."""
    lats = np.linspace(origin[0], dest[0], n)
    lons = np.linspace(origin[1], dest[1], n)
    alts = np.full(n, cruise_alt_ft, dtype=float)
    mass = np.linspace(67000, 62000, n)
    ts = np.linspace(0, 8000, n)
    return pd.DataFrame({
        "latitude": lats,
        "longitude": lons,
        "altitude": alts,
        "mass": mass,
        "ts": ts,
    })


@pytest.fixture(scope="module")
def proj():
    """A real projection from a Cruise optimizer — multi_start perturbation
    will use the same one in production."""
    opt = top.Cruise(
        "A320",
        (52.362, 13.501),
        (40.472, -3.563),
        m0=0.85,
    )
    return opt.proj


class TestPerturbGuessAltitude:
    def test_altitude_only_shifts_altitude_column(self, proj):
        df = _make_canonical_df()
        perturbed = _perturb_guess(df, lateral_km=0.0, altitude_ft=1500.0, proj=proj)
        np.testing.assert_allclose(
            perturbed["altitude"].values, df["altitude"].values + 1500.0  # type: ignore[operator]  # pandas .values type is opaque to numpy stubs
        )

    def test_altitude_zero_leaves_altitude_unchanged(self, proj):
        df = _make_canonical_df()
        perturbed = _perturb_guess(df, lateral_km=0.0, altitude_ft=0.0, proj=proj)
        np.testing.assert_array_equal(
            perturbed["altitude"].values, df["altitude"].values
        )

    def test_altitude_negative_shifts_down(self, proj):
        df = _make_canonical_df()
        perturbed = _perturb_guess(df, lateral_km=0.0, altitude_ft=-2000.0, proj=proj)
        np.testing.assert_allclose(
            perturbed["altitude"].values, df["altitude"].values - 2000.0  # type: ignore[operator]  # pandas .values type is opaque to numpy stubs
        )

    def test_perturb_does_not_mutate_input(self, proj):
        df = _make_canonical_df()
        df_copy = df.copy()
        _ = _perturb_guess(df, lateral_km=50.0, altitude_ft=1000.0, proj=proj)
        pd.testing.assert_frame_equal(df, df_copy)

    def test_perturb_preserves_other_columns(self, proj):
        df = _make_canonical_df()
        perturbed = _perturb_guess(df, lateral_km=0.0, altitude_ft=1000.0, proj=proj)
        for col in ("mass", "ts", "latitude", "longitude"):
            np.testing.assert_array_equal(perturbed[col].values, df[col].values)


class TestPerturbGuessLateral:
    def test_endpoints_unchanged_for_any_lateral(self, proj):
        df = _make_canonical_df()
        for lat_km in (-150.0, -50.0, 0.0, 50.0, 150.0):
            perturbed = _perturb_guess(df, lateral_km=lat_km, altitude_ft=0.0, proj=proj)
            assert abs(perturbed["latitude"].iloc[0] - df["latitude"].iloc[0]) < 1e-6
            assert abs(perturbed["longitude"].iloc[0] - df["longitude"].iloc[0]) < 1e-6
            assert abs(perturbed["latitude"].iloc[-1] - df["latitude"].iloc[-1]) < 1e-6
            assert abs(perturbed["longitude"].iloc[-1] - df["longitude"].iloc[-1]) < 1e-6

    def test_lateral_zero_leaves_path_unchanged(self, proj):
        df = _make_canonical_df()
        perturbed = _perturb_guess(df, lateral_km=0.0, altitude_ft=0.0, proj=proj)
        np.testing.assert_allclose(perturbed["latitude"].values, df["latitude"].values,  # type: ignore[arg-type]  # pandas .values type is opaque to numpy stubs
                                   atol=1e-9)
        np.testing.assert_allclose(perturbed["longitude"].values, df["longitude"].values,  # type: ignore[arg-type]  # pandas .values type is opaque to numpy stubs
                                   atol=1e-9)

    def test_lateral_sign_flips_direction(self, proj):
        df = _make_canonical_df()
        pos = _perturb_guess(df, lateral_km=100.0, altitude_ft=0.0, proj=proj)
        neg = _perturb_guess(df, lateral_km=-100.0, altitude_ft=0.0, proj=proj)
        mid = len(df) // 2
        canonical_mid_lat = df["latitude"].iloc[mid]
        pos_mid_lat = pos["latitude"].iloc[mid]
        neg_mid_lat = neg["latitude"].iloc[mid]
        assert (pos_mid_lat - canonical_mid_lat) * (neg_mid_lat - canonical_mid_lat) < 0

    def test_lateral_peak_near_midpoint(self, proj):
        """The deflection should be largest at the midpoint (sin(pi/2) = 1)."""
        df = _make_canonical_df(n=21)       # odd n → exact midpoint
        perturbed = _perturb_guess(df, lateral_km=100.0, altitude_ft=0.0, proj=proj)
        xp, yp = proj(df["longitude"].values, df["latitude"].values)
        xp_p, yp_p = proj(perturbed["longitude"].values, perturbed["latitude"].values)
        deflection = np.hypot(xp_p - xp, yp_p - yp)
        mid = len(df) // 2
        assert deflection[mid] > deflection[0] + 1e3
        assert deflection[mid] > deflection[-1] + 1e3
        assert deflection[0] < 1.0
        assert deflection[-1] < 1.0
        np.testing.assert_allclose(deflection[mid], 100_000.0, rtol=0.01)

    def test_lateral_deterministic(self, proj):
        df = _make_canonical_df()
        a = _perturb_guess(df, lateral_km=75.0, altitude_ft=0.0, proj=proj)
        b = _perturb_guess(df, lateral_km=75.0, altitude_ft=0.0, proj=proj)
        pd.testing.assert_frame_equal(a, b)


@pytest.fixture(scope="module")
def _fast_optimizer():
    """A minimal Cruise optimizer used across multi-start integration tests.
    Short route and coarse grid to keep wall time under a minute."""
    opt = top.Cruise(
        "A320",
        (52.362, 13.501),
        (52.560, 13.287),   # short intra-Berlin hop; solver converges fast
        m0=0.85,
    )
    opt.setup(max_iter=500)
    return opt


class TestMultiStartGuards:
    def test_n_starts_zero_raises(self, _fast_optimizer):
        with pytest.raises(ValueError, match="n_starts"):
            _fast_optimizer.multi_start_trajectory(objective="fuel", n_starts=0)

    def test_n_starts_negative_raises(self, _fast_optimizer):
        with pytest.raises(ValueError, match="n_starts"):
            _fast_optimizer.multi_start_trajectory(objective="fuel", n_starts=-3)


class TestMultiStartSingle:
    def test_n_starts_one_returns_tuple(self, _fast_optimizer):
        trajectory, candidates = _fast_optimizer.multi_start_trajectory(
            objective="fuel", n_starts=1
        )
        assert isinstance(trajectory, pd.DataFrame)
        assert isinstance(candidates, list)
        assert len(candidates) == 1

    def test_n_starts_one_matches_plain_trajectory(self, _fast_optimizer):
        """With n_starts=1 the returned trajectory must equal what a plain
        trajectory() call would produce (no perturbation, same seed
        independence)."""
        ms_traj, _ = _fast_optimizer.multi_start_trajectory(
            objective="fuel", n_starts=1
        )
        plain_traj = _fast_optimizer.trajectory(objective="fuel")
        # Compare on a subset of columns that are directly comparable
        # (mass, altitude from the mass trajectory).
        np.testing.assert_allclose(
            ms_traj["mass"].values, plain_traj["mass"].values, rtol=1e-6
        )

    def test_candidate_zero_has_expected_fields(self, _fast_optimizer):
        trajectory, candidates = _fast_optimizer.multi_start_trajectory(
            objective="fuel", n_starts=1
        )
        c = candidates[0]
        for field in ("start_index", "objective", "fuel", "grid_cost",
                      "success", "return_status", "iters", "perturbation",
                      "wall_time_s", "trajectory"):
            assert field in c, f"missing field: {field}"
        assert c["start_index"] == 0
        assert c["perturbation"] == {"lateral_km": 0.0, "altitude_ft": 0.0}
        assert isinstance(c["trajectory"], pd.DataFrame)
        assert isinstance(c["return_status"], str)
        # The returned `trajectory` must be the same object stored in the
        # winning candidate's trajectory field.
        assert trajectory is c["trajectory"]

    def test_n_starts_one_forwards_initial_guess_to_trajectory(self, _fast_optimizer):
        """The n_starts=1 fast path must forward `initial_guess=` through to
        the underlying trajectory() call. Verify by spying on trajectory."""
        # First, get a real trajectory DataFrame to use as initial_guess.
        baseline = _fast_optimizer.trajectory(objective="fuel")

        captured = {}
        original_trajectory = _fast_optimizer.trajectory

        def spy(*args, **kwargs):
            captured.update(kwargs)
            return original_trajectory(*args, **kwargs)

        _fast_optimizer.trajectory = spy
        try:
            _fast_optimizer.multi_start_trajectory(
                objective="fuel", n_starts=1, initial_guess=baseline,
            )
        finally:
            _fast_optimizer.trajectory = original_trajectory

        assert "initial_guess" in captured, "initial_guess kwarg was dropped"
        assert captured["initial_guess"] is baseline, (
            "initial_guess was forwarded but not as the same object"
        )


class TestMultiStartLoop:
    def test_n_starts_two_produces_two_candidates(self, _fast_optimizer):
        _, candidates = _fast_optimizer.multi_start_trajectory(
            objective="fuel", n_starts=2, seed=0
        )
        assert len(candidates) == 2
        # Start indices should be consecutive 0, 1 after sort (ranking is by
        # success/objective, not by start index). But every candidate must
        # exist with its start_index intact.
        indices = sorted(c["start_index"] for c in candidates)
        assert indices == [0, 1]

    def test_perturbation_drawn_within_jitter_range(self, _fast_optimizer):
        _, candidates = _fast_optimizer.multi_start_trajectory(
            objective="fuel",
            n_starts=4,
            lateral_jitter_km=50.0,
            altitude_jitter_ft=1500.0,
            seed=0,
        )
        # Start 0 perturbation must be zero.
        start0 = next(c for c in candidates if c["start_index"] == 0)
        assert start0["perturbation"]["lateral_km"] == 0.0
        assert start0["perturbation"]["altitude_ft"] == 0.0
        # Starts 1..N-1 must be within +/- the jitter bounds.
        for c in candidates:
            if c["start_index"] == 0:
                continue
            assert abs(c["perturbation"]["lateral_km"]) <= 50.0
            assert abs(c["perturbation"]["altitude_ft"]) <= 1500.0

    def test_seed_is_reproducible(self, _fast_optimizer):
        _, cand_a = _fast_optimizer.multi_start_trajectory(
            objective="fuel", n_starts=3, seed=42
        )
        _, cand_b = _fast_optimizer.multi_start_trajectory(
            objective="fuel", n_starts=3, seed=42
        )
        pert_a = {c["start_index"]: c["perturbation"] for c in cand_a}
        pert_b = {c["start_index"]: c["perturbation"] for c in cand_b}
        assert pert_a == pert_b

    def test_different_seeds_produce_different_perturbations(self, _fast_optimizer):
        _, cand_a = _fast_optimizer.multi_start_trajectory(
            objective="fuel", n_starts=3, seed=1
        )
        _, cand_b = _fast_optimizer.multi_start_trajectory(
            objective="fuel", n_starts=3, seed=2
        )
        pert_a = {c["start_index"]: c["perturbation"] for c in cand_a}
        pert_b = {c["start_index"]: c["perturbation"] for c in cand_b}
        assert any(pert_a[i] != pert_b[i] for i in (1, 2))

    def test_zero_jitter_makes_all_perturbations_zero(self, _fast_optimizer):
        _, candidates = _fast_optimizer.multi_start_trajectory(
            objective="fuel",
            n_starts=3,
            lateral_jitter_km=0.0,
            altitude_jitter_ft=0.0,
            seed=0,
        )
        for c in candidates:
            assert c["perturbation"]["lateral_km"] == 0.0
            assert c["perturbation"]["altitude_ft"] == 0.0

    def test_winner_is_first_in_list(self, _fast_optimizer):
        trajectory, candidates = _fast_optimizer.multi_start_trajectory(
            objective="fuel", n_starts=3, seed=0
        )
        assert trajectory is candidates[0]["trajectory"]

    def test_ranking_feasibility_first_then_objective(self):
        """Synthetic records to test the sort order independently of the solver."""
        from opentop._multi_start import _rank_candidates
        records = [
            {"start_index": 0, "success": False, "objective": 100.0,
             "trajectory": "df_a"},
            {"start_index": 1, "success": True,  "objective": 200.0,
             "trajectory": "df_b"},
            {"start_index": 2, "success": False, "objective": 50.0,
             "trajectory": "df_c"},
            {"start_index": 3, "success": True,  "objective": 150.0,
             "trajectory": "df_d"},
        ]
        ranked = _rank_candidates(records)
        indices = [c["start_index"] for c in ranked]
        # Expect: feasible runs first (in objective order: 150, 200),
        # then stalled (in objective order: 50, 100).
        assert indices == [3, 1, 2, 0]


class TestMultiStartWithInterpolant:
    def test_grid_cost_populated_when_interpolant_given(self):
        """With an interpolant passed, every candidate's grid_cost field is
        a float (not NaN), and the returned trajectory has a grid_cost
        column."""
        opt = top.Cruise(
            "A320",
            (52.362, 13.501),
            (52.560, 13.287),
            m0=0.85,
        )
        opt.setup(max_iter=500)

        # Build a trivial 3D interpolant (all zeros) to exercise the
        # interpolant code path without affecting the optimum.
        lons = np.linspace(10, 18, 5)
        lats = np.linspace(50, 56, 5)
        heights = np.linspace(5000, 12000, 4)
        lon_g, lat_g, h_g = np.meshgrid(lons, lats, heights, indexing="ij")
        df_cost = pd.DataFrame({
            "longitude": lon_g.ravel(),
            "latitude": lat_g.ravel(),
            "height": h_g.ravel(),
            "cost": np.zeros(lon_g.size),
        })
        interp = top.tools.interpolant_from_dataframe(df_cost, shape="bspline")

        trajectory, candidates = opt.multi_start_trajectory(
            objective="fuel",
            interpolant=interp,
            n_starts=2,
            seed=0,
        )
        assert "grid_cost" in trajectory.columns
        for c in candidates:
            assert isinstance(c["grid_cost"], float)
            assert not math.isnan(c["grid_cost"])   # populated, not NaN

    def test_grid_cost_nan_when_no_interpolant(self, _fast_optimizer):
        """Without an interpolant, grid_cost in each candidate is NaN
        (consistent with the DataFrame's grid_cost column, which is all-NaN
        without an interpolant)."""
        _, candidates = _fast_optimizer.multi_start_trajectory(
            objective="fuel", n_starts=2, seed=0
        )
        for c in candidates:
            assert isinstance(c["grid_cost"], float)
            assert math.isnan(c["grid_cost"])


class TestMultiStartResultObject:
    def test_multi_start_with_result_object_kwarg_does_not_crash(self, _fast_optimizer):
        """result_object=True must be silently dropped; multi-start always returns DataFrames."""
        trajectory, candidates = _fast_optimizer.multi_start_trajectory(
            objective="fuel", n_starts=1, result_object=True
        )
        assert isinstance(trajectory, pd.DataFrame)
        for c in candidates:
            assert isinstance(c["trajectory"], pd.DataFrame)
