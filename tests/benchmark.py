#!/usr/bin/env python
"""Benchmark script for openap-top.

Generates a text report of trajectory optimization performance across
standard and grid-cost test cases. Supports benchmarking any released
version via uv's isolated environments.

Usage:
    python tests/benchmark.py                    # Benchmark HEAD (local)
    python tests/benchmark.py --version 2.0.0    # Benchmark PyPI version
    python tests/benchmark.py --output report.txt

Grid cost tests require tests/tmp/contrail.nc. If absent, the grid cost
section is skipped.
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Version compatibility shims
# ---------------------------------------------------------------------------


def _obj_value(opt):
    """Get objective value (v1: solution['f'], v2: objective_value)."""
    val = getattr(opt, "objective_value", None)
    if val is not None:
        return float(val)
    return float(opt.solution["f"])


def _setup(opt, **kwargs):
    """Call setup() with version-compatible iteration kwarg."""
    try:
        opt.setup(**kwargs)
    except TypeError:
        if "max_iter" in kwargs:
            kwargs["max_iterations"] = kwargs.pop("max_iter")
        opt.setup(**kwargs)


def _solver_stats(opt):
    """Get (success, iterations) tuple, handling missing attributes."""
    try:
        stats = opt.solver.stats()
        return bool(stats.get("success", False)), int(stats.get("iter_count", 0))
    except AttributeError:
        return None, None


def _import_top():
    """Import the trajectory optimizer module.

    Returns the v2+ ``opentop`` top-level package when available, falling
    back to the v1 ``openap.top`` namespace sub-package for older releases
    benchmarked via ``uv run --with openap-top==X.Y.Z``.
    """
    try:
        import opentop

        return opentop
    except ImportError:
        pass
    # v1.x fallback: openap.top lives under the openap namespace
    try:
        from openap import top

        return top
    except ImportError:
        import openap

        openap.__path__.insert(0, str(REPO_ROOT / "openap"))
        from openap import top

        return top


# ---------------------------------------------------------------------------
# Benchmark cases
# ---------------------------------------------------------------------------


def _run_standard_case(label, factory, objective, **traj_kwargs):
    """Run a single standard trajectory benchmark."""
    t0 = time.time()
    try:
        opt = factory()
        df = opt.trajectory(objective=objective, **traj_kwargs)
        elapsed = time.time() - t0

        success, iterations = _solver_stats(opt)
        try:
            obj = _obj_value(opt)
        except (KeyError, AttributeError):
            obj = None

        return {
            "label": label,
            "success": success,
            "iterations": iterations,
            "elapsed": round(elapsed, 3),
            "objective": obj,
            "fuel": round(float(df.mass.iloc[0] - df.mass.iloc[-1]), 2),
            "time": round(float(df.ts.iloc[-1]), 1),
            "alt_max": round(float(df.altitude.max()), 0),
            "n_points": int(len(df)),
        }
    except Exception as e:
        return {
            "label": label,
            "success": False,
            "error": str(e)[:200],
            "elapsed": round(time.time() - t0, 3),
        }


def run_benchmarks():
    """Run all benchmarks and return results dict."""
    import platform
    import warnings

    import casadi as ca
    import openap

    warnings.filterwarnings("ignore")

    top = _import_top()

    results = {"metadata": {}, "standard": [], "grid_cost": []}

    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, cwd=REPO_ROOT
        ).strip()[:10]
    except Exception:
        sha = "unknown"

    results["metadata"] = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "git_sha": sha,
        "python": platform.python_version(),
        "casadi": ca.__version__,
        "openap": getattr(openap, "__version__", "unknown"),
    }

    cases = [
        ("cruise_short_fuel", lambda: top.Cruise("A320", "EHAM", "EDDF", 0.85), "fuel"),
        (
            "cruise_medium_fuel",
            lambda: top.Cruise("A320", "EHAM", "LGAV", 0.85),
            "fuel",
        ),
        ("cruise_short_time", lambda: top.Cruise("A320", "EHAM", "EDDF", 0.85), "time"),
        (
            "cruise_medium_ci50",
            lambda: top.Cruise("A320", "EHAM", "LGAV", 0.85),
            "ci:50",
        ),
        (
            "cruise_medium_gwp100",
            lambda: top.Cruise("A320", "EHAM", "LGAV", 0.85),
            "gwp100",
        ),
        (
            "cruise_medium_gtp100",
            lambda: top.Cruise("A320", "EHAM", "LGAV", 0.85),
            "gtp100",
        ),
        (
            "complete_short_fuel",
            lambda: top.CompleteFlight("A320", "EHAM", "EDDF", 0.85),
            "fuel",
        ),
        (
            "complete_medium_fuel",
            lambda: top.CompleteFlight("A320", "EHAM", "LGAV", 0.85),
            "fuel",
        ),
    ]

    for label, factory, objective in cases:
        print(f"  {label}...", file=sys.stderr, flush=True)
        results["standard"].append(_run_standard_case(label, factory, objective))

    contrail_nc = REPO_ROOT / "tests" / "tmp" / "contrail.nc"
    if contrail_nc.exists():
        print("  grid cost cases...", file=sys.stderr, flush=True)
        try:
            results["grid_cost"] = _run_grid_cost_cases(contrail_nc, top)
        except Exception as e:
            results["grid_cost_error"] = str(e)[:500]
    else:
        results["grid_cost_skipped"] = f"{contrail_nc.name} not found"

    return results


def _run_grid_cost_cases(contrail_nc, top):
    """Run contrail+CO2 grid cost benchmarks."""
    import openap
    import openap.casadi as oc
    import pandas as pd
    import xarray as xr
    from scipy.ndimage import gaussian_filter

    ds = xr.open_dataset(str(contrail_nc)).sel(time="2015-12-18")
    level_pressure = [
        0.0,
        10.0,
        30.0,
        50.0,
        70.0,
        90.0787,
        110.6606,
        132.3968,
        155.7909,
        181.1544,
        208.6494,
        238.3258,
        270.1530,
        304.0465,
        339.8891,
        377.5467,
        416.8789,
        457.7442,
        500.0,
        543.4970,
        588.0685,
        633.5144,
        679.5799,
        725.9285,
        772.1102,
        817.5241,
        861.3757,
        902.6287,
        939.9520,
        971.6610,
        995.6532,
        1009.3396,
    ]
    df = (
        ds.to_dataframe()
        .reset_index()
        .assign(lev=lambda x: x.lev.astype(int))
        .merge(
            pd.DataFrame(level_pressure, columns=["hPa"]).reset_index(names="lev"),
            on="lev",
        )
        .assign(height=lambda x: openap.aero.h_isa(x.hPa * 100).round(-2))
        .assign(longitude=lambda x: (x.lon + 180) % 360 - 180)
        .query("height<15000")
    )
    df_cost = (
        df.rename(columns={"lat": "latitude", "atr20_contrail": "cost"})[
            ["time", "latitude", "longitude", "hPa", "height", "cost"]
        ]
        .query("-20<longitude<40 and 30<latitude<70 and time.dt.hour==12")
        .sort_values(["height", "latitude", "longitude"])
    )
    cost = df_cost.cost.values.reshape(
        df_cost.height.nunique(),
        df_cost.latitude.nunique(),
        df_cost.longitude.nunique(),
    )
    cost_ = gaussian_filter(cost, sigma=1, mode="nearest")
    df_cost = df_cost.assign(cost=cost_.flatten())
    interpolant = top.tools.interpolant_from_dataframe(df_cost)

    def make_obj(optimizer):
        def objective(x, u, dt, **kwargs):
            vtas = oc.aero.mach2tas(u[0], x[2])
            kwargs.pop("n_dim", None)
            kwargs.pop("time_dependent", None)
            contrail_cost = (
                optimizer.obj_grid_cost(
                    x, u, dt, n_dim=3, time_dependent=False, **kwargs
                )
                * vtas
                * 1e-3
            )
            co2_cost = optimizer.obj_fuel(x, u, dt, **kwargs) * 7.03e-15
            return contrail_cost + co2_cost

        return objective

    cases = [
        ("cruise_contrail", top.Cruise),
        ("complete_contrail", top.CompleteFlight),
    ]

    results = []
    for label, cls in cases:
        print(f"    {label}...", file=sys.stderr, flush=True)
        t0 = time.time()
        try:
            opt = cls("A320", "EHAM", "LGAV", 0.85)
            _setup(opt, max_iter=2000)
            df_fuel = opt.trajectory(objective="fuel")
            df = opt.trajectory(
                objective=make_obj(opt),
                interpolant=interpolant,
                n_dim=3,
                initial_guess=df_fuel,
                return_failed=True,
            )
            elapsed = time.time() - t0
            success, iterations = _solver_stats(opt)
            try:
                obj = _obj_value(opt)
            except (KeyError, AttributeError):
                obj = None

            results.append(
                {
                    "label": label,
                    "success": success,
                    "iterations": iterations,
                    "elapsed": round(elapsed, 3),
                    "objective": obj,
                    "fuel": round(float(df.mass.iloc[0] - df.mass.iloc[-1]), 2),
                    "time": round(float(df.ts.iloc[-1]), 1),
                    "alt_max": round(float(df.altitude.max()), 0),
                    "grid_cost_sum": float(df.grid_cost.sum()),
                    "fuel_cost_sum": round(float(df.fuel_cost.sum()), 1),
                }
            )
        except Exception as e:
            results.append(
                {
                    "label": label,
                    "success": False,
                    "error": str(e)[:200],
                    "elapsed": round(time.time() - t0, 3),
                }
            )

    return results


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------


def _fmt(value, spec, missing="-"):
    """Format a value with the given spec, or return missing marker."""
    if value is None:
        return missing
    return format(value, spec)


def format_report(results, version_label):
    """Format benchmark results as a plain-text report."""
    lines = []
    lines.append("=" * 80)
    lines.append("  openap-top Benchmark Report")
    lines.append("=" * 80)
    lines.append("")

    md = results.get("metadata", {})
    lines.append(f"Version:      {version_label}")
    lines.append(f"Git SHA:      {md.get('git_sha', 'unknown')}")
    lines.append(f"Date:         {md.get('date', 'unknown')}")
    lines.append(f"Python:       {md.get('python', 'unknown')}")
    lines.append(f"CasADi:       {md.get('casadi', 'unknown')}")
    lines.append(f"OpenAP:       {md.get('openap', 'unknown')}")
    lines.append("")

    # Trajectory metrics
    lines.append("-" * 80)
    lines.append("  Standard Trajectory Metrics")
    lines.append("-" * 80)
    lines.append(
        f"{'Test Case':<26}{'Fuel (kg)':>12}{'Time (s)':>11}"
        f"{'Alt Max':>11}{'Points':>9}"
    )
    lines.append("-" * 80)
    for r in results.get("standard", []):
        if "error" in r:
            lines.append(f"{r['label']:<26}  ERROR: {r['error']}")
            continue
        lines.append(
            f"{r['label']:<26}"
            f"{_fmt(r.get('fuel'), '>12.2f')}"
            f"{_fmt(r.get('time'), '>11.1f')}"
            f"{_fmt(r.get('alt_max'), '>11.0f')}"
            f"{_fmt(r.get('n_points'), '>9d')}"
        )
    lines.append("")

    # Solver performance
    lines.append("-" * 80)
    lines.append("  Solver Performance")
    lines.append("-" * 80)
    lines.append(
        f"{'Test Case':<26}{'Status':>10}{'Iter':>7}"
        f"{'Elapsed (s)':>14}{'Objective':>20}"
    )
    lines.append("-" * 80)
    for r in results.get("standard", []):
        if "error" in r:
            continue
        if r.get("success") is None:
            status = "-"
        elif r["success"]:
            status = "success"
        else:
            status = "FAILED"
        obj = r.get("objective")
        obj_str = f"{obj:.4f}" if obj is not None else "-"
        lines.append(
            f"{r['label']:<26}"
            f"{status:>10}"
            f"{_fmt(r.get('iterations'), '>7d')}"
            f"{_fmt(r.get('elapsed'), '>14.3f')}"
            f"{obj_str:>20}"
        )
    lines.append("")

    # Grid cost
    gc = results.get("grid_cost", [])
    if gc:
        lines.append("-" * 80)
        lines.append("  Grid Cost (Contrail + CO2)")
        lines.append("-" * 80)
        lines.append(
            f"{'Test Case':<22}{'Fuel (kg)':>12}{'Time (s)':>11}"
            f"{'Alt Max':>11}{'Grid Cost':>14}{'Elapsed':>10}"
        )
        lines.append("-" * 80)
        for r in gc:
            if "error" in r:
                lines.append(f"{r['label']:<22}  ERROR: {r['error']}")
                continue
            gc_sum = r.get("grid_cost_sum")
            gc_str = f"{gc_sum:.4e}" if gc_sum is not None else "-"
            lines.append(
                f"{r['label']:<22}"
                f"{_fmt(r.get('fuel'), '>12.2f')}"
                f"{_fmt(r.get('time'), '>11.1f')}"
                f"{_fmt(r.get('alt_max'), '>11.0f')}"
                f"{gc_str:>14}"
                f"{_fmt(r.get('elapsed'), '>10.3f')}"
            )
        lines.append("")
    elif "grid_cost_skipped" in results:
        lines.append("-" * 80)
        lines.append(f"  Grid Cost: skipped ({results['grid_cost_skipped']})")
        lines.append("")
    elif "grid_cost_error" in results:
        lines.append("-" * 80)
        lines.append(f"  Grid Cost: error ({results['grid_cost_error']})")
        lines.append("")

    lines.append("=" * 80)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Isolated version runner
# ---------------------------------------------------------------------------


def run_with_uv(version):
    """Run benchmark in an isolated uv env with pip-installed version.

    Args:
        version: PyPI version string. Accepts "2.0.0" or "v2.0.0".

    Returns:
        Results dict parsed from the subprocess JSON output.
    """
    pypi_version = version.lstrip("v")
    # v1.x shipped as the PyPI package `openap-top` (imported as openap.top);
    # v2.0.0+ ships as the top-level `opentop` package.
    pypi_name = "opentop" if int(pypi_version.split(".")[0]) >= 2 else "openap-top"
    cmd = [
        "uv",
        "run",
        "--no-project",
        "--with",
        f"{pypi_name}=={pypi_version}",
        "python",
        str(Path(__file__).resolve()),
        "--run-only",
    ]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=3600,
    )
    if result.stderr:
        sys.stderr.write(result.stderr)
    if result.returncode != 0:
        raise RuntimeError(f"Benchmark failed for version {version}")
    lines = [ln for ln in result.stdout.strip().split("\n") if ln.strip()]
    return json.loads(lines[-1])


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Benchmark openap-top")
    parser.add_argument(
        "--version",
        default="HEAD",
        help="PyPI version to benchmark (e.g. 2.0.0), or HEAD for local dev",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output file path. Default: tests/benchmarks/<version>.txt",
    )
    parser.add_argument(
        "--run-only",
        action="store_true",
        help="Run benchmarks and print JSON (internal use by version runner)",
    )
    args = parser.parse_args()

    if args.run_only:
        results = run_benchmarks()
        print(json.dumps(results))
        return

    if args.version == "HEAD":
        print("Benchmarking HEAD (local dev)...", file=sys.stderr)
        results = run_benchmarks()
        version_label = "HEAD"
    else:
        print(
            f"Benchmarking openap-top=={args.version} (isolated uv env)...",
            file=sys.stderr,
        )
        results = run_with_uv(args.version)
        version_label = args.version

    report = format_report(results, version_label)

    if args.output:
        output_path = Path(args.output)
    else:
        bench_dir = REPO_ROOT / "tests" / "benchmarks"
        bench_dir.mkdir(exist_ok=True)
        safe = version_label.replace("/", "_")
        output_path = bench_dir / f"{safe}.txt"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report)
    print(report)
    print(f"\nReport saved to: {output_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
