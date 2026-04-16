"""opentop replay command."""

from __future__ import annotations

import sys
import time
from pathlib import Path
from tempfile import gettempdir
from typing import Optional

import click

import opentop as top

DEFAULT_ERA5_STORE = str(Path(gettempdir()) / "opentop-era5")


def _check_replay_deps(need_opensky: bool, need_meteo: bool) -> Optional[str]:
    """Return an error message if optional deps are missing for the chosen mode."""
    needed = []
    if need_opensky:
        needed.append("traffic")
    if need_meteo:
        needed += ["fastmeteo", "scipy"]

    for pkg in needed:
        try:
            __import__(pkg)
        except ImportError:
            return (
                f"opentop replay requires '{pkg}' (and others). "
                'Install with: pip install "opentop[replay]"'
            )
    return None


@click.command()
@click.argument("callsign", required=False)
@click.option("--date", "date_", help="Date (YYYY-MM-DD). Fetches 00:00-24:00 UTC.")
@click.option("--start", help="ISO datetime for narrower window start.")
@click.option("--stop", help="ISO datetime for narrower window end.")
@click.option(
    "--from-file",
    "from_file",
    type=click.Path(exists=True, path_type=Path),
    help="Read flight from a saved Traffic parquet; skips OpenSky.",
)
@click.option("-a", "--aircraft", help="Aircraft type. Default: infer from ICAO24.")
@click.option("--m0", type=float, default=0.85, show_default=True)
@click.option(
    "--phase",
    type=click.Choice(["all", "cruise", "climb", "descent"]),
    default="all",
    show_default=True,
)
@click.option("--obj", "objective", default="fuel", show_default=True)
@click.option("--max-iter", "max_iter", type=int, default=1500, show_default=True)
@click.option("--no-wind", "no_wind", is_flag=True, help="Skip ERA5 wind.")
@click.option("--sigma", type=int, default=2, show_default=True)
@click.option(
    "--era5-store",
    "era5_store",
    default=DEFAULT_ERA5_STORE,
    show_default=True,
    help="ArcoEra5 local store path.",
)
@click.option(
    "-o",
    "--output-dir",
    "output_dir",
    type=click.Path(path_type=Path),
    help="Output directory. Default: ./replay_<callsign>_<date>.",
)
@click.option("-v", "--debug", is_flag=True, help="Verbose IPOPT output.")
def replay(
    callsign: Optional[str],
    date_: Optional[str],
    start: Optional[str],
    stop: Optional[str],
    from_file: Optional[Path],
    aircraft: Optional[str],
    m0: float,
    phase: str,
    objective: str,
    max_iter: int,
    no_wind: bool,
    sigma: int,
    era5_store: str,
    output_dir: Optional[Path],
    debug: bool,
) -> None:
    """Replay a real flight by callsign with optimization.

    If CALLSIGN is omitted, drops into the interactive wizard.
    """
    import pandas as pd
    from opentop import replay as replay_mod

    # No positional arg → dispatch to interactive wizard
    if callsign is None and from_file is None:
        from ._interactive import run_wizard

        opts = run_wizard()
        if opts is None:
            sys.exit(0)
        callsign = opts["callsign"]
        date_ = opts.get("date")
        start = opts.get("start")
        stop = opts.get("stop")
        from_file = opts.get("from_file")
        aircraft = opts.get("aircraft")
        m0 = opts["m0"]
        phase = opts["phase"]
        objective = opts["objective"]
        max_iter = opts["max_iter"]
        no_wind = opts["no_wind"]
        sigma = opts["sigma"]
        era5_store = opts["era5_store"]
        output_dir = opts["output_dir"]
        debug = opts["debug"]

    # Dep guard
    need_opensky = from_file is None
    need_meteo = not no_wind or any(tok in objective for tok in ("grid", "gwp", "gtp"))
    err = _check_replay_deps(need_opensky=need_opensky, need_meteo=need_meteo)
    if err:
        click.echo(err, err=True)
        sys.exit(1)

    # Resolve time window
    if date_ and (start or stop):
        click.echo("--date is mutually exclusive with --start/--stop", err=True)
        sys.exit(1)
    if date_:
        start = f"{date_}T00:00"
        stop = f"{date_}T23:59"
    if from_file is None and (not start or not stop):
        click.echo("Provide --date or both --start and --stop.", err=True)
        sys.exit(1)
    # from_file without window: synthetic window (file itself dictates what's loaded)
    if from_file is not None and (not start or not stop):
        start = start or "1970-01-01"
        stop = stop or "2099-12-31"

    # Fetch flight
    click.echo(f"Fetching flight {callsign}...")
    source = from_file if from_file else "opensky"
    assert start is not None and stop is not None  # guards above narrowed these
    flight = replay_mod.fetch_flight(callsign or "", start, stop, source=source)
    click.echo(f"  got {len(flight)} points")

    # Resolve aircraft
    if not aircraft:
        aircraft = replay_mod.infer_aircraft(flight)
        if not aircraft:
            click.echo(
                "Cannot infer aircraft from ICAO24; pass -a <type>.",
                err=True,
            )
            sys.exit(4)
        click.echo(f"  aircraft: {aircraft} (auto-detected)")

    # Default output dir
    if output_dir is None:
        try:
            date_tag = pd.Timestamp(start).strftime("%Y-%m-%d")  # type: ignore[union-attr]
        except (ValueError, AttributeError):
            date_tag = "unknown"
        cs_tag = callsign or "flight"
        output_dir = Path(f"./replay_{cs_tag}_{date_tag}")

    # Build optimizer
    phase_map = {
        "all": top.CompleteFlight,
        "cruise": top.Cruise,
        "climb": top.Climb,
        "descent": top.Descent,
    }
    opt_cls = phase_map[phase]

    # Filter obvious altitude outliers in the flight (OpenSky baro spikes)
    flight_filtered = flight[
        (flight["altitude"] > 0) & (flight["altitude"] < 45000)
    ].reset_index(drop=True)
    if flight_filtered.empty:
        click.echo(
            "Flight trace has no samples with 0 < altitude < 45000 ft.",
            err=True,
        )
        sys.exit(3)

    lat0 = float(flight_filtered["latitude"].iloc[0])  # type: ignore[union-attr]
    lon0 = float(flight_filtered["longitude"].iloc[0])  # type: ignore[union-attr]
    lat1 = float(flight_filtered["latitude"].iloc[-1])  # type: ignore[union-attr]
    lon1 = float(flight_filtered["longitude"].iloc[-1])  # type: ignore[union-attr]
    opt = opt_cls(aircraft, (lat0, lon0), (lat1, lon1), m0=m0)
    opt.setup(debug=debug, max_iter=max_iter)

    # Fetch meteo once if we need it for either wind or contrail grid.
    needs_grid = any(tok in objective for tok in ("grid", "gwp", "gtp"))
    meteo_df = None
    wind_df = None
    if not no_wind or needs_grid:
        click.echo("Fetching ERA5 meteo...")
        meteo_df, wind_df = replay_mod.build_meteo_and_wind(
            flight, era5_store=era5_store
        )

    if not no_wind and wind_df is not None:
        click.echo("  enabling wind field")
        opt.enable_wind(wind_df)

    interpolant = None
    if needs_grid and meteo_df is not None:
        click.echo("Building contrail interpolant...")
        interpolant = replay_mod.build_contrail_interpolant(meteo_df, sigma=sigma)

    # Build objective callable
    from ._helpers import build_objective_callable, parse_objective

    terms = parse_objective(objective)
    obj_callable = build_objective_callable(opt, terms, interpolant=interpolant)

    # Run optimization
    click.echo("Running optimization...")
    t0 = time.time()
    traj_kwargs: dict = {"objective": obj_callable, "return_failed": True}
    if interpolant is not None:
        traj_kwargs["interpolant"] = interpolant
        traj_kwargs["n_dim"] = 4
        traj_kwargs["time_dependent"] = True
    optimized = opt.trajectory(**traj_kwargs)
    wall = time.time() - t0

    # Summary
    stats = opt.stats
    click.echo("")
    click.echo(f"  success:       {stats.get('success')}")
    click.echo(f"  iterations:    {stats.get('iter_count')}")
    click.echo(f"  wall time:     {wall:.1f} s")
    try:
        click.echo(f"  objective:     {opt.objective_value:.4e}")
    except (AttributeError, TypeError):
        pass

    if optimized is not None and not optimized.empty:
        opt_fuel = float(optimized.mass.iloc[0] - optimized.mass.iloc[-1])
        click.echo(f"  optimized fuel: {opt_fuel:.0f} kg")
        if "mass" in flight.columns and bool(flight["mass"].notna().any()):
            actual_fuel = float(flight.mass.iloc[0] - flight.mass.iloc[-1])
            if actual_fuel > 0:
                delta_pct = (opt_fuel - actual_fuel) / actual_fuel * 100
                click.echo(
                    f"  actual fuel:    {actual_fuel:.0f} kg  "
                    f"(optimized Δ {delta_pct:+.1f}%)"
                )

    # Save outputs
    output_dir.mkdir(parents=True, exist_ok=True)
    flight.to_parquet(output_dir / "actual.parquet")
    if optimized is not None:
        optimized.to_parquet(output_dir / "optimized.parquet")

        from opentop import vis

        # vis.trajectory expects ts/tas/vertical_rate; the raw OpenSky trace
        # has timestamp/groundspeed/vertical_rate. Derive missing columns for
        # a best-effort overlay plot.
        flight_plot = flight_filtered.copy()
        if "ts" not in flight_plot.columns and "timestamp" in flight_plot.columns:
            flight_plot["ts"] = (
                flight_plot["timestamp"] - flight_plot["timestamp"].iloc[0]  # type: ignore[union-attr]
            ).dt.total_seconds()
        if "tas" not in flight_plot.columns and "groundspeed" in flight_plot.columns:
            # Ground speed is a reasonable stand-in when TAS isn't available.
            flight_plot["tas"] = flight_plot["groundspeed"]
        if "vertical_rate" not in flight_plot.columns:
            flight_plot["vertical_rate"] = 0.0

        plt = vis.trajectory([flight_plot, optimized], labels=["actual", "optimized"])
        plt.savefig(output_dir / "trajectory.png", dpi=120, bbox_inches="tight")
        plt.close("all")

    click.echo(f"\n  outputs saved to {output_dir}")

    if not stats.get("success", False):
        sys.exit(1)
