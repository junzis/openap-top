"""opentop optimize subcommand."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import click

import opentop as top

from ._helpers import build_objective_callable, load_grid_file, parse_objective


@click.command()
@click.argument("origin")
@click.argument("destination")
@click.option("-a", "--aircraft", required=True, help="Aircraft type, e.g. A320.")
@click.option(
    "--m0",
    type=float,
    default=0.85,
    show_default=True,
    help="Initial mass ratio (fraction of MTOW).",
)
@click.option(
    "--phase",
    type=click.Choice(["all", "cruise", "climb", "descent"]),
    default="all",
    show_default=True,
    help="Flight phase (all = CompleteFlight).",
)
@click.option(
    "--obj",
    "objective",
    default="fuel",
    show_default=True,
    help='Objective expression, e.g. "fuel", "ci:30", "0.3*fuel+0.7*grid".',
)
@click.option(
    "--grid",
    "grid_path",
    type=click.Path(exists=True, path_type=Path),
    help="Grid cost file (.casadi preferred, .parquet also accepted).",
)
@click.option(
    "--max-iter",
    type=int,
    default=1500,
    show_default=True,
    help="Maximum IPOPT iterations.",
)
@click.option(
    "-o",
    "--output",
    type=click.Path(path_type=Path),
    help="Save trajectory to this parquet file (stdout summary only if omitted).",
)
@click.option("-v", "--debug", is_flag=True, help="Verbose IPOPT output.")
def optimize(
    origin: str,
    destination: str,
    aircraft: str,
    m0: float,
    phase: str,
    objective: str,
    grid_path: Path | None,
    max_iter: int,
    output: Path | None,
    debug: bool,
) -> None:
    """Run trajectory optimization for ORIGIN → DESTINATION."""
    phase_map = {
        "all": top.CompleteFlight,
        "cruise": top.Cruise,
        "climb": top.Climb,
        "descent": top.Descent,
    }
    phase_cls = phase_map[phase]

    interpolant = load_grid_file(grid_path) if grid_path else None
    terms = parse_objective(objective)

    click.echo(f"  aircraft:  {aircraft}")
    click.echo(f"  route:     {origin} → {destination}")
    click.echo(f"  phase:     {phase}")
    click.echo(f"  objective: {objective}")
    click.echo(f"  m0:        {m0}")
    click.echo(f"  max_iter:  {max_iter}")
    if grid_path:
        click.echo(f"  grid file: {grid_path}")

    opt = phase_cls(aircraft, origin, destination, m0)
    opt.setup(debug=debug, max_iter=max_iter)

    obj_arg = build_objective_callable(opt, terms, interpolant=interpolant)

    traj_kwargs: dict = {
        "objective": obj_arg,
        "return_failed": True,
    }
    if interpolant is not None:
        traj_kwargs["interpolant"] = interpolant

    t0 = time.time()
    df = opt.trajectory(**traj_kwargs)
    wall = time.time() - t0

    stats = opt.stats
    click.echo("")
    click.echo(f"  success:       {stats.get('success')}")
    click.echo(f"  return_status: {stats.get('return_status')}")
    click.echo(f"  iter_count:    {stats.get('iter_count')}")
    click.echo(f"  wall time:     {wall:.1f} s")
    click.echo(f"  objective:     {opt.objective_value:.4e}")

    if df is not None and not df.empty:
        fuel = df.mass.iloc[0] - df.mass.iloc[-1]
        click.echo(f"  fuel burn:     {fuel:.1f} kg")
        click.echo(f"  max altitude:  {df.altitude.max():.0f} ft")
        click.echo(f"  flight time:   {df.ts.iloc[-1] / 60:.1f} min")

    if output is not None and df is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output)
        click.echo(f"\n  trajectory saved to {output}")

    if not stats.get("success", False):
        sys.exit(1)
