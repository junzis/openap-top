"""opentop gengrid subcommand."""

from __future__ import annotations

import time
from pathlib import Path
from typing import cast

import click

import pandas as pd
from opentop import tools

from ._helpers import _pad_altitudes, _parse_bbox, _parse_time_window


@click.command()
@click.option(
    "--in",
    "input_path",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Input grid file (parquet).",
)
@click.option(
    "--out",
    "output_path",
    required=True,
    type=click.Path(path_type=Path),
    help="Output .casadi cache file.",
)
@click.option("--bbox", help="Spatial bbox as LAT_MIN:LAT_MAX,LON_MIN:LON_MAX.")
@click.option(
    "--time",
    "time_window",
    help="Time window as START,STOP (ISO-format timestamps, UTC assumed).",
)
@click.option(
    "--pad-altitudes/--no-pad-altitudes",
    default=True,
    show_default=True,
    help="Extend altitude coverage with zero-cost rows from 0 to FL480.",
)
@click.option(
    "--shape",
    type=click.Choice(["linear", "bspline"]),
    default="bspline",
    show_default=True,
    help="Interpolant type. 'bspline' has smooth gradients and is strongly "
    "preferred for CompleteFlight solves.",
)
def gengrid(
    input_path: Path,
    output_path: Path,
    bbox: str | None,
    time_window: str | None,
    pad_altitudes: bool,
    shape: str,
) -> None:
    """Build and cache a CasADi interpolant from a raw cost grid parquet."""
    click.echo(f"Loading {input_path}")
    df = pd.read_parquet(input_path)
    click.echo(f"  raw shape: {df.shape}")

    if time_window is not None:
        t0, t1 = _parse_time_window(time_window)
        df = cast(pd.DataFrame, df[(df.timestamp >= t0) & (df.timestamp <= t1)].copy())
        if "ts" in df.columns:
            df["ts"] = df.ts - df.ts.min()
        click.echo(f"  after time slice: {df.shape}")

    if bbox is not None:
        lat_min, lat_max, lon_min, lon_max = _parse_bbox(bbox)
        df = cast(
            pd.DataFrame,
            df[
                (df.latitude >= lat_min)
                & (df.latitude <= lat_max)
                & (df.longitude >= lon_min)
                & (df.longitude <= lon_max)
            ].copy(),
        )
        click.echo(f"  after bbox slice: {df.shape}")

    if pad_altitudes:
        before = df.shape[0]
        df = _pad_altitudes(df)
        click.echo(
            f"  after altitude padding: {df.shape} (+{df.shape[0] - before} rows)"
        )

    click.echo(f"\nBuilding {shape} interpolant ({df.shape[0]:,} points)...")
    t0 = time.time()
    interpolant = tools.interpolant_from_dataframe(df, shape=shape)
    click.echo(f"  build time: {time.time() - t0:.1f} s")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tools.save_interpolant(interpolant, output_path)
    size_mb = output_path.stat().st_size / 1024 / 1024
    click.echo(f"\n  saved to {output_path} ({size_mb:.1f} MB)")
