"""Interactive wizard for `opentop replay`.

Prompts the user for every option, with the default shown in brackets and
enter-to-accept. Conditional prompts (e.g. sigma only for grid objectives)
are skipped when irrelevant.
"""

from __future__ import annotations

from pathlib import Path
from tempfile import gettempdir
from typing import Any, Optional

import click


def _uses_grid(objective: str) -> bool:
    return any(tok in objective for tok in ("grid", "gwp", "gtp"))


def _prompt_with_default(label: str, default: str) -> str:
    """Prompt with the default shown inline; enter accepts default."""
    answer = click.prompt(f"{label} [{default}]", default=default, show_default=False)
    return str(answer).strip()


def run_wizard() -> Optional[dict[str, Any]]:
    """Collect all replay options interactively. Returns dict or None on cancel."""
    click.echo("opentop replay — interactive mode\n")

    callsign = click.prompt("Callsign")
    date_ = click.prompt("Date (YYYY-MM-DD)")
    narrower = _prompt_with_default("Use a narrower time window? [y/n]", "n").lower()
    if narrower == "y":
        start = click.prompt("Start (ISO datetime)")
        stop = click.prompt("Stop (ISO datetime)")
    else:
        start = f"{date_}T00:00"
        stop = f"{date_}T23:59"

    aircraft_in = _prompt_with_default("Aircraft [auto-detect]", "").strip()
    aircraft: Optional[str] = aircraft_in or None

    m0 = float(_prompt_with_default("Initial mass ratio m0", "0.85"))
    phase = _prompt_with_default("Phase [all/cruise/climb/descent]", "all")
    objective = _prompt_with_default("Objective", "fuel")

    sigma = 2
    if _uses_grid(objective):
        sigma = int(_prompt_with_default("Contrail smoothing sigma", "2"))

    no_wind_in = _prompt_with_default("Disable wind? [y/n]", "n").lower()
    no_wind = no_wind_in == "y"

    max_iter = int(_prompt_with_default("IPOPT max iterations", "1500"))

    era5_default = str(Path(gettempdir()) / "opentop-era5")
    era5_store = _prompt_with_default("ERA5 store path", era5_default)

    output_default = f"./replay_{callsign}_{date_}"
    output_dir = Path(_prompt_with_default("Output directory", output_default))

    debug_in = _prompt_with_default("Verbose IPOPT output? [y/n]", "n").lower()
    debug = debug_in == "y"

    click.echo("\nSummary:")
    click.echo(f"  callsign={callsign} date={date_} window={start}..{stop}")
    click.echo(
        f"  aircraft={aircraft or '(auto)'} m0={m0} phase={phase} "
        f"objective={objective} wind={'off' if no_wind else 'on'}"
    )
    click.echo(f"  sigma={sigma} max_iter={max_iter} output={output_dir}")

    confirm = _prompt_with_default("Proceed? [y/n]", "y").lower()
    if confirm and confirm[0] == "n":
        click.echo("Cancelled.")
        return None

    return {
        "callsign": callsign,
        "date": date_,
        "start": start,
        "stop": stop,
        "from_file": None,
        "aircraft": aircraft,
        "m0": m0,
        "phase": phase,
        "objective": objective,
        "sigma": sigma,
        "no_wind": no_wind,
        "max_iter": max_iter,
        "era5_store": era5_store,
        "output_dir": output_dir,
        "debug": debug,
    }
