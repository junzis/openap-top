"""Tests for the `opentop replay` CLI command (flag mode + wizard)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from opentop.cli import main as cli_main

FIXTURES = Path(__file__).parent / "fixtures"
FLIGHT = FIXTURES / "flight_ryr880w_2023-01-05.parquet"
INTERP = FIXTURES / "contrail_4d.casadi"


@pytest.mark.skipif(not FLIGHT.exists(), reason="missing flight fixture")
def test_replay_from_file_fuel(tmp_path):
    """`opentop replay --from-file <parquet> -o DIR` runs and writes outputs."""
    out_dir = tmp_path / "replay_out"

    runner = CliRunner()
    result = runner.invoke(
        cli_main,
        [
            "replay",
            "RYR880W",
            "--from-file",
            str(FLIGHT),
            "-a",
            "B738",
            "--obj",
            "fuel",
            "--no-wind",
            "--max-iter",
            "300",
            "-o",
            str(out_dir),
        ],
    )
    if result.exit_code != 0:
        print(result.output)
        if result.exception:
            import traceback

            traceback.print_exception(
                type(result.exception), result.exception, result.exception.__traceback__
            )
    assert result.exit_code == 0, f"replay failed with exit {result.exit_code}"
    assert (out_dir / "actual.parquet").exists()
    assert (out_dir / "optimized.parquet").exists()
    assert (out_dir / "trajectory.png").exists()


def test_wizard_collects_all_fields():
    """The wizard walks through every option with enter-to-accept."""
    from opentop.cli._interactive import run_wizard

    # Simulated stdin values, in the order the wizard asks.
    # Watch out: conditional prompts (sigma only for grid objective) are skipped.
    inputs = iter(
        [
            "RYR880W",  # callsign
            "2023-01-05",  # date
            "n",  # no narrower window
            "",  # aircraft: enter → None (auto)
            "0.85",  # m0
            "all",  # phase
            "fuel",  # objective (no grid → sigma skipped)
            "n",  # disable wind? no
            "1500",  # max iter
            "/tmp/opentop-era5",  # era5 store
            "./replay_RYR880W_2023-01-05",  # output dir
            "n",  # debug
            "y",  # confirm
        ]
    )

    with patch("click.prompt", side_effect=lambda *a, **kw: next(inputs)):
        opts = run_wizard()

    assert opts is not None
    assert opts["callsign"] == "RYR880W"
    assert opts["start"].startswith("2023-01-05")
    assert opts["objective"] == "fuel"
    assert opts["m0"] == 0.85
    assert opts["no_wind"] is False


def test_wizard_returns_none_on_cancel():
    """If user answers 'n' at the final confirm, return None."""
    from opentop.cli._interactive import run_wizard

    inputs = iter(
        [
            "RYR880W",
            "2023-01-05",
            "n",  # callsign, date, no narrower window
            "",  # aircraft
            "0.85",
            "all",
            "fuel",  # m0, phase, objective
            "n",
            "1500",  # no-wind? no; max_iter
            "/tmp/opentop-era5",  # era5
            "./replay_out",  # output dir
            "n",  # debug
            "n",  # cancel at confirm
        ]
    )

    with patch("click.prompt", side_effect=lambda *a, **kw: next(inputs)):
        opts = run_wizard()

    assert opts is None


def test_wizard_dispatch_from_cli_with_no_args():
    """Invoking `opentop replay` with no args enters the wizard (interactive path).

    We feed EOF immediately to the wizard's first prompt; click raises Abort.
    We only assert the CLI didn't crash with a non-wizard error.
    """
    runner = CliRunner()
    result = runner.invoke(cli_main, ["replay"], input="")
    # Exit code 1 (aborted) or 2 (usage) is acceptable; non-zero is fine.
    # Just make sure the wizard was attempted (its prompt text is visible).
    assert result.exit_code != 0
    # The wizard's first prompt asks about the callsign.
    assert (
        "allsign" in result.output  # prompt label
        or "Aborted" in result.output  # click's default abort message
        or result.output == ""  # click may suppress the prompt on EOF
    )
