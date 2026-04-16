"""Tests for the `opentop replay` CLI command (flag mode)."""

from __future__ import annotations

from pathlib import Path

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
                type(result.exception),
                result.exception,
                result.exception.__traceback__,
            )
    assert result.exit_code == 0, f"replay failed with exit {result.exit_code}"
    assert (out_dir / "actual.parquet").exists()
    assert (out_dir / "optimized.parquet").exists()
    assert (out_dir / "trajectory.png").exists()


@pytest.mark.skip(reason="wizard ships in Task 10")
def test_replay_missing_args_drops_into_wizard():
    runner = CliRunner()
    result = runner.invoke(cli_main, ["replay"], input="\n")
    assert "Callsign" in result.output or "callsign" in result.output.lower()
