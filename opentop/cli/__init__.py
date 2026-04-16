"""opentop CLI package.

Entry point `main` is a click.group that registers all subcommands.
"""

from __future__ import annotations

import click

from ._helpers import parse_objective
from .gengrid import gengrid
from .optimize import optimize


@click.group()
@click.version_option(package_name="opentop", prog_name="opentop")
def main() -> None:
    """opentop: aircraft trajectory optimization CLI."""


main.add_command(optimize)
main.add_command(gengrid)


__all__ = ["main", "parse_objective"]
