"""Entry-point bootstrap for the ``opentop`` console script.

``openap-top`` ships ``openap.top`` as a namespace extension of the PyPI
``openap`` package. Under an editable install (``.pth``-based), Python finds
the PyPI ``openap/__init__.py`` first and ``openap.top`` is not discoverable
until we manually extend ``openap.__path__`` to include our local
``openap/`` directory. This wrapper performs that patching before importing
``openap.top.cli``, so the console script works regardless of install mode.
"""

from __future__ import annotations


def main() -> None:
    import sys
    from pathlib import Path

    import openap

    if hasattr(openap, "__path__"):
        repo_openap = str(Path(__file__).resolve().parent / "openap")
        if repo_openap not in openap.__path__:
            openap.__path__.insert(0, repo_openap)

    from openap.top.cli import main as _cli_main

    sys.exit(_cli_main())


if __name__ == "__main__":
    main()
