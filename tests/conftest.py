"""Shared fixtures for opentop tests."""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pytest  # noqa: E402  (sys.path must be set before importing project packages)


@pytest.fixture(scope="session")
def aircraft_type():
    return "A320"


@pytest.fixture(scope="session")
def short_flight():
    return {"origin": "EHAM", "destination": "EDDF", "m0": 0.85}


@pytest.fixture(scope="session")
def medium_flight():
    return {"origin": "EHAM", "destination": "LGAV", "m0": 0.85}
