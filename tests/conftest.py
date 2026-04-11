"""Shared fixtures for opentop tests."""

import sys
from pathlib import Path

# Add the project root to path so ``import opentop`` resolves to the local
# checkout even before the package is properly installed.
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pytest


@pytest.fixture
def aircraft_type():
    """Default aircraft type for testing."""
    return "A320"


@pytest.fixture
def short_flight():
    """Short flight route for faster tests."""
    return {
        "origin": "EHAM",  # Amsterdam
        "destination": "EDDF",  # Frankfurt
        "m0": 0.85,
    }


@pytest.fixture
def medium_flight():
    """Medium flight route."""
    return {
        "origin": "EHAM",  # Amsterdam
        "destination": "LGAV",  # Athens
        "m0": 0.85,
    }
