"""Tests for opentop.replay.infer_aircraft."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

import pandas as pd
from opentop import replay


def test_infer_aircraft_returns_type_from_icao24():
    pytest.importorskip("traffic")
    flight_df = pd.DataFrame({"icao24": ["4bb9b1"]})

    fake_lookup = MagicMock()
    fake_lookup.get.return_value = pd.DataFrame({"typecode": ["B738"]})

    with patch("traffic.data.aircraft", fake_lookup):
        result = replay.infer_aircraft(flight_df)

    assert result == "B738"


def test_infer_aircraft_returns_none_on_lookup_failure():
    pytest.importorskip("traffic")
    flight_df = pd.DataFrame({"icao24": ["ffffff"]})

    fake_lookup = MagicMock()
    fake_lookup.get.return_value = pd.DataFrame()

    with patch("traffic.data.aircraft", fake_lookup):
        result = replay.infer_aircraft(flight_df)

    assert result is None


def test_infer_aircraft_returns_none_on_empty_icao24_column():
    """Empty or missing icao24 column → no lookup, return None."""
    result_empty_col = replay.infer_aircraft(pd.DataFrame(columns=["icao24"]))
    assert result_empty_col is None

    result_missing_col = replay.infer_aircraft(pd.DataFrame({"foo": [1]}))
    assert result_missing_col is None
