"""Unit tests for the opentop CLI."""

import click
import pytest

from opentop.cli import parse_objective


class TestParseObjective:
    """Tests for the objective-expression parser."""

    def test_single_term_no_weight(self):
        assert parse_objective("fuel") == [(1.0, "fuel", None)]

    def test_single_term_with_ci(self):
        assert parse_objective("ci:30") == [(1.0, "ci", "30")]

    def test_single_term_climate(self):
        assert parse_objective("gwp100") == [(1.0, "gwp100", None)]

    def test_two_terms_weighted(self):
        parsed = parse_objective("0.3*fuel+0.7*grid")
        assert parsed == [(0.3, "fuel", None), (0.7, "grid", None)]

    def test_three_terms(self):
        parsed = parse_objective("0.4*fuel+0.4*grid+0.2*time")
        assert parsed == [
            (0.4, "fuel", None),
            (0.4, "grid", None),
            (0.2, "time", None),
        ]

    def test_blend_climate_and_fuel(self):
        parsed = parse_objective("0.5*fuel+0.5*gwp100")
        assert parsed == [(0.5, "fuel", None), (0.5, "gwp100", None)]

    def test_whitespace_tolerated(self):
        parsed = parse_objective(" 0.3 * fuel + 0.7 * grid ")
        assert parsed == [(0.3, "fuel", None), (0.7, "grid", None)]

    def test_mixed_weight_and_no_weight(self):
        parsed = parse_objective("0.5*fuel+grid")
        assert parsed == [(0.5, "fuel", None), (1.0, "grid", None)]

    def test_unknown_term_raises(self):
        with pytest.raises(click.UsageError, match="unknown objective term"):
            parse_objective("bogus")

    def test_ci_without_parameter_raises(self):
        with pytest.raises(click.UsageError, match="'ci' requires a parameter"):
            parse_objective("ci")

    def test_malformed_term_raises(self):
        with pytest.raises(click.UsageError, match="cannot parse objective term"):
            parse_objective("0.3*")

    def test_empty_string_raises(self):
        with pytest.raises(click.UsageError):
            parse_objective("")
