"""Unit tests for the configurable criteria extension in
:mod:`research_pipeline.quality.graduated_rubric`."""

from __future__ import annotations

import pytest

from research_pipeline.quality.graduated_rubric import (
    Criterion,
    aggregate_criteria_by_dimension,
    load_criteria_from_json,
    score_criteria,
)


def test_default_scorer_detects_keywords():
    crits = [
        Criterion(
            name="cites_papers",
            dimension="evidence",
            description="citation reference paper",
        ),
        Criterion(
            name="gap_analysis",
            dimension="coverage",
            description="gap limitation missing",
        ),
    ]
    text = "This report includes a paper citation and a reference to prior work."
    results = score_criteria(text, crits)
    by_name = {r.criterion.name: r for r in results}
    assert by_name["cites_papers"].score > 0.0
    assert by_name["cites_papers"].passed is True
    assert by_name["gap_analysis"].score == 0.0


def test_custom_scorer_receives_criterion():
    crits = [
        Criterion(name="x", dimension="d1", description="foo"),
        Criterion(name="y", dimension="d1", description="bar"),
    ]
    calls = []

    def scorer(text: str, criterion: Criterion) -> float:
        calls.append(criterion.name)
        return 0.75

    results = score_criteria("unused", crits, scorer=scorer)
    assert [r.score for r in results] == [0.75, 0.75]
    assert calls == ["x", "y"]
    assert all(r.passed for r in results)


def test_aggregate_by_dimension_respects_weights():
    crits = [
        Criterion(name="a", dimension="evidence", weight=1.0, description="foo"),
        Criterion(name="b", dimension="evidence", weight=3.0, description="bar"),
    ]

    def stub(text: str, c: Criterion) -> float:
        return 1.0 if c.name == "b" else 0.0

    results = score_criteria("text", crits, scorer=stub)
    agg = aggregate_criteria_by_dimension(results)
    # Weighted mean: (0*1 + 1*3) / (1+3) = 0.75
    assert agg["evidence"] == pytest.approx(0.75)


def test_load_criteria_from_json_roundtrip():
    payload = [
        {
            "name": "c1",
            "dimension": "d",
            "weight": 2.0,
            "pass_threshold": 0.4,
            "description": "alpha beta",
        },
        {"name": "c2", "dimension": "d"},
    ]
    loaded = load_criteria_from_json(payload)
    assert len(loaded) == 2
    assert loaded[0].weight == 2.0
    assert loaded[0].pass_threshold == 0.4
    assert loaded[1].weight == 1.0  # default
    assert loaded[1].pass_threshold == 0.5  # default


def test_load_criteria_missing_required_field_raises():
    with pytest.raises(ValueError, match="missing"):
        load_criteria_from_json([{"dimension": "d"}])  # no name


def test_empty_dimension_aggregates_to_zero():
    crits = [Criterion(name="c", dimension="d", weight=0.0, description="x")]
    results = score_criteria("x", crits, scorer=lambda _t, _c: 1.0)
    agg = aggregate_criteria_by_dimension(results)
    assert agg["d"] == 0.0
