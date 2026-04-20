"""Unit tests for the Recall/Reasoning/Presentation diagnostic (Theme 16)."""

from __future__ import annotations

import json

from research_pipeline.evaluation.recall_diagnostic import (
    RRPDiagnostic,
    compute_rrp_diagnostic,
)

_GOOD_REPORT = """
# Executive Summary

This review covers arXiv:1234.5678 and arXiv:8765.4321. [1][2]

## Themes

Theme A is well supported [1]. Theme B has evidence in [2].

## Contradictions

Paper arXiv:1234.5678 contradicts arXiv:8765.4321 on the measurement of X;
the disagreement stems from differing baselines.

## Gaps

Key open question: no work addresses the long-horizon regime — future work
should focus on this unaddressed area.

## Confidence

These claims are supported with high confidence where multiple sources agree,
and low confidence where only single-paper evidence is available.
""" + (" filler text to exceed minimum word count." * 80)


def test_returns_rrp_diagnostic() -> None:
    result = compute_rrp_diagnostic(_GOOD_REPORT, ["arXiv:1234.5678"])
    assert isinstance(result, RRPDiagnostic)
    assert 0.0 <= result.overall <= 1.0


def test_full_coverage_yields_high_recall() -> None:
    result = compute_rrp_diagnostic(
        _GOOD_REPORT, ["arXiv:1234.5678", "arXiv:8765.4321"]
    )
    assert result.recall > 0.6


def test_empty_shortlist_zeros_recall() -> None:
    result = compute_rrp_diagnostic(_GOOD_REPORT, [])
    assert result.recall == 0.0
    assert result.bottleneck == "recall"


def test_missing_reasoning_signals_lowers_reasoning_axis() -> None:
    plain = "Introduction. " * 200
    result = compute_rrp_diagnostic(plain, ["arXiv:1234.5678"])
    assert result.reasoning < 0.3


def test_missing_sections_lowers_presentation() -> None:
    result = compute_rrp_diagnostic("one sentence.", ["arXiv:1234.5678"])
    assert result.presentation < 0.3


def test_bottleneck_is_lowest_axis() -> None:
    # Report with strong presentation/reasoning but no shortlist coverage
    text = _GOOD_REPORT  # contains contradictions/gaps/confidence
    result = compute_rrp_diagnostic(text, ["arXiv:9999.0000"])  # not in text
    assert result.bottleneck == "recall"


def test_to_dict_is_json_serializable() -> None:
    result = compute_rrp_diagnostic(_GOOD_REPORT, ["arXiv:1234.5678"])
    payload = result.to_dict()
    json.dumps(payload)
    assert set(payload) >= {
        "recall",
        "reasoning",
        "presentation",
        "overall",
        "bottleneck",
        "details",
    }


def test_citation_density_contributes_to_recall() -> None:
    # Zero markers vs many markers with same shortlist
    no_cites = "arXiv:1234.5678 is mentioned but with no citation markers."
    many_cites = "arXiv:1234.5678 [1] is well cited [2][3][4][5]."
    r1 = compute_rrp_diagnostic(no_cites, ["arXiv:1234.5678"])
    r2 = compute_rrp_diagnostic(many_cites, ["arXiv:1234.5678"])
    assert r2.recall >= r1.recall


def test_scores_never_negative_or_above_one() -> None:
    result = compute_rrp_diagnostic(_GOOD_REPORT, ["arXiv:1234.5678"])
    for axis in (result.recall, result.reasoning, result.presentation, result.overall):
        assert 0.0 <= axis <= 1.0
