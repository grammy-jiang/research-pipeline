"""Tests for RACE report quality scoring."""

from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from research_pipeline.quality.race_scoring import (
    RACEScore,
    compute_race_score,
    score_actionability,
    score_comprehensiveness,
    score_evidence,
    score_readability,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_well_structured_text() -> str:
    """Build a multi-paragraph, well-structured report excerpt."""
    sections = [
        "# Research Report",
        "",
        "## Executive Summary",
        "",
        "This study examines recent advances in transformer architectures. "
        "We surveyed ten papers published between 2023 and 2024. "
        "The findings suggest significant performance improvements. "
        "Further investigation is recommended.",
        "",
        "## Research Question",
        "",
        "How do modern transformer variants compare for time-series tasks? "
        "This question guides our literature survey and analysis.",
        "",
        "## Methodology",
        "",
        "We searched arXiv and Google Scholar for relevant papers. "
        "Each paper was screened for relevance using BM25 scoring. "
        "Papers were evaluated for quality using citation metrics.",
        "",
        "- Step one: query formulation",
        "- Step two: candidate retrieval",
        "- Step three: relevance screening",
        "- Step four: quality evaluation",
        "- Step five: synthesis",
        "",
        "## Papers Reviewed",
        "",
        "| Paper | Year | Method |",
        "| --- | --- | --- |",
        "| [2401.12345] | 2024 | Attention |",
        "| [2401.67890] | 2024 | Convolution |",
        "",
        "## Research Landscape",
        "",
        "The field is active with 50% growth year over year. "
        "Key themes include efficiency and scalability.",
        "",
        "## Research Gaps",
        "",
        "ACADEMIC gaps remain in long-context modelling. "
        "ENGINEERING gaps exist in deployment tooling.",
        "",
        "## Practical Recommendations",
        "",
        "Practitioners should consider using efficient attention. "
        "We recommend adopting sliding-window transformers. "
        "Teams should implement proper benchmarking. "
        "Results suggest applying pruning for edge deployment.",
        "",
        "## References",
        "",
        "- [2401.12345] Paper A (2024)",
        "- [2401.67890] Paper B (2024)",
        "- [2401.11111] Paper C (2024)",
        "- [2401.22222] Paper D (2024)",
        "- [2401.33333] Paper E (2024)",
        "",
        "## Appendix",
        "",
        "Run metadata and supplementary tables.",
    ]
    return "\n".join(sections)


def _make_full_report() -> str:
    """Build a complete report with all RACE-relevant features."""
    sections = [
        "# Research Report",
        "",
        "## Executive Summary",
        "",
        "This study examines transformer architectures [2401.12345]. "
        "We surveyed 10 papers achieving 95.2% coverage. "
        "Findings suggest improvements of 12% on benchmarks.",
        "",
        "## Research Question",
        "",
        "How do modern transformers compare?",
        "",
        "## Methodology",
        "",
        "We used BM25 scoring and citation analysis [2401.67890].",
        "",
        "## Papers Reviewed",
        "",
        "| Paper | Year |",
        "| --- | --- |",
        "| [2401.12345] | 2024 |",
        "| [2401.67890] | 2024 |",
        "",
        "## Research Landscape",
        "",
        "🟢 **High confidence**: Transformers dominate NLP [2401.12345].",
        "🟡 **Medium confidence**: Vision transformers are competitive [2401.67890].",
        "🔴 **Low confidence**: Audio transformers need more study.",
        "",
        "> Evidence: Paper A reports 95% accuracy on benchmark X.",
        "> Evidence: Paper B shows 12% improvement over baseline.",
        "> Evidence: Paper C finds diminishing returns above 1B params.",
        "",
        "## Confidence-Graded Findings",
        "",
        "🟢 High confidence: Attention is all you need.",
        "🟡 Medium confidence: Sparse attention is promising.",
        "",
        "## Research Gaps",
        "",
        "- ACADEMIC: Long-context modelling (HIGH severity)",
        "- ENGINEERING: Deployment on edge devices",
        "",
        "## Evidence Map",
        "",
        "| Claim | Sources | Confidence |",
        "| --- | --- | --- |",
        "| Attention works | [2401.12345], [2401.67890] | High |",
        "",
        "## Practical Recommendations",
        "",
        "1. Teams should adopt efficient attention mechanisms.",
        "2. We recommend implementing sliding-window transformers.",
        "3. Consider applying pruning for smaller models.",
        "4. Use quantization to reduce memory footprint.",
        "",
        "## References",
        "",
        "- [2401.12345] Paper A",
        "- [2401.67890] Paper B",
        "- [2401.11111] Paper C",
        "- [2401.22222] Paper D",
        "- [2401.33333] Paper E",
        "",
        "## Appendix",
        "",
        "Run metadata follows.",
    ]
    return "\n".join(sections)


# ---------------------------------------------------------------------------
# Readability
# ---------------------------------------------------------------------------


class TestScoreReadability:
    def test_short_text_low_score(self) -> None:
        """Very short text should score low on readability."""
        score, details = score_readability("Hello world.")
        assert 0.0 <= score <= 1.0
        assert score < 0.5
        assert details["word_count"] == 2

    def test_well_structured_text(self) -> None:
        """Multi-paragraph text with headings and lists scores higher."""
        text = _make_well_structured_text()
        score, details = score_readability(text)
        assert 0.0 <= score <= 1.0
        assert score > 0.4
        assert details["heading_count"] >= 5
        assert details["bullet_items"] >= 5
        assert details["paragraph_count"] >= 5

    def test_empty_text(self) -> None:
        """Empty string should return zero."""
        score, details = score_readability("")
        assert score == 0.0
        assert details["word_count"] == 0

    def test_details_keys(self) -> None:
        """Details dict should contain all expected keys."""
        _, details = score_readability("Some text here.")
        expected_keys = {
            "word_count",
            "sentence_count",
            "avg_sentence_length",
            "paragraph_count",
            "avg_paragraph_length",
            "heading_count",
            "heading_density_per_1000",
            "bullet_items",
            "numbered_items",
            "sentence_score",
            "paragraph_score",
            "heading_score",
            "list_score",
        }
        assert expected_keys.issubset(details.keys())


# ---------------------------------------------------------------------------
# Actionability
# ---------------------------------------------------------------------------


class TestScoreActionability:
    def test_text_with_recommendations(self) -> None:
        """Text with many actionable keywords scores higher."""
        text = (
            "## Practical Recommendations\n\n"
            "Teams should adopt this approach. "
            "We recommend using the framework. "
            "Consider implementing caching. "
            "This suggests applying batching. "
            "Practitioners should implement monitoring."
        )
        score, details = score_actionability(text)
        assert 0.0 <= score <= 1.0
        assert score > 0.5
        assert details["actionable_keyword_total"] >= 5
        assert details["has_practical_recommendations_section"] is True

    def test_text_lacking_actionable_content(self) -> None:
        """Descriptive text without recommendations scores low."""
        text = (
            "## Introduction\n\n"
            "The field of natural language processing has evolved rapidly. "
            "Many models have been proposed over the years. "
            "Datasets are available for benchmarking."
        )
        score, details = score_actionability(text)
        assert 0.0 <= score <= 1.0
        assert score < 0.3
        assert details["has_practical_recommendations_section"] is False

    def test_findings_with_numbers(self) -> None:
        """Lines with percentages and measurements boost the score."""
        text = (
            "## Practical Recommendations\n\n"
            "Performance improved by 15%. "
            "The model achieved 92.5% accuracy. "
            "Latency dropped to 45ms."
        )
        score, details = score_actionability(text)
        assert details["findings_line_count"] >= 1

    def test_empty_text(self) -> None:
        """Empty string returns zero actionability."""
        score, _ = score_actionability("")
        assert score == 0.0


# ---------------------------------------------------------------------------
# Comprehensiveness
# ---------------------------------------------------------------------------


class TestScoreComprehensiveness:
    def test_full_report(self) -> None:
        """A complete report should score well on comprehensiveness."""
        text = _make_full_report()
        score, details = score_comprehensiveness(text)
        assert 0.0 <= score <= 1.0
        assert score > 0.3
        assert details["required_sections_present"] >= 7
        assert details["unique_citations"] >= 3

    def test_minimal_text(self) -> None:
        """Very short text with no sections scores low."""
        score, details = score_comprehensiveness("Just a sentence.")
        assert 0.0 <= score <= 1.0
        assert score < 0.2
        assert details["section_count"] == 0
        assert details["citation_count"] == 0

    def test_word_count_target(self) -> None:
        """Reports in the 3000-15000 word range get full word_score."""
        # Build a ~4000 word text
        paragraph = "This is a test sentence for padding purposes. " * 20 + "\n\n"
        text = "# Report\n\n" + paragraph * 20
        _, details = score_comprehensiveness(text)
        assert details["word_score"] == 1.0

    def test_empty_text(self) -> None:
        """Empty string returns zero comprehensiveness."""
        score, _ = score_comprehensiveness("")
        assert score == 0.0


# ---------------------------------------------------------------------------
# Evidence
# ---------------------------------------------------------------------------


class TestScoreEvidence:
    def test_citation_rich_text(self) -> None:
        """Text with many citations, blockquotes, and confidence scores high."""
        text = (
            "## Evidence Map\n\n"
            "Finding one [2401.12345] confirms the hypothesis. "
            "Finding two [2401.67890] supports this. "
            "Finding three [2401.11111] adds nuance. "
            "Finding four [2401.22222] provides context. "
            "Finding five [2401.33333] disagrees.\n\n"
            "> Evidence: 95% accuracy reported.\n"
            "> Evidence: Latency reduced by 40%.\n"
            "> Evidence: Cost savings of 30%.\n\n"
            "🟢 High confidence: Main finding.\n"
            "🟡 Medium confidence: Secondary finding.\n"
            "🔴 Low confidence: Tentative finding.\n"
            "🟢 High confidence: Another result.\n"
            "🟡 Medium confidence: Related result.\n"
        )
        score, details = score_evidence(text)
        assert 0.0 <= score <= 1.0
        assert score > 0.5
        assert details["citation_count"] >= 5
        assert details["blockquote_count"] >= 3
        assert details["confidence_annotation_count"] >= 5
        assert details["has_evidence_map_section"] is True

    def test_no_citations(self) -> None:
        """Text without any evidence markers scores low."""
        text = (
            "## Discussion\n\n"
            "The results are interesting. "
            "More work is needed in this area. "
            "The community is active."
        )
        score, details = score_evidence(text)
        assert 0.0 <= score <= 1.0
        assert score < 0.2
        assert details["citation_count"] == 0
        assert details["blockquote_count"] == 0
        assert details["has_evidence_map_section"] is False

    def test_evidence_density_calculation(self) -> None:
        """Citation density per 500 words is computed correctly."""
        # 50 words + 5 citations → density = 5/50 * 500 = 50
        words = " ".join(["word"] * 45)
        text = f"{words} [A] [B] [C] [D] [E]"
        _, details = score_evidence(text)
        assert details["citation_density_per_500w"] > 0
        assert details["citation_count"] == 5

    def test_empty_text(self) -> None:
        """Empty string returns zero evidence score."""
        score, _ = score_evidence("")
        assert score == 0.0


# ---------------------------------------------------------------------------
# Composite RACE score
# ---------------------------------------------------------------------------


class TestComputeRaceScore:
    def test_produces_valid_race_score(self) -> None:
        """compute_race_score returns a valid RACEScore."""
        text = _make_full_report()
        race = compute_race_score(text)
        assert isinstance(race, RACEScore)
        assert 0.0 <= race.readability <= 1.0
        assert 0.0 <= race.actionability <= 1.0
        assert 0.0 <= race.comprehensiveness <= 1.0
        assert 0.0 <= race.evidence <= 1.0
        assert 0.0 <= race.overall <= 1.0

    def test_overall_is_average(self) -> None:
        """Overall score must be the mean of the four dimensions."""
        text = _make_full_report()
        race = compute_race_score(text)
        expected = (
            race.readability
            + race.actionability
            + race.comprehensiveness
            + race.evidence
        ) / 4
        assert math.isclose(race.overall, round(expected, 4), abs_tol=1e-4)

    def test_details_contains_all_dimensions(self) -> None:
        """Details dict should have all four dimension keys."""
        race = compute_race_score("Some text.")
        assert "readability" in race.details
        assert "actionability" in race.details
        assert "comprehensiveness" in race.details
        assert "evidence" in race.details

    def test_empty_text_zero_overall(self) -> None:
        """Empty string should yield zero overall."""
        race = compute_race_score("")
        assert race.overall == 0.0


# ---------------------------------------------------------------------------
# Model serialization
# ---------------------------------------------------------------------------


class TestRACEScoreModel:
    def test_serialization_roundtrip(self) -> None:
        """RACEScore can be serialized and deserialized."""
        original = RACEScore(
            readability=0.75,
            actionability=0.60,
            comprehensiveness=0.80,
            evidence=0.55,
            overall=0.675,
            details={"readability": {"word_count": 500}},
        )
        dumped = original.model_dump()
        restored = RACEScore(**dumped)
        assert restored == original

    def test_json_roundtrip(self) -> None:
        """RACEScore survives JSON serialization."""
        original = RACEScore(
            readability=0.5,
            actionability=0.5,
            comprehensiveness=0.5,
            evidence=0.5,
            overall=0.5,
            details={},
        )
        json_str = original.model_dump_json()
        restored = RACEScore.model_validate_json(json_str)
        assert restored == original

    def test_bounds_enforced(self) -> None:
        """Scores outside [0, 1] should be rejected by Pydantic."""
        with pytest.raises(ValidationError):
            RACEScore(
                readability=1.5,
                actionability=0.5,
                comprehensiveness=0.5,
                evidence=0.5,
                overall=0.5,
                details={},
            )

    def test_model_dump_keys(self) -> None:
        """model_dump should include all fields."""
        race = compute_race_score("Test text.")
        dumped = race.model_dump()
        assert set(dumped.keys()) == {
            "readability",
            "actionability",
            "comprehensiveness",
            "evidence",
            "overall",
            "details",
        }
