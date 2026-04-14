"""Tests for the safety-as-multiplicative-gate feature.

When a paper is flagged as retracted or fabricated the composite quality
score must be zeroed regardless of other metrics.
"""

from datetime import UTC, datetime

from research_pipeline.models.candidate import CandidateRecord
from research_pipeline.models.quality import QualityScore
from research_pipeline.quality.composite import compute_quality_score


def _make_candidate(**overrides: object) -> CandidateRecord:
    """Build a minimal CandidateRecord with sensible defaults."""
    defaults: dict[str, object] = {
        "arxiv_id": "2401.12345",
        "version": "v1",
        "title": "Test Paper",
        "authors": ["Author A"],
        "published": datetime(2024, 1, 1, tzinfo=UTC),
        "updated": datetime(2024, 1, 1, tzinfo=UTC),
        "categories": ["cs.AI"],
        "primary_category": "cs.AI",
        "abstract": "A test abstract with enough words for scoring.",
        "abs_url": "https://arxiv.org/abs/2401.12345",
        "pdf_url": "https://arxiv.org/pdf/2401.12345",
        "source": "arxiv",
        "citation_count": 100,
    }
    defaults.update(overrides)
    return CandidateRecord(**defaults)  # type: ignore[arg-type]


class TestSafetyGateCleanPaper:
    """A clean paper (safety_flag=None) keeps its normal score."""

    def test_clean_paper_scores_normally(self) -> None:
        candidate = _make_candidate()
        score = compute_quality_score(candidate, safety_flag=None)
        assert score.composite_score > 0.0
        assert score.safety_flag is None


class TestSafetyGateRetracted:
    """A retracted paper must receive a zero composite score."""

    def test_retracted_paper_scores_zero(self) -> None:
        candidate = _make_candidate()
        score = compute_quality_score(candidate, safety_flag="retracted")
        assert score.composite_score == 0.0

    def test_retracted_flag_stored_in_model(self) -> None:
        candidate = _make_candidate()
        score = compute_quality_score(candidate, safety_flag="retracted")
        assert score.safety_flag == "retracted"

    def test_retracted_flag_in_details(self) -> None:
        candidate = _make_candidate()
        score = compute_quality_score(candidate, safety_flag="retracted")
        assert score.details["safety_flag"] == "retracted"

    def test_high_citation_retracted_still_zero(self) -> None:
        candidate = _make_candidate(citation_count=50000)
        score = compute_quality_score(candidate, safety_flag="retracted")
        assert score.composite_score == 0.0


class TestSafetyGateFabricated:
    """A fabricated paper must receive a zero composite score."""

    def test_fabricated_paper_scores_zero(self) -> None:
        candidate = _make_candidate()
        score = compute_quality_score(candidate, safety_flag="fabricated")
        assert score.composite_score == 0.0

    def test_fabricated_flag_stored_in_model(self) -> None:
        candidate = _make_candidate()
        score = compute_quality_score(candidate, safety_flag="fabricated")
        assert score.safety_flag == "fabricated"


class TestQualityScoreModel:
    """Direct model-level tests for the safety_flag field."""

    def test_safety_flag_default_is_none(self) -> None:
        qs = QualityScore(
            paper_id="test",
            citation_impact=0.5,
            venue_score=0.5,
            author_credibility=0.5,
            composite_score=0.5,
        )
        assert qs.safety_flag is None

    def test_safety_flag_roundtrip(self) -> None:
        qs = QualityScore(
            paper_id="test",
            citation_impact=0.5,
            venue_score=0.5,
            author_credibility=0.5,
            composite_score=0.0,
            safety_flag="retracted",
        )
        data = qs.model_dump()
        restored = QualityScore.model_validate(data)
        assert restored.safety_flag == "retracted"
