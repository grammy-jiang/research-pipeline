"""Unit tests for quality evaluation modules."""

from datetime import UTC, datetime

import pytest

from research_pipeline.models.candidate import CandidateRecord
from research_pipeline.models.quality import QualityScore
from research_pipeline.quality.author_metrics import author_credibility
from research_pipeline.quality.citation_metrics import (
    citation_impact,
    citation_velocity,
    compute_citation_metrics,
)
from research_pipeline.quality.composite import compute_quality_score
from research_pipeline.quality.venue_scoring import (
    get_venue_tier,
    reset_venue_cache,
    venue_score,
)


def _make_candidate(**kwargs) -> CandidateRecord:  # type: ignore[no-untyped-def]
    defaults = {
        "arxiv_id": "2401.12345",
        "version": "v1",
        "title": "Test Paper",
        "abstract": "Test abstract",
        "authors": ["Author One"],
        "published": datetime(2024, 1, 1, tzinfo=UTC),
        "updated": datetime(2024, 1, 1, tzinfo=UTC),
        "primary_category": "cs.AI",
        "categories": ["cs.AI"],
        "pdf_url": "https://arxiv.org/pdf/2401.12345",
        "abs_url": "https://arxiv.org/abs/2401.12345",
    }
    defaults.update(kwargs)
    return CandidateRecord(**defaults)


class TestCitationMetrics:
    """Tests for citation_metrics module."""

    def test_zero_citations(self) -> None:
        assert citation_impact(0) == 0.0

    def test_none_citations(self) -> None:
        assert citation_impact(None) == 0.0

    def test_high_citations(self) -> None:
        score = citation_impact(1000)
        assert score == pytest.approx(1.0, abs=0.01)

    def test_moderate_citations(self) -> None:
        score = citation_impact(50)
        assert 0.0 < score < 1.0

    def test_monotonic(self) -> None:
        s10 = citation_impact(10)
        s100 = citation_impact(100)
        s1000 = citation_impact(1000)
        assert s10 < s100 < s1000

    def test_velocity_with_published(self) -> None:
        vel = citation_velocity(
            100,
            published=datetime(2022, 1, 1, tzinfo=UTC),
        )
        assert vel > 0

    def test_velocity_none_citations(self) -> None:
        assert citation_velocity(None) == 0.0

    def test_velocity_no_date(self) -> None:
        assert citation_velocity(100) == 0.0

    def test_compute_citation_metrics_full(self) -> None:
        candidate = _make_candidate(
            citation_count=100,
            influential_citation_count=20,
        )
        metrics = compute_citation_metrics(candidate)
        assert "citation_impact" in metrics
        assert "citation_velocity" in metrics
        assert "influential_ratio" in metrics
        assert metrics["influential_ratio"] == pytest.approx(0.2, abs=0.01)


class TestVenueScoring:
    """Tests for venue_scoring module."""

    def setup_method(self) -> None:
        reset_venue_cache()

    def test_known_a_star_venue(self) -> None:
        tier = get_venue_tier("NeurIPS")
        assert tier == "A*"

    def test_known_a_venue(self) -> None:
        tier = get_venue_tier("AISTATS")
        assert tier == "A"

    def test_known_b_venue(self) -> None:
        tier = get_venue_tier("LREC")
        assert tier == "B"

    def test_case_insensitive(self) -> None:
        tier = get_venue_tier("neurips")
        assert tier == "A*"

    def test_unknown_venue(self) -> None:
        tier = get_venue_tier("Unknown Workshop 2024")
        assert tier is None

    def test_empty_venue(self) -> None:
        tier = get_venue_tier("")
        assert tier is None

    def test_venue_score_a_star(self) -> None:
        score = venue_score("NeurIPS")
        assert score == 1.0

    def test_venue_score_unknown(self) -> None:
        score = venue_score("Random Workshop")
        assert score == 0.1

    def test_venue_score_none(self) -> None:
        score = venue_score(None)
        assert score == 0.1

    def test_venue_score_preprint(self) -> None:
        score = venue_score("")
        assert score == 0.1


class TestAuthorMetrics:
    """Tests for author_metrics module."""

    def test_zero_h_index(self) -> None:
        assert author_credibility(0) == 0.0

    def test_none_h_index(self) -> None:
        assert author_credibility(None) == 0.0

    def test_high_h_index(self) -> None:
        score = author_credibility(100)
        assert score == pytest.approx(1.0, abs=0.01)

    def test_moderate_h_index(self) -> None:
        score = author_credibility(30)
        assert 0.0 < score < 1.0

    def test_monotonic(self) -> None:
        s5 = author_credibility(5)
        s30 = author_credibility(30)
        s100 = author_credibility(100)
        assert s5 < s30 < s100


class TestCompositeScore:
    """Tests for composite quality scoring."""

    def test_basic_composite(self) -> None:
        candidate = _make_candidate(
            citation_count=50,
            venue="NeurIPS",
        )
        qs = compute_quality_score(candidate, max_h_index=40)
        assert isinstance(qs, QualityScore)
        assert 0.0 <= qs.composite_score <= 1.0
        assert qs.paper_id == "2401.12345"

    def test_no_metadata(self) -> None:
        candidate = _make_candidate()
        qs = compute_quality_score(candidate)
        assert isinstance(qs, QualityScore)
        assert qs.citation_impact == 0.0
        assert qs.author_credibility == 0.0

    def test_high_quality_paper(self) -> None:
        candidate = _make_candidate(
            citation_count=500,
            venue="ICML",
        )
        qs = compute_quality_score(candidate, max_h_index=80)
        assert qs.composite_score > 0.5

    def test_custom_weights(self) -> None:
        candidate = _make_candidate(citation_count=100)
        w = {
            "citation_weight": 1.0,
            "venue_weight": 0.0,
            "author_weight": 0.0,
            "recency_weight": 0.0,
        }
        qs = compute_quality_score(candidate, weights=w)
        assert qs.composite_score == pytest.approx(qs.citation_impact, abs=0.01)

    def test_score_roundtrip(self) -> None:
        candidate = _make_candidate(citation_count=10, venue="AAAI")
        qs = compute_quality_score(candidate, max_h_index=25)
        data = qs.model_dump(mode="json")
        qs2 = QualityScore(**data)
        assert qs2.composite_score == qs.composite_score
