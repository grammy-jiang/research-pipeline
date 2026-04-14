"""Tests for confidence-gated retrieval depth classification."""

from datetime import UTC, datetime

from research_pipeline.models.candidate import CandidateRecord
from research_pipeline.models.screening import CheapScoreBreakdown, RelevanceDecision
from research_pipeline.screening.depth_gate import (
    classify_retrieval_depth,
    papers_needing_expansion,
)


def _make_decision(arxiv_id: str, score: float) -> RelevanceDecision:
    candidate = CandidateRecord(
        arxiv_id=arxiv_id,
        version="v1",
        title=f"Paper {arxiv_id}",
        authors=["A"],
        published=datetime(2024, 1, 1, tzinfo=UTC),
        updated=datetime(2024, 1, 1, tzinfo=UTC),
        categories=["cs.AI"],
        primary_category="cs.AI",
        abstract="Test abstract",
        abs_url="",
        pdf_url="",
        source="test",
    )
    cheap = CheapScoreBreakdown(
        bm25_title=0.0,
        bm25_abstract=0.0,
        cat_match=0.0,
        negative_penalty=0.0,
        recency_bonus=0.0,
        cheap_score=score,
    )
    return RelevanceDecision(
        paper=candidate,
        cheap=cheap,
        final_score=score,
        download=True,
        download_reason="score_threshold",
    )


class TestClassifyRetrievalDepth:
    """Tests for classify_retrieval_depth."""

    def test_high_score_classified_as_high_tier(self) -> None:
        decisions = [_make_decision("2401.0001", 0.90)]
        tiers = classify_retrieval_depth(decisions)
        assert len(tiers) == 1
        assert tiers[0].tier == "high"
        assert tiers[0].expand_depth == 0

    def test_medium_score_classified_as_medium_tier(self) -> None:
        decisions = [_make_decision("2401.0002", 0.70)]
        tiers = classify_retrieval_depth(decisions)
        assert len(tiers) == 1
        assert tiers[0].tier == "medium"
        assert tiers[0].expand_depth == 1

    def test_low_score_classified_as_low_tier(self) -> None:
        decisions = [_make_decision("2401.0003", 0.30)]
        tiers = classify_retrieval_depth(decisions)
        assert len(tiers) == 1
        assert tiers[0].tier == "low"
        assert tiers[0].expand_depth == 2

    def test_boundary_high_threshold_is_high(self) -> None:
        decisions = [_make_decision("2401.0004", 0.85)]
        tiers = classify_retrieval_depth(decisions)
        assert tiers[0].tier == "high"

    def test_boundary_low_threshold_is_medium(self) -> None:
        decisions = [_make_decision("2401.0005", 0.50)]
        tiers = classify_retrieval_depth(decisions)
        assert tiers[0].tier == "medium"

    def test_custom_thresholds(self) -> None:
        decisions = [_make_decision("2401.0006", 0.75)]
        tiers = classify_retrieval_depth(
            decisions, high_threshold=0.90, low_threshold=0.60
        )
        assert tiers[0].tier == "medium"

        tiers = classify_retrieval_depth(
            decisions, high_threshold=0.70, low_threshold=0.60
        )
        assert tiers[0].tier == "high"

    def test_custom_depths(self) -> None:
        decisions = [
            _make_decision("2401.0010", 0.95),
            _make_decision("2401.0011", 0.70),
            _make_decision("2401.0012", 0.30),
        ]
        tiers = classify_retrieval_depth(
            decisions, high_depth=1, medium_depth=3, low_depth=5
        )
        assert tiers[0].expand_depth == 1
        assert tiers[1].expand_depth == 3
        assert tiers[2].expand_depth == 5

    def test_empty_list_returns_empty(self) -> None:
        tiers = classify_retrieval_depth([])
        assert tiers == []

    def test_mixed_list_counts_correctly(self) -> None:
        decisions = [
            _make_decision("2401.0020", 0.95),  # high
            _make_decision("2401.0021", 0.90),  # high
            _make_decision("2401.0022", 0.70),  # medium
            _make_decision("2401.0023", 0.60),  # medium
            _make_decision("2401.0024", 0.55),  # medium
            _make_decision("2401.0025", 0.30),  # low
        ]
        tiers = classify_retrieval_depth(decisions)
        tier_names = [t.tier for t in tiers]
        assert tier_names.count("high") == 2
        assert tier_names.count("medium") == 3
        assert tier_names.count("low") == 1

    def test_final_score_is_rounded(self) -> None:
        decisions = [_make_decision("2401.0030", 0.123456789)]
        tiers = classify_retrieval_depth(decisions)
        assert tiers[0].final_score == 0.1235

    def test_arxiv_id_preserved(self) -> None:
        decisions = [_make_decision("2401.99999", 0.70)]
        tiers = classify_retrieval_depth(decisions)
        assert tiers[0].arxiv_id == "2401.99999"


class TestPapersNeedingExpansion:
    """Tests for papers_needing_expansion."""

    def test_filters_out_high_tier(self) -> None:
        decisions = [
            _make_decision("2401.0040", 0.95),  # high → depth 0
            _make_decision("2401.0041", 0.70),  # medium → depth 1
            _make_decision("2401.0042", 0.30),  # low → depth 2
        ]
        tiers = classify_retrieval_depth(decisions)
        result = papers_needing_expansion(tiers)
        ids = [r[0] for r in result]
        assert "2401.0040" not in ids
        assert "2401.0041" in ids
        assert "2401.0042" in ids

    def test_min_depth_2_only_returns_low(self) -> None:
        decisions = [
            _make_decision("2401.0050", 0.95),  # high → depth 0
            _make_decision("2401.0051", 0.70),  # medium → depth 1
            _make_decision("2401.0052", 0.30),  # low → depth 2
        ]
        tiers = classify_retrieval_depth(decisions)
        result = papers_needing_expansion(tiers, min_depth=2)
        assert len(result) == 1
        assert result[0] == ("2401.0052", 2)

    def test_empty_tiers_returns_empty(self) -> None:
        result = papers_needing_expansion([])
        assert result == []

    def test_all_high_returns_empty(self) -> None:
        decisions = [
            _make_decision("2401.0060", 0.90),
            _make_decision("2401.0061", 0.95),
        ]
        tiers = classify_retrieval_depth(decisions)
        result = papers_needing_expansion(tiers)
        assert result == []
