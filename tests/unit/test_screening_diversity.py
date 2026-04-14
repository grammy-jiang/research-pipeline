"""Tests for screening diversity (heuristic.py select_topk with diversity)."""

from datetime import UTC, datetime

from research_pipeline.models.candidate import CandidateRecord
from research_pipeline.models.screening import CheapScoreBreakdown
from research_pipeline.screening.heuristic import (
    _compute_novelty,
    _diverse_select,
    select_topk,
)


def _make_candidate(
    arxiv_id: str = "2401.00001",
    primary_category: str = "cs.AI",
    source: str = "arxiv",
    year: int = 2024,
) -> CandidateRecord:
    """Create a candidate with configurable diversity dimensions."""
    return CandidateRecord(
        arxiv_id=arxiv_id,
        version="v1",
        title=f"Paper {arxiv_id}",
        authors=["Author A"],
        published=datetime(year, 1, 15, tzinfo=UTC),
        updated=datetime(year, 1, 15, tzinfo=UTC),
        categories=[primary_category],
        primary_category=primary_category,
        abstract=f"Abstract of {arxiv_id}",
        abs_url=f"https://arxiv.org/abs/{arxiv_id}",
        pdf_url=f"https://arxiv.org/pdf/{arxiv_id}",
        source=source,
        year=year,
    )


def _make_score(cheap_score: float) -> CheapScoreBreakdown:
    return CheapScoreBreakdown(
        bm25_title=cheap_score,
        bm25_abstract=cheap_score,
        cat_match=0.5,
        negative_penalty=0.0,
        recency_bonus=0.5,
        cheap_score=cheap_score,
    )


class TestSelectTopkDiversity:
    def test_no_diversity_is_pure_topk(self) -> None:
        """Without diversity, selection is pure top-k by score."""
        candidates = [_make_candidate(f"240{i}.0000{i}") for i in range(5)]
        scores = [_make_score(0.1 * (i + 1)) for i in range(5)]

        result = select_topk(candidates, scores, top_k=3, diversity=False)
        assert len(result) == 3
        # Should be sorted by score descending
        result_scores = [s.cheap_score for _, s in result]
        assert result_scores == sorted(result_scores, reverse=True)

    def test_diversity_promotes_different_categories(self) -> None:
        """Diversity should promote papers from underrepresented categories."""
        # 3 cs.AI papers with high scores, 2 cs.CL papers with slightly lower
        candidates = [
            _make_candidate("2401.00001", primary_category="cs.AI"),
            _make_candidate("2401.00002", primary_category="cs.AI"),
            _make_candidate("2401.00003", primary_category="cs.AI"),
            _make_candidate("2401.00004", primary_category="cs.CL"),
            _make_candidate("2401.00005", primary_category="cs.CL"),
        ]
        scores = [
            _make_score(0.9),
            _make_score(0.85),
            _make_score(0.8),
            _make_score(0.75),
            _make_score(0.7),
        ]

        # With high diversity, cs.CL papers should be promoted
        result = select_topk(
            candidates, scores, top_k=4, diversity=True, diversity_lambda=0.5
        )
        categories = [c.primary_category for c, _ in result]
        assert "cs.CL" in categories

    def test_diversity_with_different_sources(self) -> None:
        """Diversity should promote papers from different sources."""
        candidates = [
            _make_candidate("2401.00001", source="arxiv"),
            _make_candidate("2401.00002", source="arxiv"),
            _make_candidate("2401.00003", source="arxiv"),
            _make_candidate("2401.00004", source="scholar"),
            _make_candidate("2401.00005", source="semantic_scholar"),
        ]
        scores = [
            _make_score(0.9),
            _make_score(0.88),
            _make_score(0.85),
            _make_score(0.7),
            _make_score(0.65),
        ]

        result = select_topk(
            candidates, scores, top_k=4, diversity=True, diversity_lambda=0.4
        )
        sources = {c.source for c, _ in result}
        # Should include more than just arxiv
        assert len(sources) >= 2

    def test_diversity_with_different_years(self) -> None:
        """Diversity should promote temporal spread."""
        candidates = [
            _make_candidate("2401.00001", year=2024),
            _make_candidate("2401.00002", year=2024),
            _make_candidate("2401.00003", year=2024),
            _make_candidate("2401.00004", year=2022),
            _make_candidate("2401.00005", year=2020),
        ]
        scores = [
            _make_score(0.9),
            _make_score(0.88),
            _make_score(0.85),
            _make_score(0.6),
            _make_score(0.5),
        ]

        result = select_topk(
            candidates, scores, top_k=4, diversity=True, diversity_lambda=0.5
        )
        years = {c.year for c, _ in result}
        # Should include more than just 2024
        assert len(years) >= 2

    def test_diversity_lambda_zero_equals_pure_topk(self) -> None:
        """diversity_lambda=0 should behave like pure top-k."""
        candidates = [_make_candidate(f"240{i}.0000{i}") for i in range(5)]
        scores = [_make_score(0.1 * (5 - i)) for i in range(5)]

        result_div = select_topk(
            candidates, scores, top_k=3, diversity=True, diversity_lambda=0.0
        )
        result_pure = select_topk(candidates, scores, top_k=3, diversity=False)

        div_ids = [c.arxiv_id for c, _ in result_div]
        pure_ids = [c.arxiv_id for c, _ in result_pure]
        assert div_ids == pure_ids

    def test_empty_candidates(self) -> None:
        result = select_topk([], [], top_k=5, diversity=True)
        assert result == []

    def test_min_score_filter_applied(self) -> None:
        candidates = [_make_candidate(f"240{i}.0000{i}") for i in range(3)]
        scores = [_make_score(0.1), _make_score(0.5), _make_score(0.9)]

        result = select_topk(
            candidates, scores, top_k=10, min_score=0.4, diversity=True
        )
        assert len(result) == 2


class TestComputeNovelty:
    def test_first_candidate_maximally_novel(self) -> None:
        c = _make_candidate()
        novelty = _compute_novelty(c, {}, {}, {})
        assert novelty == 1.0

    def test_same_category_reduces_novelty(self) -> None:
        c = _make_candidate(primary_category="cs.AI")
        novelty = _compute_novelty(c, {"cs.AI": 5}, {2024: 1}, {"arxiv": 1})
        assert novelty < 1.0

    def test_new_category_is_more_novel(self) -> None:
        c_existing = _make_candidate(primary_category="cs.AI")
        c_new = _make_candidate(primary_category="cs.CL")

        covered = {"cs.AI": 3}
        years = {2024: 3}
        sources = {"arxiv": 3}

        novelty_existing = _compute_novelty(c_existing, covered, years, sources)
        novelty_new = _compute_novelty(c_new, covered, years, sources)
        assert novelty_new > novelty_existing


class TestDiverseSelect:
    def test_selects_correct_count(self) -> None:
        pool = [
            (_make_candidate(f"240{i}.0000{i}"), _make_score(0.5)) for i in range(10)
        ]
        result = _diverse_select(pool, 5, 0.3)
        assert len(result) == 5

    def test_empty_pool(self) -> None:
        assert _diverse_select([], 5, 0.3) == []

    def test_pool_smaller_than_topk(self) -> None:
        pool = [
            (_make_candidate("2401.00001"), _make_score(0.5)),
        ]
        result = _diverse_select(pool, 10, 0.3)
        assert len(result) == 1
