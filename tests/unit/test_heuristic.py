"""Unit tests for screening.heuristic module."""

from datetime import UTC, datetime, timedelta

from research_pipeline.models.candidate import CandidateRecord
from research_pipeline.screening.heuristic import score_candidates, select_topk


def _make_candidate(
    arxiv_id: str = "2401.12345",
    title: str = "Neural Information Retrieval",
    abstract: str = "This paper explores neural approaches to information retrieval.",
    categories: list[str] | None = None,
    primary_category: str = "cs.IR",
    published: datetime | None = None,
) -> CandidateRecord:
    if categories is None:
        categories = ["cs.IR"]
    if published is None:
        published = datetime.now(UTC) - timedelta(days=30)
    return CandidateRecord(
        arxiv_id=arxiv_id,
        version="v1",
        title=title,
        authors=["Author"],
        published=published,
        updated=published,
        categories=categories,
        primary_category=primary_category,
        abstract=abstract,
        abs_url=f"https://arxiv.org/abs/{arxiv_id}",
        pdf_url=f"https://arxiv.org/pdf/{arxiv_id}",
    )


class TestScoreCandidates:
    def test_returns_correct_count(self) -> None:
        candidates = [
            _make_candidate("1", title="Neural search"),
            _make_candidate("2", title="Quantum computing"),
        ]
        scores = score_candidates(
            candidates,
            must_terms=["neural"],
            nice_terms=["search"],
            negative_terms=[],
            target_categories=["cs.IR"],
        )
        assert len(scores) == 2

    def test_relevant_scores_higher(self) -> None:
        candidates = [
            _make_candidate(
                "1",
                title="Neural information retrieval with embeddings",
                abstract="Dense retrieval using neural embeddings for search.",
            ),
            _make_candidate(
                "2",
                title="Quantum error correction codes",
                abstract="We study error correction in quantum computing systems.",
                categories=["quant-ph"],
                primary_category="quant-ph",
            ),
        ]
        scores = score_candidates(
            candidates,
            must_terms=["neural", "retrieval"],
            nice_terms=["embedding"],
            negative_terms=[],
            target_categories=["cs.IR"],
        )
        assert scores[0].cheap_score > scores[1].cheap_score

    def test_negative_penalty_applied(self) -> None:
        candidates = [
            _make_candidate(
                "1",
                title="Neural search survey review",
                abstract="A comprehensive survey of neural search methods.",
            ),
            _make_candidate(
                "2",
                title="Neural search methods",
                abstract="Novel neural search techniques for dense retrieval.",
            ),
        ]
        scores = score_candidates(
            candidates,
            must_terms=["neural", "search"],
            nice_terms=[],
            negative_terms=["survey"],
            target_categories=[],
        )
        # The candidate with "survey" in title and abstract should have negative penalty
        assert scores[0].negative_penalty > scores[1].negative_penalty

    def test_category_match_bonus(self) -> None:
        candidates = [
            _make_candidate(
                "1",
                title="Test paper on ML",
                abstract="Machine learning paper.",
                primary_category="cs.IR",
                categories=["cs.IR", "cs.LG"],
            ),
            _make_candidate(
                "2",
                title="Test paper on ML",
                abstract="Machine learning paper.",
                primary_category="stat.ML",
                categories=["stat.ML"],
            ),
        ]
        scores = score_candidates(
            candidates,
            must_terms=["ML"],
            nice_terms=[],
            negative_terms=[],
            target_categories=["cs.IR"],
        )
        assert scores[0].cat_match == 1.0
        assert scores[1].cat_match == 0.0

    def test_secondary_category_match(self) -> None:
        candidates = [
            _make_candidate(
                "1",
                title="Test",
                abstract="Test abstract.",
                primary_category="cs.LG",
                categories=["cs.LG", "cs.IR"],
            ),
        ]
        scores = score_candidates(
            candidates,
            must_terms=["test"],
            nice_terms=[],
            negative_terms=[],
            target_categories=["cs.IR"],
        )
        assert scores[0].cat_match == 0.5

    def test_recency_bonus_recent_higher(self) -> None:
        recent = datetime.now(UTC) - timedelta(days=7)
        old = datetime.now(UTC) - timedelta(days=365)
        candidates = [
            _make_candidate("1", title="Test paper", published=recent),
            _make_candidate("2", title="Test paper", published=old),
        ]
        scores = score_candidates(
            candidates,
            must_terms=["test"],
            nice_terms=[],
            negative_terms=[],
            target_categories=[],
        )
        assert scores[0].recency_bonus > scores[1].recency_bonus

    def test_scores_bounded_zero_one(self) -> None:
        candidates = [
            _make_candidate("1"),
            _make_candidate("2", title="Totally irrelevant paper about cooking"),
        ]
        scores = score_candidates(
            candidates,
            must_terms=["neural"],
            nice_terms=[],
            negative_terms=["cooking"],
            target_categories=[],
        )
        for s in scores:
            assert 0.0 <= s.cheap_score <= 1.0

    def test_empty_candidates(self) -> None:
        scores = score_candidates(
            [],
            must_terms=["neural"],
            nice_terms=[],
            negative_terms=[],
            target_categories=[],
        )
        assert scores == []


class TestSelectTopk:
    def test_selects_top_k(self) -> None:
        candidates = [
            _make_candidate("1", title="Neural retrieval methods"),
            _make_candidate("2", title="Neural embedding search"),
            _make_candidate("3", title="Quantum computing theory"),
        ]
        scores = score_candidates(
            candidates,
            must_terms=["neural"],
            nice_terms=["retrieval"],
            negative_terms=[],
            target_categories=["cs.IR"],
        )
        selected = select_topk(candidates, scores, top_k=2)
        assert len(selected) <= 2

    def test_min_score_filter(self) -> None:
        candidates = [
            _make_candidate("1", title="Neural retrieval"),
        ]
        scores = score_candidates(
            candidates,
            must_terms=["neural"],
            nice_terms=[],
            negative_terms=[],
            target_categories=[],
        )
        # Filter with impossibly high threshold
        selected = select_topk(candidates, scores, top_k=10, min_score=999.0)
        assert len(selected) == 0

    def test_sorted_descending(self) -> None:
        candidates = [
            _make_candidate("1", title="Neural retrieval methods for search"),
            _make_candidate("2", title="Cooking recipes for beginners"),
            _make_candidate("3", title="Neural network training optimization"),
        ]
        scores = score_candidates(
            candidates,
            must_terms=["neural"],
            nice_terms=["retrieval"],
            negative_terms=["cooking"],
            target_categories=["cs.IR"],
        )
        selected = select_topk(candidates, scores, top_k=10)
        for i in range(len(selected) - 1):
            assert selected[i][1].cheap_score >= selected[i + 1][1].cheap_score
