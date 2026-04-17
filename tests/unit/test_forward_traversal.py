"""Tests for sources.forward_traversal — forward citation traversal."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from research_pipeline.sources.forward_traversal import (
    CitingPaper,
    ForwardCitationTraverser,
    ForwardStopReason,
    ForwardTraversalConfig,
    ForwardTraversalResult,
    TraversalRound,
    _bm25_simple,
    composite_score,
    forward_expand,
    influence_score,
    recency_bonus,
)

# ---------------------------------------------------------------------------
# Scoring functions
# ---------------------------------------------------------------------------


class TestRecencyBonus:
    def test_current_year(self) -> None:
        year = datetime.now(UTC).year
        assert recency_bonus(year) == pytest.approx(1.0)

    def test_half_life_ago(self) -> None:
        year = datetime.now(UTC).year - 3
        assert recency_bonus(year, half_life=3) == pytest.approx(0.5, abs=0.01)

    def test_old_paper(self) -> None:
        assert recency_bonus(1990) < 0.01

    def test_none_year(self) -> None:
        assert recency_bonus(None) == 0.0

    def test_zero_half_life(self) -> None:
        assert recency_bonus(2024, half_life=0) == 0.0

    def test_future_year(self) -> None:
        year = datetime.now(UTC).year + 5
        assert recency_bonus(year) == pytest.approx(1.0)


class TestInfluenceScore:
    def test_single_seed(self) -> None:
        assert influence_score(1, 5) == pytest.approx(0.2)

    def test_all_seeds(self) -> None:
        assert influence_score(5, 5) == pytest.approx(1.0)

    def test_more_than_total(self) -> None:
        assert influence_score(10, 5) == 1.0

    def test_zero_total(self) -> None:
        assert influence_score(1, 0) == 0.0


class TestCompositeScore:
    def test_all_ones(self) -> None:
        assert composite_score(1.0, 1.0, 1.0) == pytest.approx(1.0)

    def test_all_zeros(self) -> None:
        assert composite_score(0.0, 0.0, 0.0) == pytest.approx(0.0)

    def test_relevance_dominant(self) -> None:
        cfg = ForwardTraversalConfig(
            relevance_weight=0.9, recency_weight=0.05, influence_weight=0.05
        )
        score = composite_score(1.0, 0.0, 0.0, cfg)
        assert score > 0.8

    def test_zero_weights(self) -> None:
        cfg = ForwardTraversalConfig(
            relevance_weight=0.0, recency_weight=0.0, influence_weight=0.0
        )
        assert composite_score(1.0, 1.0, 1.0, cfg) == 0.0


class TestBm25Simple:
    def test_exact_match(self) -> None:
        assert _bm25_simple("knowledge graph", ["knowledge", "graph"]) > 0.5

    def test_no_match(self) -> None:
        assert _bm25_simple("quantum physics", ["knowledge", "graph"]) == 0.0

    def test_empty_text(self) -> None:
        assert _bm25_simple("", ["test"]) == 0.0

    def test_empty_terms(self) -> None:
        assert _bm25_simple("hello world", []) == 0.0

    def test_partial_match(self) -> None:
        score = _bm25_simple("knowledge base system", ["knowledge", "graph"])
        assert 0.0 < score < 1.0


# ---------------------------------------------------------------------------
# CitingPaper
# ---------------------------------------------------------------------------


class TestCitingPaper:
    def test_to_dict(self) -> None:
        cp = CitingPaper(
            paper_id="p1",
            title="Test Paper",
            year=2024,
            cited_seed_ids=["s1", "s2"],
            relevance_score=0.8,
            recency_bonus=0.9,
            influence_score=0.5,
            composite_score=0.75,
        )
        d = cp.to_dict()
        assert d["paper_id"] == "p1"
        assert d["cited_seed_count"] == 2
        assert d["composite_score"] == 0.75


# ---------------------------------------------------------------------------
# TraversalRound
# ---------------------------------------------------------------------------


class TestTraversalRound:
    def test_to_dict(self) -> None:
        tr = TraversalRound(
            depth=1,
            seeds_used=3,
            fetched=10,
            new_unique=7,
            mean_score=0.6,
            yield_ratio=0.7,
        )
        d = tr.to_dict()
        assert d["depth"] == 1
        assert d["yield_ratio"] == 0.7


# ---------------------------------------------------------------------------
# ForwardTraversalResult
# ---------------------------------------------------------------------------


class TestForwardTraversalResult:
    def test_empty(self) -> None:
        r = ForwardTraversalResult(seed_ids=["s1"])
        assert r.total_discovered == 0
        assert r.top_papers(5) == []

    def test_top_papers(self) -> None:
        r = ForwardTraversalResult(
            seed_ids=["s1"],
            discovered=[
                CitingPaper(paper_id="p1", composite_score=0.9),
                CitingPaper(paper_id="p2", composite_score=0.5),
                CitingPaper(paper_id="p3", composite_score=0.7),
            ],
        )
        top = r.top_papers(2)
        assert len(top) == 2
        assert top[0].paper_id == "p1"
        assert top[1].paper_id == "p3"

    def test_to_dict(self) -> None:
        r = ForwardTraversalResult(seed_ids=["s1"])
        d = r.to_dict()
        assert d["total_discovered"] == 0
        assert d["stop_reason"] == "max_depth"


# ---------------------------------------------------------------------------
# ForwardCitationTraverser
# ---------------------------------------------------------------------------


def _make_fetch_fn(
    data: dict[str, list[dict[str, Any]]],
) -> Any:
    """Create a fetch function from a static dict of paper_id → citing papers."""

    def fetch(paper_id: str, limit: int) -> list[dict[str, Any]]:
        return data.get(paper_id, [])[:limit]

    return fetch


class TestForwardCitationTraverser:
    def test_dry_run(self) -> None:
        traverser = ForwardCitationTraverser()
        result = traverser.traverse(
            seed_ids=["s1"],
            query_terms=["test"],
        )
        assert result.total_discovered == 0

    def test_single_depth(self) -> None:
        data = {
            "s1": [
                {
                    "paper_id": "p1",
                    "title": "Knowledge Graph Analysis",
                    "abstract": "A study of knowledge graphs",
                    "year": 2024,
                },
                {
                    "paper_id": "p2",
                    "title": "Neural Networks",
                    "abstract": "Deep learning",
                    "year": 2023,
                },
            ],
        }
        cfg = ForwardTraversalConfig(max_depth=1, limit_per_seed=10)
        traverser = ForwardCitationTraverser(config=cfg)
        result = traverser.traverse(
            seed_ids=["s1"],
            query_terms=["knowledge", "graph"],
            fetch_fn=_make_fetch_fn(data),
        )
        assert result.total_discovered == 2
        assert len(result.rounds) == 1
        assert result.rounds[0].depth == 1
        assert result.total_api_calls == 1

    def test_multi_depth(self) -> None:
        current_year = datetime.now(UTC).year
        data = {
            "s1": [
                {
                    "paper_id": "p1",
                    "title": "Graph Methods",
                    "abstract": "graph analysis",
                    "year": current_year,
                },
            ],
            "p1": [
                {
                    "paper_id": "p2",
                    "title": "Deep Graph",
                    "abstract": "deep graph nets",
                    "year": current_year,
                },
            ],
        }
        cfg = ForwardTraversalConfig(max_depth=3, limit_per_seed=10)
        traverser = ForwardCitationTraverser(config=cfg)
        result = traverser.traverse(
            seed_ids=["s1"],
            query_terms=["graph"],
            fetch_fn=_make_fetch_fn(data),
        )
        assert result.total_discovered == 2
        assert len(result.rounds) >= 2

    def test_dedup(self) -> None:
        data = {
            "s1": [
                {
                    "paper_id": "p1",
                    "title": "Paper One",
                    "abstract": "test",
                    "year": 2024,
                },
            ],
            "s2": [
                {
                    "paper_id": "p1",
                    "title": "Paper One",
                    "abstract": "test",
                    "year": 2024,
                },
                {
                    "paper_id": "p2",
                    "title": "Paper Two",
                    "abstract": "test",
                    "year": 2024,
                },
            ],
        }
        cfg = ForwardTraversalConfig(max_depth=1, limit_per_seed=10)
        traverser = ForwardCitationTraverser(config=cfg)
        result = traverser.traverse(
            seed_ids=["s1", "s2"],
            query_terms=["test"],
            fetch_fn=_make_fetch_fn(data),
        )
        assert result.total_discovered == 2  # p1 + p2 (deduped)

    def test_max_papers_stop(self) -> None:
        data = {
            "s1": [
                {
                    "paper_id": f"p{i}",
                    "title": f"Paper {i}",
                    "abstract": "test",
                    "year": 2024,
                }
                for i in range(20)
            ],
        }
        cfg = ForwardTraversalConfig(max_depth=3, max_papers=5, limit_per_seed=50)
        traverser = ForwardCitationTraverser(config=cfg)
        result = traverser.traverse(
            seed_ids=["s1"],
            query_terms=["test"],
            fetch_fn=_make_fetch_fn(data),
        )
        assert result.total_discovered <= 5
        assert result.stop_reason == ForwardStopReason.MAX_PAPERS

    def test_no_new_papers_stop(self) -> None:
        data = {
            "s1": [
                {
                    "paper_id": "p1",
                    "title": "Paper One",
                    "abstract": "graph",
                    "year": 2024,
                },
            ],
            "p1": [],  # no new citing papers
        }
        cfg = ForwardTraversalConfig(max_depth=5, limit_per_seed=10)
        traverser = ForwardCitationTraverser(config=cfg)
        result = traverser.traverse(
            seed_ids=["s1"],
            query_terms=["graph"],
            fetch_fn=_make_fetch_fn(data),
        )
        assert result.stop_reason == ForwardStopReason.NO_NEW_PAPERS

    def test_yield_decay_stop(self) -> None:
        # Round 1: lots of new papers. Round 2+: almost all dupes
        all_papers = [
            {
                "paper_id": f"p{i}",
                "title": f"Paper {i}",
                "abstract": "test",
                "year": 2024,
            }
            for i in range(10)
        ]
        data = {
            "s1": all_papers[:5],
        }
        # p0-p4 will be seeds for depth 2, but they all return the same papers (dupes)
        for i in range(5):
            data[f"p{i}"] = all_papers[:5]  # all dupes

        cfg = ForwardTraversalConfig(
            max_depth=5,
            limit_per_seed=10,
            yield_decay_threshold=0.1,
            yield_patience=1,
        )
        traverser = ForwardCitationTraverser(config=cfg)
        result = traverser.traverse(
            seed_ids=["s1"],
            query_terms=["test"],
            fetch_fn=_make_fetch_fn(data),
        )
        assert result.stop_reason in (
            ForwardStopReason.YIELD_DECAY,
            ForwardStopReason.NO_NEW_PAPERS,
        )

    def test_fetch_error_handled(self) -> None:
        def bad_fetch(paper_id: str, limit: int) -> list[dict[str, Any]]:
            raise RuntimeError("API Error")

        cfg = ForwardTraversalConfig(max_depth=1)
        traverser = ForwardCitationTraverser(config=cfg)
        result = traverser.traverse(
            seed_ids=["s1"],
            query_terms=["test"],
            fetch_fn=bad_fetch,
        )
        assert result.total_discovered == 0

    def test_seed_not_in_discovered(self) -> None:
        data = {
            "s1": [
                {
                    "paper_id": "s1",
                    "title": "Self Citation",
                    "abstract": "test",
                    "year": 2024,
                },
                {"paper_id": "p1", "title": "Other", "abstract": "test", "year": 2024},
            ],
        }
        cfg = ForwardTraversalConfig(max_depth=1)
        traverser = ForwardCitationTraverser(config=cfg)
        result = traverser.traverse(
            seed_ids=["s1"],
            query_terms=["test"],
            fetch_fn=_make_fetch_fn(data),
        )
        # s1 should be excluded (it's a seed)
        ids = {p.paper_id for p in result.discovered}
        assert "s1" not in ids
        assert "p1" in ids


# ---------------------------------------------------------------------------
# forward_expand convenience function
# ---------------------------------------------------------------------------


class TestForwardExpand:
    def test_basic(self) -> None:
        data = {
            "s1": [
                {
                    "paper_id": "p1",
                    "title": "Graph Paper",
                    "abstract": "knowledge graph",
                    "year": 2024,
                },
            ],
        }
        result = forward_expand(
            seed_ids=["s1"],
            query_terms=["graph"],
            fetch_fn=_make_fetch_fn(data),
            config=ForwardTraversalConfig(max_depth=1),
        )
        assert result.total_discovered == 1

    def test_no_fetch_fn(self) -> None:
        result = forward_expand(seed_ids=["s1"], query_terms=["test"])
        assert result.total_discovered == 0


# ---------------------------------------------------------------------------
# ForwardTraversalConfig
# ---------------------------------------------------------------------------


class TestForwardTraversalConfig:
    def test_defaults(self) -> None:
        cfg = ForwardTraversalConfig()
        assert cfg.max_depth == 3
        assert cfg.max_papers == 200
        assert cfg.relevance_weight == 0.5

    def test_custom(self) -> None:
        cfg = ForwardTraversalConfig(max_depth=5, max_papers=100)
        assert cfg.max_depth == 5
        assert cfg.max_papers == 100
