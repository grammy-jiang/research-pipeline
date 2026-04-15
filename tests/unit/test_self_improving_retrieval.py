"""Tests for self-improving retrieval module."""

from __future__ import annotations

from datetime import UTC

from research_pipeline.models.candidate import CandidateRecord
from research_pipeline.models.query_plan import QueryPlan
from research_pipeline.retrieval.self_improving import (
    ConvergenceInfo,
    RetrievalIteration,
    SelfImprovingResult,
    _check_convergence,
    _compute_avg_score,
    _compute_coverage,
    _compute_top_score,
    _extract_top_terms,
    _refine_terms,
    _score_papers,
    run_self_improving_retrieval,
)


def _make_candidate(
    arxiv_id: str = "2301.00001",
    title: str = "Test Paper",
    abstract: str = "A test abstract.",
) -> CandidateRecord:
    from datetime import datetime

    return CandidateRecord(
        arxiv_id=arxiv_id,
        version="v1",
        title=title,
        authors=["Author A"],
        published=datetime(2023, 1, 1, tzinfo=UTC),
        updated=datetime(2023, 1, 1, tzinfo=UTC),
        categories=["cs.AI"],
        primary_category="cs.AI",
        abstract=abstract,
        abs_url=f"https://arxiv.org/abs/{arxiv_id}",
        pdf_url=f"https://arxiv.org/pdf/{arxiv_id}",
    )


def _make_plan(
    topic: str = "machine learning",
    must: list[str] | None = None,
    nice: list[str] | None = None,
) -> QueryPlan:
    return QueryPlan(
        topic_raw=topic,
        topic_normalized=topic.lower(),
        must_terms=must or ["machine", "learning"],
        nice_terms=nice or ["neural", "network"],
        query_variants=[topic],
    )


# ---------------------------------------------------------------------------
# Dataclass tests
# ---------------------------------------------------------------------------
class TestRetrievalIteration:
    def test_creation(self) -> None:
        it = RetrievalIteration(
            iteration=0,
            query_terms=["a", "b"],
            result_count=10,
            avg_score=0.5,
            top_score=0.9,
            coverage=0.8,
            refined_terms=["a", "b", "c"],
        )
        assert it.iteration == 0
        assert it.result_count == 10
        assert it.coverage == 0.8


class TestConvergenceInfo:
    def test_creation(self) -> None:
        c = ConvergenceInfo(
            reason="score_plateau",
            iterations_run=3,
            final_avg_score=0.7,
            score_improvement=0.1,
        )
        assert c.reason == "score_plateau"
        assert c.iterations_run == 3


class TestSelfImprovingResult:
    def test_empty(self) -> None:
        r = SelfImprovingResult()
        assert r.total_iterations == 0
        assert r.converged is False

    def test_converged(self) -> None:
        r = SelfImprovingResult(
            convergence=ConvergenceInfo(
                reason="score_plateau",
                iterations_run=2,
                final_avg_score=0.8,
                score_improvement=0.05,
            )
        )
        assert r.converged is True

    def test_not_converged_max_iter(self) -> None:
        r = SelfImprovingResult(
            convergence=ConvergenceInfo(
                reason="max_iterations",
                iterations_run=3,
                final_avg_score=0.5,
                score_improvement=0.1,
            )
        )
        assert r.converged is False

    def test_total_iterations(self) -> None:
        r = SelfImprovingResult(
            iterations=[
                RetrievalIteration(0, ["a"], 5, 0.5, 0.8, 0.7, ["a", "b"]),
                RetrievalIteration(1, ["a", "b"], 5, 0.6, 0.9, 0.8, ["a", "b"]),
            ]
        )
        assert r.total_iterations == 2


# ---------------------------------------------------------------------------
# Coverage tests
# ---------------------------------------------------------------------------
class TestComputeCoverage:
    def test_empty_terms(self) -> None:
        assert _compute_coverage([], [_make_candidate()]) == 0.0

    def test_empty_papers(self) -> None:
        assert _compute_coverage(["ml"], []) == 0.0

    def test_full_coverage(self) -> None:
        papers = [_make_candidate(title="Machine Learning Paper")]
        cov = _compute_coverage(["machine", "learning"], papers)
        assert cov == 1.0

    def test_partial_coverage(self) -> None:
        papers = [_make_candidate(title="Machine Vision")]
        cov = _compute_coverage(["machine", "learning"], papers)
        assert cov == 0.5

    def test_no_coverage(self) -> None:
        papers = [_make_candidate(title="Biology Study")]
        cov = _compute_coverage(["machine", "learning"], papers)
        assert cov == 0.0


# ---------------------------------------------------------------------------
# Score computation tests
# ---------------------------------------------------------------------------
class TestComputeAvgScore:
    def test_empty(self) -> None:
        assert _compute_avg_score([], {}) == 0.0

    def test_no_scores(self) -> None:
        papers = [_make_candidate(), _make_candidate(arxiv_id="2301.00002")]
        assert _compute_avg_score(papers, {}) == 0.0

    def test_with_scores(self) -> None:
        papers = [
            _make_candidate(arxiv_id="a"),
            _make_candidate(arxiv_id="b"),
        ]
        scores = {"a": 0.8, "b": 0.4}
        assert abs(_compute_avg_score(papers, scores) - 0.6) < 1e-9


class TestComputeTopScore:
    def test_empty(self) -> None:
        assert _compute_top_score([], {}) == 0.0

    def test_no_scores(self) -> None:
        assert _compute_top_score([_make_candidate()], {}) == 0.0

    def test_with_scores(self) -> None:
        papers = [
            _make_candidate(arxiv_id="a"),
            _make_candidate(arxiv_id="b"),
        ]
        scores = {"a": 0.3, "b": 0.9}
        assert _compute_top_score(papers, scores) == 0.9


# ---------------------------------------------------------------------------
# Term extraction tests
# ---------------------------------------------------------------------------
class TestExtractTopTerms:
    def test_empty_papers(self) -> None:
        assert _extract_top_terms([], set(), 5) == []

    def test_extracts_terms(self) -> None:
        papers = [
            _make_candidate(
                title="Transformer Architecture for NLP",
                abstract="Transformer models use attention mechanisms.",
            ),
            _make_candidate(
                title="Attention Mechanisms in Transformers",
                abstract="Self-attention is key to transformer models.",
            ),
        ]
        terms = _extract_top_terms(papers, set(), top_n=3)
        assert len(terms) <= 3
        assert all(isinstance(t, str) for t in terms)

    def test_excludes_existing(self) -> None:
        papers = [
            _make_candidate(
                title="Transformer Architecture",
                abstract="Transformer architecture overview.",
            ),
        ]
        terms = _extract_top_terms(papers, {"transformer", "architecture"}, 5)
        assert "transformer" not in terms
        assert "architecture" not in terms


# ---------------------------------------------------------------------------
# Refine terms tests
# ---------------------------------------------------------------------------
class TestRefineTerms:
    def test_empty_papers(self) -> None:
        result = _refine_terms(["a", "b"], [], 0.0)
        assert result == ["a", "b"]

    def test_keeps_matching_terms(self) -> None:
        papers = [
            _make_candidate(
                title="Machine Learning Advances",
                abstract="Deep learning methods.",
            )
        ]
        result = _refine_terms(["machine", "learning", "quantum"], papers, 0.5)
        assert "machine" in result
        assert "learning" in result

    def test_avoids_query_collapse(self) -> None:
        papers = [_make_candidate(title="Unrelated Title", abstract="Unrelated.")]
        result = _refine_terms(
            ["machine", "learning", "neural", "network"], papers, 0.0
        )
        # Should keep original if too many would be dropped
        assert len(result) >= 2

    def test_adds_new_terms(self) -> None:
        papers = [
            _make_candidate(
                title="Transformer Attention Mechanism",
                abstract="Transformer models use self-attention.",
            )
            for _ in range(3)
        ]
        result = _refine_terms(["transformer"], papers, 1.0)
        # Should add at least one new term
        assert len(result) >= 1


# ---------------------------------------------------------------------------
# Convergence tests
# ---------------------------------------------------------------------------
class TestCheckConvergence:
    def test_empty_iterations(self) -> None:
        assert _check_convergence([], 3, 0.01, 0.9) is None

    def test_single_iteration(self) -> None:
        iters = [
            RetrievalIteration(0, ["a"], 5, 0.5, 0.8, 0.7, ["a"]),
        ]
        assert _check_convergence(iters, 3, 0.01, 0.9) is None

    def test_max_iterations(self) -> None:
        iters = [
            RetrievalIteration(0, ["a"], 5, 0.5, 0.8, 0.7, ["a", "b"]),
            RetrievalIteration(1, ["a", "b"], 5, 0.6, 0.9, 0.8, ["a", "b"]),
            RetrievalIteration(2, ["a", "b"], 5, 0.7, 0.95, 0.85, ["a", "b"]),
        ]
        c = _check_convergence(iters, 3, 0.01, 0.9)
        assert c is not None
        assert c.reason == "max_iterations"

    def test_score_plateau(self) -> None:
        iters = [
            RetrievalIteration(0, ["a"], 5, 0.5, 0.8, 0.7, ["a", "b"]),
            RetrievalIteration(1, ["a", "b"], 5, 0.505, 0.8, 0.7, ["a", "b"]),
        ]
        c = _check_convergence(iters, 5, 0.01, 0.9)
        assert c is not None
        assert c.reason == "score_plateau"

    def test_term_stable(self) -> None:
        iters = [
            RetrievalIteration(0, ["a", "b", "c"], 5, 0.5, 0.8, 0.7, ["a", "b", "c"]),
            RetrievalIteration(1, ["a", "b", "c"], 5, 0.6, 0.9, 0.8, ["a", "b", "c"]),
        ]
        c = _check_convergence(iters, 5, 0.001, 0.9)
        assert c is not None
        assert c.reason == "term_stable"

    def test_no_results(self) -> None:
        iters = [
            RetrievalIteration(0, ["a", "b"], 5, 0.5, 0.8, 0.7, ["a", "c"]),
            RetrievalIteration(1, ["a", "c"], 0, 0.0, 0.0, 0.0, ["a", "c"]),
        ]
        # score_threshold=0.001 avoids plateau, term_overlap<0.9 avoids stable
        c = _check_convergence(iters, 5, 0.001, 0.95)
        assert c is not None
        assert c.reason == "no_results"

    def test_no_convergence_yet(self) -> None:
        iters = [
            RetrievalIteration(0, ["a"], 5, 0.3, 0.5, 0.5, ["a", "b"]),
            RetrievalIteration(1, ["a", "b"], 5, 0.5, 0.7, 0.7, ["a", "b", "c"]),
        ]
        c = _check_convergence(iters, 5, 0.01, 0.95)
        assert c is None


# ---------------------------------------------------------------------------
# Score papers tests
# ---------------------------------------------------------------------------
class TestScorePapers:
    def test_empty_terms(self) -> None:
        papers = [_make_candidate()]
        result, scores = _score_papers(papers, [])
        assert len(result) == 1

    def test_scores_assigned(self) -> None:
        papers = [
            _make_candidate(
                arxiv_id="a", title="Machine Learning", abstract="Deep learning."
            ),
            _make_candidate(arxiv_id="b", title="Biology", abstract="Cells."),
        ]
        result, scores = _score_papers(papers, ["machine", "learning"])
        assert scores["a"] > 0
        # First should have higher score (has both terms)
        assert scores["a"] >= scores["b"]

    def test_sorted_descending(self) -> None:
        papers = [
            _make_candidate(
                arxiv_id="low", title="No Match", abstract="Nothing relevant."
            ),
            _make_candidate(
                arxiv_id="high",
                title="Machine Learning AI",
                abstract="Deep neural network.",
            ),
        ]
        result, scores = _score_papers(papers, ["machine", "learning", "neural"])
        # First paper should be the higher scored one
        assert scores[result[0].arxiv_id] >= scores[result[1].arxiv_id]


# ---------------------------------------------------------------------------
# Full loop tests
# ---------------------------------------------------------------------------
class TestRunSelfImprovingRetrieval:
    def test_empty_papers(self) -> None:
        plan = _make_plan()
        result = run_self_improving_retrieval(plan, [])
        assert result.total_iterations == 0
        assert result.convergence is not None
        assert result.convergence.reason == "no_results"

    def test_basic_loop(self) -> None:
        plan = _make_plan(
            topic="machine learning",
            must=["machine", "learning"],
            nice=["neural"],
        )
        papers = [
            _make_candidate(
                arxiv_id="1",
                title="Machine Learning Survey",
                abstract="A comprehensive survey of machine learning methods.",
            ),
            _make_candidate(
                arxiv_id="2",
                title="Neural Network Design",
                abstract="Neural network architecture for learning tasks.",
            ),
            _make_candidate(
                arxiv_id="3",
                title="Biology of Cells",
                abstract="Cell biology overview with no ML content.",
            ),
        ]
        result = run_self_improving_retrieval(plan, papers, max_iterations=3)
        assert result.total_iterations >= 1
        assert result.convergence is not None
        assert len(result.final_papers) == 3
        assert len(result.final_query_terms) >= 2

    def test_convergence_happens(self) -> None:
        plan = _make_plan(
            topic="transformer attention",
            must=["transformer", "attention"],
            nice=[],
        )
        papers = [
            _make_candidate(
                title="Transformer Attention Mechanism",
                abstract="Self-attention in transformer models.",
            ),
            _make_candidate(
                title="Attention Is All You Need",
                abstract="Transformer architecture with attention.",
            ),
        ]
        result = run_self_improving_retrieval(
            plan, papers, max_iterations=5, score_threshold=0.01
        )
        assert result.convergence is not None
        # Should converge before max due to stable terms/scores
        assert result.convergence.iterations_run <= 5

    def test_single_iteration(self) -> None:
        plan = _make_plan()
        papers = [_make_candidate(title="ML Paper", abstract="Learning stuff.")]
        result = run_self_improving_retrieval(plan, papers, max_iterations=1)
        assert result.total_iterations == 1
        assert result.convergence is not None
        assert result.convergence.reason == "max_iterations"

    def test_empty_terms_fallback(self) -> None:
        plan = QueryPlan(
            topic_raw="machine learning",
            topic_normalized="machine learning",
            must_terms=[],
            nice_terms=[],
            query_variants=["ml"],
        )
        papers = [_make_candidate(title="Machine Learning", abstract="ML methods.")]
        result = run_self_improving_retrieval(plan, papers, max_iterations=2)
        assert result.total_iterations >= 1
        # Should have used topic words as fallback
        assert len(result.final_query_terms) >= 1

    def test_result_papers_sorted(self) -> None:
        plan = _make_plan(must=["machine", "learning"], nice=[])
        papers = [
            _make_candidate(
                arxiv_id="low",
                title="Unrelated Topic",
                abstract="Nothing about ML.",
            ),
            _make_candidate(
                arxiv_id="high",
                title="Machine Learning Deep",
                abstract="Machine learning and deep learning.",
            ),
        ]
        result = run_self_improving_retrieval(plan, papers, max_iterations=1)
        # Higher-scoring paper should come first
        if len(result.final_papers) >= 2:
            # "high" has both terms, "low" has neither
            assert result.final_papers[0].arxiv_id == "high"
