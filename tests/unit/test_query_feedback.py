"""Tests for screening.query_feedback — query refinement feedback."""

from datetime import UTC, datetime

import pytest

from research_pipeline.models.candidate import CandidateRecord
from research_pipeline.models.query_plan import QueryPlan
from research_pipeline.screening.query_feedback import (
    STOPWORDS,
    QueryRefinement,
    _tokenize,
    _tokenize_paper,
    compute_query_refinement,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NOW = datetime(2025, 1, 1, tzinfo=UTC)


def _make_paper(
    title: str = "Default Title",
    abstract: str = "Default abstract text.",
    arxiv_id: str = "2501.00001",
) -> CandidateRecord:
    """Create a minimal CandidateRecord for testing."""
    return CandidateRecord(
        arxiv_id=arxiv_id,
        version="v1",
        title=title,
        authors=["Author One"],
        published=_NOW,
        updated=_NOW,
        categories=["cs.IR"],
        primary_category="cs.IR",
        abstract=abstract,
        abs_url=f"https://arxiv.org/abs/{arxiv_id}",
        pdf_url=f"https://arxiv.org/pdf/{arxiv_id}",
    )


def _make_plan(
    topic: str = "neural retrieval",
    must_terms: list[str] | None = None,
    nice_terms: list[str] | None = None,
) -> QueryPlan:
    """Create a minimal QueryPlan for testing."""
    return QueryPlan(
        topic_raw=topic,
        topic_normalized=topic,
        must_terms=must_terms or [],
        nice_terms=nice_terms or [],
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestTokenize:
    """Tests for low-level tokenizer helpers."""

    def test_tokenize_lowercase(self) -> None:
        assert _tokenize("Hello World") == ["hello", "world"]

    def test_tokenize_strips_short_and_numeric(self) -> None:
        tokens = _tokenize("A 1 ab x2 foo")
        assert "foo" in tokens
        assert "ab" in tokens
        # Single char 'a' excluded by regex (need len>1)

    def test_tokenize_paper_removes_stopwords(self) -> None:
        paper = _make_paper(
            title="The transformer architecture",
            abstract="We propose a new method for retrieval.",
        )
        tokens = _tokenize_paper(paper)
        assert "transformer" in tokens
        assert "architecture" in tokens
        assert "retrieval" in tokens
        # Stopwords removed
        assert "the" not in tokens
        assert "we" not in tokens
        assert "propose" not in tokens


class TestEmptyPapers:
    """Edge case: no papers should return empty suggestions."""

    def test_empty_papers_returns_empty_suggestions(self) -> None:
        plan = _make_plan(must_terms=["neural", "retrieval"])
        result = compute_query_refinement(plan, [])

        assert result.original_query == "neural retrieval"
        assert result.suggested_additions == []
        assert result.suggested_removals == []
        assert result.term_coverage == {}
        assert result.emergent_terms == []
        assert result.refined_must_terms == []
        assert result.refined_nice_terms == []


class TestTermCoverage:
    """Tests for term_coverage computation."""

    def test_term_in_all_papers_has_coverage_one(self) -> None:
        papers = [
            _make_paper(
                title="Neural network approaches", abstract="Neural nets rock."
            ),
            _make_paper(
                title="Deep neural models",
                abstract="We study neural methods.",
                arxiv_id="2501.00002",
            ),
        ]
        plan = _make_plan(must_terms=["neural"])
        result = compute_query_refinement(plan, papers)

        assert result.term_coverage["neural"] == pytest.approx(1.0)

    def test_term_in_no_papers_has_coverage_zero(self) -> None:
        papers = [
            _make_paper(title="Graph algorithms", abstract="Shortest paths."),
        ]
        plan = _make_plan(must_terms=["quantum"])
        result = compute_query_refinement(plan, papers)

        assert result.term_coverage["quantum"] == pytest.approx(0.0)

    def test_term_in_half_papers_has_coverage_half(self) -> None:
        papers = [
            _make_paper(
                title="Attention mechanisms",
                abstract="Transformers use attention.",
                arxiv_id="2501.00001",
            ),
            _make_paper(
                title="Graph networks",
                abstract="Message passing.",
                arxiv_id="2501.00002",
            ),
        ]
        plan = _make_plan(nice_terms=["attention"])
        result = compute_query_refinement(plan, papers)

        assert result.term_coverage["attention"] == pytest.approx(0.5)


class TestSuggestedRemovals:
    """Tests for low-coverage term removal suggestions."""

    def test_low_coverage_term_suggested_for_removal(self) -> None:
        papers = [
            _make_paper(
                title="Transformer attention",
                abstract="Self-attention layers.",
                arxiv_id="2501.00001",
            ),
            _make_paper(
                title="Recurrent networks",
                abstract="LSTM and GRU.",
                arxiv_id="2501.00002",
            ),
            _make_paper(
                title="Convolutional models",
                abstract="Image classification.",
                arxiv_id="2501.00003",
            ),
        ]
        # "attention" only in 1/3 papers (~0.33), "transformer" in 1/3.
        # With k_threshold=0.5, both should be suggested for removal.
        plan = _make_plan(must_terms=["transformer", "attention"])
        result = compute_query_refinement(plan, papers, k_threshold=0.5)

        assert "transformer" in result.suggested_removals
        assert "attention" in result.suggested_removals

    def test_high_coverage_term_not_suggested_for_removal(self) -> None:
        papers = [
            _make_paper(
                title="Neural retrieval",
                abstract="Neural information retrieval.",
                arxiv_id="2501.00001",
            ),
            _make_paper(
                title="Dense retrieval",
                abstract="Embedding-based retrieval.",
                arxiv_id="2501.00002",
            ),
        ]
        plan = _make_plan(must_terms=["retrieval"])
        result = compute_query_refinement(plan, papers, k_threshold=0.3)

        assert "retrieval" not in result.suggested_removals


class TestEmergentTerms:
    """Tests for emergent term discovery."""

    def test_emergent_terms_found(self) -> None:
        papers = [
            _make_paper(
                title="Embedding vectors for search",
                abstract="Dense embedding retrieval with vectors.",
                arxiv_id="2501.00001",
            ),
            _make_paper(
                title="Embedding similarity search",
                abstract="Cosine similarity over embedding space.",
                arxiv_id="2501.00002",
            ),
        ]
        plan = _make_plan(topic="search", must_terms=["search"])
        result = compute_query_refinement(plan, papers)

        # "embedding" appears in both papers but not in the query
        assert "embedding" in result.emergent_terms

    def test_stopwords_not_in_emergent_terms(self) -> None:
        papers = [
            _make_paper(
                title="The model and the data",
                abstract="We propose a method using this approach.",
                arxiv_id="2501.00001",
            ),
        ]
        plan = _make_plan(must_terms=["unrelated"])
        result = compute_query_refinement(plan, papers)

        for term in result.emergent_terms:
            assert term not in STOPWORDS

    def test_short_tokens_excluded_from_emergent(self) -> None:
        papers = [
            _make_paper(
                title="An NLP task",
                abstract="NLP is fun but hard.",
                arxiv_id="2501.00001",
            ),
        ]
        plan = _make_plan(must_terms=["something"])
        result = compute_query_refinement(plan, papers)

        # "nlp" is only 3 chars — excluded by len > 3 filter
        assert "nlp" not in result.emergent_terms


class TestRefinedTerms:
    """Tests for refined must/nice term construction."""

    def test_low_coverage_removed_from_refined(self) -> None:
        papers = [
            _make_paper(
                title="Attention transformer",
                abstract="Self-attention mechanism.",
                arxiv_id="2501.00001",
            ),
            _make_paper(
                title="Attention mechanism analysis",
                abstract="Multi-head attention.",
                arxiv_id="2501.00002",
            ),
            _make_paper(
                title="Attention in vision models",
                abstract="Visual attention layers.",
                arxiv_id="2501.00003",
            ),
        ]
        # "quantum" appears in 0/3 papers — should be removed
        plan = _make_plan(
            must_terms=["attention", "quantum"],
            nice_terms=["transformer"],
        )
        result = compute_query_refinement(plan, papers, k_threshold=0.3)

        assert "quantum" not in result.refined_must_terms
        assert "attention" in result.refined_must_terms

    def test_emergent_terms_added_to_refined(self) -> None:
        papers = [
            _make_paper(
                title="Contrastive learning for embeddings",
                abstract="Contrastive loss improves embedding quality.",
                arxiv_id="2501.00001",
            ),
            _make_paper(
                title="Contrastive representation learning",
                abstract="Self-supervised contrastive framework.",
                arxiv_id="2501.00002",
            ),
        ]
        plan = _make_plan(topic="representation", must_terms=["representation"])
        result = compute_query_refinement(plan, papers)

        # "contrastive" should be emergent and added to refined terms
        assert "contrastive" in result.emergent_terms
        all_refined = result.refined_must_terms + result.refined_nice_terms
        assert "contrastive" in all_refined


class TestKThreshold:
    """Tests for k_threshold parameter behavior."""

    def test_strict_threshold_removes_more(self) -> None:
        papers = [
            _make_paper(
                title="Alpha method beta",
                abstract="Alpha and gamma.",
                arxiv_id="2501.00001",
            ),
            _make_paper(
                title="Alpha approach",
                abstract="Delta and alpha.",
                arxiv_id="2501.00002",
            ),
            _make_paper(
                title="Gamma systems",
                abstract="Gamma analysis.",
                arxiv_id="2501.00003",
            ),
        ]
        plan = _make_plan(must_terms=["alpha", "beta", "gamma"])

        # Lenient threshold: only terms in < 10% removed
        lenient = compute_query_refinement(plan, papers, k_threshold=0.1)
        # Strict threshold: terms in < 80% removed
        strict = compute_query_refinement(plan, papers, k_threshold=0.8)

        assert len(strict.suggested_removals) >= len(lenient.suggested_removals)

    def test_zero_threshold_removes_nothing(self) -> None:
        papers = [
            _make_paper(
                title="Foo bar",
                abstract="Baz qux.",
                arxiv_id="2501.00001",
            ),
        ]
        plan = _make_plan(must_terms=["unmatched"])
        result = compute_query_refinement(plan, papers, k_threshold=0.0)

        # k_threshold=0.0 means even 0% coverage is acceptable (< 0.0 is impossible)
        assert result.suggested_removals == []


class TestQueryRefinementModel:
    """Tests for the QueryRefinement Pydantic model itself."""

    def test_model_roundtrip(self) -> None:
        ref = QueryRefinement(
            original_query="neural retrieval",
            suggested_additions=["embedding"],
            suggested_removals=["quantum"],
            term_coverage={"neural": 0.8, "retrieval": 1.0},
            emergent_terms=["embedding", "contrastive"],
            refined_must_terms=["neural", "retrieval"],
            refined_nice_terms=["embedding"],
        )
        data = ref.model_dump()
        restored = QueryRefinement.model_validate(data)
        assert restored == ref
