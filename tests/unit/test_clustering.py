"""Tests for screening.clustering — TF-IDF paper similarity clustering."""

from __future__ import annotations

from research_pipeline.models.candidate import CandidateRecord
from research_pipeline.screening.clustering import (
    PaperCluster,
    _agglomerative_cluster,
    _build_tfidf,
    _cluster_label,
    _cluster_top_terms,
    _cosine_similarity,
    _tokenize,
    cluster_candidates,
)


def _make_candidate(
    arxiv_id: str = "2401.00001",
    title: str = "Test Paper",
    abstract: str = "A test abstract about machine learning.",
    **kwargs: object,
) -> CandidateRecord:
    """Create a test CandidateRecord."""
    defaults: dict[str, object] = {
        "arxiv_id": arxiv_id,
        "version": "v1",
        "title": title,
        "authors": ["Author One"],
        "abstract": abstract,
        "published": "2024-01-15T00:00:00Z",
        "updated": "2024-01-15T00:00:00Z",
        "categories": ["cs.CL"],
        "primary_category": "cs.CL",
        "abs_url": f"https://arxiv.org/abs/{arxiv_id}",
        "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}",
        "source": "arxiv",
    }
    defaults.update(kwargs)
    return CandidateRecord.model_validate(defaults)


class TestTokenize:
    """Tests for _tokenize()."""

    def test_basic(self) -> None:
        tokens = _tokenize("Hello World Machine Learning")
        assert "hello" in tokens
        assert "world" in tokens
        assert "machine" in tokens
        assert "learning" in tokens

    def test_removes_stop_words(self) -> None:
        tokens = _tokenize("the model and the approach")
        assert "the" not in tokens
        assert "model" not in tokens
        assert "approach" not in tokens

    def test_short_words_removed(self) -> None:
        tokens = _tokenize("a x I")
        assert len(tokens) == 0

    def test_lowercase(self) -> None:
        tokens = _tokenize("NLP TRANSFORMER")
        assert "nlp" in tokens
        assert "transformer" in tokens


class TestBuildTfidf:
    """Tests for _build_tfidf()."""

    def test_empty_docs(self) -> None:
        vocab, vectors = _build_tfidf([])
        assert vocab == []
        assert vectors == []

    def test_single_doc(self) -> None:
        vocab, vectors = _build_tfidf([["transformer", "attention"]])
        assert len(vectors) == 1
        assert "transformer" in vectors[0]

    def test_two_docs(self) -> None:
        docs = [["transformer", "attention"], ["cnn", "convolution"]]
        vocab, vectors = _build_tfidf(docs)
        assert len(vectors) == 2
        # Terms unique to a doc should have higher IDF
        assert "transformer" in vectors[0]
        assert "cnn" in vectors[1]


class TestCosineSimilarity:
    """Tests for _cosine_similarity()."""

    def test_identical_vectors(self) -> None:
        vec = {"a": 1.0, "b": 2.0}
        assert abs(_cosine_similarity(vec, vec) - 1.0) < 1e-6

    def test_orthogonal_vectors(self) -> None:
        a = {"x": 1.0}
        b = {"y": 1.0}
        assert _cosine_similarity(a, b) == 0.0

    def test_empty_vector(self) -> None:
        assert _cosine_similarity({}, {"a": 1.0}) == 0.0


class TestAgglomerativeCluster:
    """Tests for _agglomerative_cluster()."""

    def test_empty(self) -> None:
        assert _agglomerative_cluster([], 0.5) == []

    def test_single_doc(self) -> None:
        labels = _agglomerative_cluster([{"a": 1.0}], 0.5)
        assert labels == [0]

    def test_similar_docs_merge(self) -> None:
        vectors = [{"a": 1.0, "b": 0.5}, {"a": 0.9, "b": 0.6}]
        labels = _agglomerative_cluster(vectors, 0.5)
        assert labels[0] == labels[1]

    def test_dissimilar_docs_separate(self) -> None:
        vectors = [{"a": 1.0}, {"b": 1.0}]
        labels = _agglomerative_cluster(vectors, 0.5)
        assert labels[0] != labels[1]


class TestClusterTopTerms:
    """Tests for _cluster_top_terms()."""

    def test_returns_top_n(self) -> None:
        docs = [["a", "a", "b", "c"], ["a", "b", "d"]]
        terms = _cluster_top_terms(docs, [0, 0], cluster_id=0, top_n=3)
        assert len(terms) == 3
        assert terms[0] == "a"  # most frequent


class TestClusterLabel:
    """Tests for _cluster_label()."""

    def test_basic(self) -> None:
        assert (
            _cluster_label(["nlp", "transformer", "attention"])
            == "nlp, transformer, attention"
        )

    def test_empty(self) -> None:
        assert _cluster_label([]) == "misc"

    def test_truncates_to_three(self) -> None:
        label = _cluster_label(["a", "b", "c", "d", "e"])
        assert label == "a, b, c"


class TestClusterCandidates:
    """Integration tests for cluster_candidates()."""

    def test_empty(self) -> None:
        assert cluster_candidates([]) == []

    def test_single_paper(self) -> None:
        clusters = cluster_candidates([_make_candidate()])
        assert len(clusters) == 1
        assert len(clusters[0].paper_ids) == 1

    def test_similar_papers_cluster_together(self) -> None:
        candidates = [
            _make_candidate(
                arxiv_id="2401.00001",
                title="Transformer architectures for language tasks",
                abstract="Deep learning transformer neural networks",
            ),
            _make_candidate(
                arxiv_id="2401.00002",
                title="Attention transformer language understanding",
                abstract="Transformer neural networks language learning deep",
            ),
            _make_candidate(
                arxiv_id="2401.00003",
                title="Quantum computing algorithms chemistry simulation",
                abstract="Quantum algorithms applied molecular orbital simulation",
            ),
        ]
        clusters = cluster_candidates(candidates, threshold=0.1)
        # The two language/transformer papers should be in same cluster
        nlp_ids = {"2401.00001", "2401.00002"}
        found_nlp_together = False
        for c in clusters:
            if nlp_ids.issubset(set(c.paper_ids)):
                found_nlp_together = True
                assert "2401.00003" not in c.paper_ids
        assert found_nlp_together

    def test_sorted_by_size(self) -> None:
        candidates = [
            _make_candidate(
                arxiv_id=f"2401.0000{i}",
                title=f"Paper {i} about topic A transformers",
                abstract=f"Transformer topic A content {i}",
            )
            for i in range(5)
        ]
        clusters = cluster_candidates(candidates, threshold=0.1)
        sizes = [len(c.paper_ids) for c in clusters]
        assert sizes == sorted(sizes, reverse=True)

    def test_cluster_has_fields(self) -> None:
        clusters = cluster_candidates([_make_candidate()])
        c = clusters[0]
        assert isinstance(c, PaperCluster)
        assert isinstance(c.cluster_id, int)
        assert isinstance(c.label, str)
        assert isinstance(c.paper_ids, list)
        assert isinstance(c.top_terms, list)

    def test_high_threshold_separates_all(self) -> None:
        candidates = [
            _make_candidate(
                arxiv_id="2401.00001",
                title="Quantum chemistry orbital simulation",
                abstract="Molecular dynamics quantum orbital computation",
            ),
            _make_candidate(
                arxiv_id="2401.00002",
                title="Transformer language understanding",
                abstract="Neural network attention language processing",
            ),
            _make_candidate(
                arxiv_id="2401.00003",
                title="Economic policy inflation analysis",
                abstract="Monetary fiscal economic growth inflation",
            ),
        ]
        clusters = cluster_candidates(candidates, threshold=0.99)
        assert len(clusters) == 3
