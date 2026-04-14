"""Tests for hybrid BM25 + embedding retrieval."""

import pytest

from research_pipeline.extraction.retrieval import (
    _bm25_rank,
    _is_embedding_available,
    _reciprocal_rank_fusion,
    retrieve_relevant_chunks,
)
from research_pipeline.models.extraction import ChunkMetadata


def _make_chunks(texts: list[str]) -> list[tuple[ChunkMetadata, str]]:
    """Create test chunks from a list of text strings."""
    return [
        (
            ChunkMetadata(
                chunk_id=f"chunk-{i}",
                paper_id="2401.12345",
                section_path=f"section-{i}",
                token_count=len(t.split()),
                source_span=f"L{i * 10}-L{(i + 1) * 10}",
            ),
            t,
        )
        for i, t in enumerate(texts)
    ]


class TestRetrieveRelevantChunksBM25:
    """Tests for BM25-only retrieval path."""

    def test_bm25_returns_ranked_results(self) -> None:
        """BM25-only mode returns ranked results."""
        chunks = _make_chunks(
            [
                "deep learning for natural language processing",
                "cooking recipes for italian pasta",
                "neural networks and deep learning transformers",
            ]
        )
        results = retrieve_relevant_chunks(
            chunks, ["deep", "learning"], use_embeddings=False
        )
        assert len(results) > 0
        # Scores should be in descending order
        scores = [r[2] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_explicit_bm25_only(self) -> None:
        """Explicit use_embeddings=False forces BM25-only."""
        chunks = _make_chunks(
            [
                "transformer architecture attention mechanism",
                "random unrelated text about gardening",
            ]
        )
        results = retrieve_relevant_chunks(
            chunks, ["transformer", "attention"], use_embeddings=False
        )
        assert len(results) == 2
        # The transformer chunk should rank first
        assert results[0][0].chunk_id == "chunk-0"

    def test_empty_chunks_returns_empty(self) -> None:
        """Empty chunks input returns empty list."""
        results = retrieve_relevant_chunks([], ["query"], use_embeddings=False)
        assert results == []

    def test_empty_query_returns_empty(self) -> None:
        """Empty query terms returns empty list."""
        chunks = _make_chunks(["some text"])
        results = retrieve_relevant_chunks(chunks, [], use_embeddings=False)
        assert results == []

    def test_results_capped_at_top_k(self) -> None:
        """Results are capped at the requested top_k."""
        chunks = _make_chunks(
            [
                "deep learning paper one",
                "deep learning paper two",
                "deep learning paper three",
                "deep learning paper four",
                "deep learning paper five",
            ]
        )
        results = retrieve_relevant_chunks(
            chunks, ["deep", "learning"], top_k=3, use_embeddings=False
        )
        assert len(results) <= 3


class TestBM25Rank:
    """Tests for the _bm25_rank helper."""

    def test_ranks_relevant_chunks_higher(self) -> None:
        """BM25 ranks chunks containing query terms higher."""
        chunks = _make_chunks(
            [
                "machine learning optimization gradient descent",
                "the weather today is sunny and warm",
                "gradient descent optimization for deep learning",
            ]
        )
        ranked = _bm25_rank(chunks, ["gradient", "descent", "optimization"])
        # Top-ranked indices should be 0 and 2 (contain query terms)
        top_indices = {ranked[0][0], ranked[1][0]}
        assert top_indices == {0, 2}

    def test_returns_all_chunks(self) -> None:
        """BM25 ranking returns an entry for every chunk."""
        chunks = _make_chunks(["aaa", "bbb", "ccc"])
        ranked = _bm25_rank(chunks, ["aaa"])
        assert len(ranked) == 3


class TestReciprocalRankFusion:
    """Tests for _reciprocal_rank_fusion."""

    def test_merges_two_rankings(self) -> None:
        """RRF correctly merges two rankings."""
        ranking_a = [(0, 10.0), (1, 5.0), (2, 1.0)]
        ranking_b = [(2, 10.0), (0, 5.0), (1, 1.0)]
        fused = _reciprocal_rank_fusion([ranking_a, ranking_b], k=60)
        # All indices present
        fused_indices = [idx for idx, _ in fused]
        assert set(fused_indices) == {0, 1, 2}
        # Index 0 is rank 0 in A and rank 1 in B → highest RRF
        # Index 2 is rank 2 in A and rank 0 in B → same total as 0
        # Index 1 is rank 1 in A and rank 2 in B → same total as 0
        # All three have the same RRF score: 1/61 + 1/62
        # Actually: idx 0 → 1/61 + 1/62, idx 2 → 1/63 + 1/61, idx 1 → 1/62 + 1/63
        # So order should be 0, 2, 1
        assert fused_indices[0] == 0
        assert fused_indices[1] == 2
        assert fused_indices[2] == 1

    def test_single_ranking_preserves_order(self) -> None:
        """RRF with a single ranking preserves the original order."""
        ranking = [(3, 100.0), (1, 50.0), (0, 10.0)]
        fused = _reciprocal_rank_fusion([ranking], k=60)
        fused_indices = [idx for idx, _ in fused]
        assert fused_indices == [3, 1, 0]

    def test_rrf_scores_are_positive(self) -> None:
        """All RRF scores should be positive."""
        ranking_a = [(0, 1.0), (1, 0.5)]
        ranking_b = [(1, 1.0), (0, 0.5)]
        fused = _reciprocal_rank_fusion([ranking_a, ranking_b])
        for _idx, score in fused:
            assert score > 0


class TestEmbeddingAvailability:
    """Tests for embedding availability detection."""

    def test_returns_bool(self) -> None:
        """_is_embedding_available returns a boolean."""
        result = _is_embedding_available()
        assert isinstance(result, bool)


class TestEmbeddingForceRaises:
    """Tests for forced embedding mode when SPECTER2 is unavailable."""

    def test_use_embeddings_true_raises_when_unavailable(self) -> None:
        """use_embeddings=True raises ImportError when SPECTER2 is missing."""
        if _is_embedding_available():
            pytest.skip("SPECTER2 is installed; cannot test unavailable path")
        chunks = _make_chunks(["some text about transformers"])
        with pytest.raises(ImportError, match="SPECTER2 dependencies"):
            retrieve_relevant_chunks(chunks, ["transformers"], use_embeddings=True)
