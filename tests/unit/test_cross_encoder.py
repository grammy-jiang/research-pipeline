"""Tests for cross-encoder passage reranking."""

from unittest.mock import MagicMock, patch

import pytest

from research_pipeline.models.extraction import ChunkMetadata


def _make_chunk(
    paper_id: str = "2301.00001",
    section: str = "introduction",
    chunk_id: str = "c1",
    text: str = "Some text about transformers.",
) -> tuple[ChunkMetadata, str]:
    """Helper to create a (ChunkMetadata, text) tuple."""
    meta = ChunkMetadata(
        paper_id=paper_id,
        section_path=section,
        chunk_id=chunk_id,
        source_span="1-10",
        token_count=len(text.split()),
    )
    return (meta, text)


def _make_chunks(n: int) -> list[tuple[ChunkMetadata, str]]:
    """Create *n* distinct chunks for testing."""
    return [
        _make_chunk(
            chunk_id=f"c{i}",
            section=f"section-{i}",
            text=f"Chunk number {i} about topic {i % 3}.",
        )
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# _is_cross_encoder_available
# ---------------------------------------------------------------------------


class TestIsCrossEncoderAvailable:
    """Tests for the availability check function."""

    def test_available_when_installed(self) -> None:
        mock_module = MagicMock()
        with patch.dict("sys.modules", {"sentence_transformers": mock_module}):
            from research_pipeline.extraction.cross_encoder import (
                _is_cross_encoder_available,
            )

            assert _is_cross_encoder_available() is True

    def test_unavailable_when_not_installed(self) -> None:
        with patch.dict("sys.modules", {"sentence_transformers": None}):
            from research_pipeline.extraction.cross_encoder import (
                _is_cross_encoder_available,
            )

            assert _is_cross_encoder_available() is False


# ---------------------------------------------------------------------------
# cross_encoder_rerank
# ---------------------------------------------------------------------------


class TestCrossEncoderRerank:
    """Tests for the cross_encoder_rerank function."""

    def test_empty_chunks_returns_empty(self) -> None:
        from research_pipeline.extraction.cross_encoder import cross_encoder_rerank

        result = cross_encoder_rerank(query="test", chunks=[], top_k=5)
        assert result == []

    def test_empty_query_returns_empty(self) -> None:
        from research_pipeline.extraction.cross_encoder import cross_encoder_rerank

        chunks = _make_chunks(3)
        result = cross_encoder_rerank(query="", chunks=chunks, top_k=5)
        assert result == []

    def test_unavailable_raises_import_error(self) -> None:
        with patch.dict("sys.modules", {"sentence_transformers": None}):
            from research_pipeline.extraction.cross_encoder import (
                cross_encoder_rerank,
            )

            chunks = _make_chunks(3)
            with pytest.raises(ImportError, match="sentence-transformers"):
                cross_encoder_rerank(query="test", chunks=chunks)

    def test_rerank_with_mocked_model(self) -> None:
        """Verify correct input pairs and sorted output."""
        mock_ce_class = MagicMock()
        mock_model = MagicMock()
        mock_ce_class.return_value = mock_model
        # Scores: chunk 0 gets 0.1, chunk 1 gets 0.9, chunk 2 gets 0.5
        mock_model.predict.return_value = [0.1, 0.9, 0.5]

        mock_st = MagicMock()
        mock_st.CrossEncoder = mock_ce_class

        with patch.dict("sys.modules", {"sentence_transformers": mock_st}):
            from research_pipeline.extraction.cross_encoder import (
                cross_encoder_rerank,
            )

            chunks = _make_chunks(3)
            result = cross_encoder_rerank(query="neural networks", chunks=chunks)

        # Verify pairs formed correctly
        call_args = mock_model.predict.call_args[0][0]
        assert len(call_args) == 3
        for pair in call_args:
            assert pair[0] == "neural networks"

        # Verify sorted by score descending
        assert result[0] == (1, 0.9)
        assert result[1] == (2, 0.5)
        assert result[2] == (0, 0.1)

    def test_single_chunk(self) -> None:
        mock_ce_class = MagicMock()
        mock_model = MagicMock()
        mock_ce_class.return_value = mock_model
        mock_model.predict.return_value = [0.8]

        mock_st = MagicMock()
        mock_st.CrossEncoder = mock_ce_class

        with patch.dict("sys.modules", {"sentence_transformers": mock_st}):
            from research_pipeline.extraction.cross_encoder import (
                cross_encoder_rerank,
            )

            chunks = _make_chunks(1)
            result = cross_encoder_rerank(query="test", chunks=chunks)

        assert len(result) == 1
        assert result[0] == (0, 0.8)

    def test_top_k_smaller_than_count(self) -> None:
        mock_ce_class = MagicMock()
        mock_model = MagicMock()
        mock_ce_class.return_value = mock_model
        mock_model.predict.return_value = [0.1, 0.9, 0.5, 0.3, 0.7]

        mock_st = MagicMock()
        mock_st.CrossEncoder = mock_ce_class

        with patch.dict("sys.modules", {"sentence_transformers": mock_st}):
            from research_pipeline.extraction.cross_encoder import (
                cross_encoder_rerank,
            )

            chunks = _make_chunks(5)
            result = cross_encoder_rerank(query="test", chunks=chunks, top_k=2)

        assert len(result) == 2
        # Top two by score: index 1 (0.9), index 4 (0.7)
        assert result[0][0] == 1
        assert result[1][0] == 4

    def test_top_k_larger_than_count(self) -> None:
        mock_ce_class = MagicMock()
        mock_model = MagicMock()
        mock_ce_class.return_value = mock_model
        mock_model.predict.return_value = [0.2, 0.8]

        mock_st = MagicMock()
        mock_st.CrossEncoder = mock_ce_class

        with patch.dict("sys.modules", {"sentence_transformers": mock_st}):
            from research_pipeline.extraction.cross_encoder import (
                cross_encoder_rerank,
            )

            chunks = _make_chunks(2)
            result = cross_encoder_rerank(query="test", chunks=chunks, top_k=100)

        assert len(result) == 2


# ---------------------------------------------------------------------------
# retrieve_relevant_chunks with cross-encoder
# ---------------------------------------------------------------------------


class TestRetrieveWithCrossEncoder:
    """Integration tests for cross-encoder in retrieve_relevant_chunks."""

    def _patch_cross_encoder(
        self, available: bool = True, scores: list[float] | None = None
    ):
        """Create patches for cross-encoder availability and model."""
        patches = {}

        patches["available"] = patch(
            "research_pipeline.extraction.cross_encoder._is_cross_encoder_available",
            return_value=available,
        )

        if scores is not None:
            mock_ce_class = MagicMock()
            mock_model = MagicMock()
            mock_ce_class.return_value = mock_model
            mock_model.predict.return_value = scores
            mock_st = MagicMock()
            mock_st.CrossEncoder = mock_ce_class
            patches["st_module"] = patch.dict(
                "sys.modules", {"sentence_transformers": mock_st}
            )
        return patches

    def test_auto_detect_enables_when_available(self) -> None:
        from research_pipeline.extraction.retrieval import retrieve_relevant_chunks

        chunks = _make_chunks(5)
        # scores for the candidate pool (up to 30 or 3*top_k)
        scores = [0.1, 0.9, 0.5, 0.3, 0.7]

        p = self._patch_cross_encoder(available=True, scores=scores)
        with p["available"], p["st_module"]:
            result = retrieve_relevant_chunks(
                chunks, ["test", "query"], top_k=3, use_embeddings=False
            )

        assert len(result) == 3
        # Cross-encoder should have reranked
        assert result[0][2] == 0.9

    def test_auto_detect_skips_when_unavailable(self) -> None:
        from research_pipeline.extraction.retrieval import retrieve_relevant_chunks

        chunks = _make_chunks(3)

        p = self._patch_cross_encoder(available=False)
        with p["available"]:
            result = retrieve_relevant_chunks(
                chunks, ["chunk", "number"], top_k=3, use_embeddings=False
            )

        # Should still return results (BM25-only)
        assert len(result) == 3

    def test_auto_detect_skips_when_too_many_chunks(self) -> None:
        """Auto-detect should skip cross-encoder when chunks > 100."""
        from research_pipeline.extraction.retrieval import retrieve_relevant_chunks

        chunks = _make_chunks(101)

        p = self._patch_cross_encoder(available=True)
        with p["available"]:
            result = retrieve_relevant_chunks(
                chunks, ["chunk"], top_k=5, use_embeddings=False
            )

        # Should return results via BM25 only (no cross-encoder)
        assert len(result) == 5

    def test_force_true_with_available(self) -> None:
        from research_pipeline.extraction.retrieval import retrieve_relevant_chunks

        chunks = _make_chunks(4)
        scores = [0.2, 0.8, 0.4, 0.6]

        p = self._patch_cross_encoder(available=True, scores=scores)
        with p["available"], p["st_module"]:
            result = retrieve_relevant_chunks(
                chunks,
                ["test"],
                top_k=2,
                use_embeddings=False,
                use_cross_encoder=True,
            )

        assert len(result) == 2

    def test_force_true_with_unavailable_raises(self) -> None:
        from research_pipeline.extraction.retrieval import retrieve_relevant_chunks

        chunks = _make_chunks(3)

        p = self._patch_cross_encoder(available=False)
        with p["available"], pytest.raises(ImportError, match="sentence-transformers"):
            retrieve_relevant_chunks(
                chunks,
                ["test"],
                top_k=3,
                use_embeddings=False,
                use_cross_encoder=True,
            )

    def test_force_false_skips(self) -> None:
        from research_pipeline.extraction.retrieval import retrieve_relevant_chunks

        chunks = _make_chunks(3)

        p = self._patch_cross_encoder(available=True)
        with p["available"]:
            result = retrieve_relevant_chunks(
                chunks,
                ["chunk", "number"],
                top_k=3,
                use_embeddings=False,
                use_cross_encoder=False,
            )

        # BM25 only — results returned
        assert len(result) == 3

    def test_bm25_then_cross_encoder_flow(self) -> None:
        """Full BM25 → cross-encoder reranking pipeline."""
        from research_pipeline.extraction.retrieval import retrieve_relevant_chunks

        chunks = [
            _make_chunk(chunk_id="c0", text="deep learning models for images"),
            _make_chunk(chunk_id="c1", text="transformer architecture attention"),
            _make_chunk(chunk_id="c2", text="random forest classification trees"),
        ]
        # Cross-encoder scores applied to BM25-ordered candidate pool.
        # The exact mapping depends on BM25 ordering, so we verify that
        # cross-encoder scores appear in the output (not BM25 scores).
        scores = [0.3, 0.95, 0.1]

        p = self._patch_cross_encoder(available=True, scores=scores)
        with p["available"], p["st_module"]:
            result = retrieve_relevant_chunks(
                chunks,
                ["transformer", "attention"],
                top_k=2,
                use_embeddings=False,
                use_cross_encoder=True,
            )

        assert len(result) == 2
        # The highest cross-encoder score (0.95) should be first
        assert result[0][2] == 0.95
        # Cross-encoder scores should replace BM25 scores
        result_scores = {r[2] for r in result}
        assert result_scores <= {0.3, 0.95, 0.1}
