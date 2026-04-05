"""Unit tests for screening.embedding (SPECTER2 semantic scoring).

These tests mock the heavy ML dependencies so they run fast and
without GPU/model downloads.
"""

from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from research_pipeline.models.candidate import CandidateRecord
from research_pipeline.screening.embedding import (
    _cosine_similarity,
    _is_specter2_available,
    score_semantic,
)


def _make_candidate(
    title: str = "Test Paper", abstract: str = "Content"
) -> CandidateRecord:
    return CandidateRecord(
        arxiv_id="2401.12345",
        version="v1",
        title=title,
        abstract=abstract,
        authors=["Author One"],
        published=datetime(2024, 1, 1, tzinfo=UTC),
        updated=datetime(2024, 1, 1, tzinfo=UTC),
        primary_category="cs.AI",
        categories=["cs.AI"],
        pdf_url="https://arxiv.org/pdf/2401.12345",
        abs_url="https://arxiv.org/abs/2401.12345",
    )


class TestCosineSimlarity:
    """Tests for the cosine similarity helper."""

    def test_identical_vectors(self) -> None:
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([[1.0, 2.0, 3.0]])
        result = _cosine_similarity(a, b)
        assert result[0] == pytest.approx(1.0, abs=1e-5)

    def test_orthogonal_vectors(self) -> None:
        a = np.array([1.0, 0.0])
        b = np.array([[0.0, 1.0]])
        result = _cosine_similarity(a, b)
        assert result[0] == pytest.approx(0.0, abs=1e-5)

    def test_multiple_candidates(self) -> None:
        a = np.array([1.0, 0.0])
        b = np.array([[1.0, 0.0], [0.0, 1.0], [0.707, 0.707]])
        result = _cosine_similarity(a, b)
        assert len(result) == 3
        assert result[0] == pytest.approx(1.0, abs=1e-2)
        assert result[1] == pytest.approx(0.0, abs=1e-2)
        assert result[2] == pytest.approx(0.707, abs=1e-2)

    def test_zero_vector(self) -> None:
        a = np.array([0.0, 0.0])
        b = np.array([[1.0, 0.0]])
        result = _cosine_similarity(a, b)
        assert result[0] == pytest.approx(0.0, abs=1e-5)


class TestScoreSemantic:
    """Tests for score_semantic with mocked SPECTER2."""

    @patch("research_pipeline.screening.embedding.compute_embeddings")
    def test_returns_scores_for_each_candidate(self, mock_embed: MagicMock) -> None:
        # 1 query + 3 candidates = 4 embeddings
        mock_embed.return_value = np.array(
            [
                [1.0, 0.0],  # query
                [0.9, 0.1],  # candidate 1 (similar)
                [0.5, 0.5],  # candidate 2 (mid)
                [0.0, 1.0],  # candidate 3 (different)
            ]
        )
        candidates = [_make_candidate(f"Paper {i}") for i in range(3)]
        scores = score_semantic("AI memory", candidates)

        assert len(scores) == 3
        assert all(0.0 <= s <= 1.0 for s in scores)
        # First candidate should be highest (most similar)
        assert scores[0] > scores[2]

    @patch("research_pipeline.screening.embedding.compute_embeddings")
    def test_empty_candidates(self, mock_embed: MagicMock) -> None:
        scores = score_semantic("topic", [])
        assert scores == []
        mock_embed.assert_not_called()

    @patch("research_pipeline.screening.embedding.compute_embeddings")
    def test_all_identical_returns_half(self, mock_embed: MagicMock) -> None:
        mock_embed.return_value = np.array(
            [
                [1.0, 0.0],
                [1.0, 0.0],
                [1.0, 0.0],
            ]
        )
        candidates = [_make_candidate(), _make_candidate()]
        scores = score_semantic("topic", candidates)
        assert all(s == pytest.approx(0.5) for s in scores)

    @patch("research_pipeline.screening.embedding.compute_embeddings")
    def test_passes_model_name(self, mock_embed: MagicMock) -> None:
        mock_embed.return_value = np.array([[1.0, 0.0], [0.5, 0.5]])
        candidates = [_make_candidate()]
        score_semantic("topic", candidates, model_name="custom/model")
        mock_embed.assert_called_once()
        assert mock_embed.call_args[1]["model_name"] == "custom/model"


class TestIsSpecter2Available:
    """Tests for availability check."""

    @patch.dict(
        "sys.modules",
        {"transformers": MagicMock(), "torch": MagicMock(), "adapters": MagicMock()},
    )
    def test_available_when_deps_installed(self) -> None:
        assert _is_specter2_available() is True

    def test_unavailable_checked(self) -> None:
        # Just verify the function runs without error
        result = _is_specter2_available()
        assert isinstance(result, bool)
