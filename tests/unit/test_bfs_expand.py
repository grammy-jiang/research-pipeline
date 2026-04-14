"""Tests for CitationGraphClient.bfs_expand() — BFS citation graph expansion."""

from datetime import UTC, datetime
from unittest.mock import patch

from research_pipeline.models.candidate import CandidateRecord
from research_pipeline.sources.citation_graph import (
    CitationGraphClient,
    _bm25_score_text,
)


def _make_candidate(
    arxiv_id: str, title: str = "", abstract: str = ""
) -> CandidateRecord:
    return CandidateRecord(
        arxiv_id=arxiv_id,
        version="v1",
        title=title or f"Paper {arxiv_id}",
        authors=["Author"],
        published=datetime(2024, 1, 1, tzinfo=UTC),
        updated=datetime(2024, 1, 1, tzinfo=UTC),
        categories=["cs.AI"],
        primary_category="cs.AI",
        abstract=abstract or f"Abstract for paper {arxiv_id}",
        abs_url=f"https://arxiv.org/abs/{arxiv_id}",
        pdf_url=f"https://arxiv.org/pdf/{arxiv_id}",
        source="semantic_scholar",
    )


# -- _bm25_score_text unit tests --


class TestBm25ScoreText:
    def test_empty_text_returns_zero(self) -> None:
        assert _bm25_score_text("", ["transformer"]) == 0.0

    def test_no_match_returns_zero(self) -> None:
        assert _bm25_score_text("deep learning models", ["quantum"]) == 0.0

    def test_single_match(self) -> None:
        score = _bm25_score_text("transformer architecture", ["transformer"])
        assert score > 0.0

    def test_repeated_term_saturates(self) -> None:
        score_one = _bm25_score_text("transformer", ["transformer"])
        score_many = _bm25_score_text(
            "transformer transformer transformer", ["transformer"]
        )
        assert score_many > score_one
        # BM25 saturation: 3/(3+1.5)=0.667 vs 1/(1+1.5)=0.4
        assert score_many < 1.0

    def test_case_insensitive(self) -> None:
        score = _bm25_score_text("Transformer Architecture", ["transformer"])
        assert score > 0.0

    def test_multiple_query_terms(self) -> None:
        score = _bm25_score_text(
            "transformer attention mechanism", ["transformer", "attention"]
        )
        single = _bm25_score_text("transformer attention mechanism", ["transformer"])
        assert score > single


# -- bfs_expand tests --


_PATCH_CIT = (
    "research_pipeline.sources.citation_graph.CitationGraphClient.get_citations"
)
_PATCH_REF = (
    "research_pipeline.sources.citation_graph.CitationGraphClient.get_references"
)


class TestBfsExpand:
    def _client(self) -> CitationGraphClient:
        return CitationGraphClient(api_key="", rate_limiter=None, session=None)

    def test_empty_seed_ids_returns_empty(self) -> None:
        client = self._client()
        result = client.bfs_expand(seed_ids=[], query_terms=["transformer"])
        assert result == []

    @patch(_PATCH_REF, return_value=[])
    @patch(_PATCH_CIT)
    def test_depth_one_returns_first_hop(
        self, mock_cit: object, mock_ref: object
    ) -> None:
        hop1 = [
            _make_candidate("1001", title="transformer neural network"),
            _make_candidate("1002", title="attention mechanism"),
        ]
        mock_cit.return_value = hop1  # type: ignore[union-attr]
        mock_ref.return_value = []  # type: ignore[union-attr]

        client = self._client()
        result = client.bfs_expand(
            seed_ids=["seed1"],
            query_terms=["transformer"],
            max_depth=1,
            top_k_per_hop=10,
        )
        assert len(result) == 2
        ids = {c.arxiv_id for c in result}
        assert "1001" in ids
        assert "1002" in ids

    @patch(_PATCH_REF, return_value=[])
    @patch(_PATCH_CIT)
    def test_depth_two_returns_multi_hop(
        self, mock_cit: object, mock_ref: object
    ) -> None:
        hop1 = [_make_candidate("1001", title="transformer model architecture")]
        hop2 = [_make_candidate("2001", title="transformer decoder layer")]

        def side_effect(paper_id: str, limit: int = 20) -> list[CandidateRecord]:
            if paper_id == "seed1":
                return hop1
            if paper_id == "1001":
                return hop2
            return []

        mock_cit.side_effect = side_effect  # type: ignore[union-attr]
        mock_ref.return_value = []  # type: ignore[union-attr]

        client = self._client()
        result = client.bfs_expand(
            seed_ids=["seed1"],
            query_terms=["transformer"],
            max_depth=2,
            top_k_per_hop=10,
        )
        ids = {c.arxiv_id for c in result}
        assert "1001" in ids
        assert "2001" in ids

    @patch(_PATCH_REF, return_value=[])
    @patch(_PATCH_CIT)
    def test_deduplication_across_hops(
        self, mock_cit: object, mock_ref: object
    ) -> None:
        """A paper seen in hop 1 should not appear again in hop 2."""
        dup_candidate = _make_candidate("1001", title="transformer model")

        def side_effect(paper_id: str, limit: int = 20) -> list[CandidateRecord]:
            # Both seed and hop-1 paper return the same candidate
            return [dup_candidate]

        mock_cit.side_effect = side_effect  # type: ignore[union-attr]
        mock_ref.return_value = []  # type: ignore[union-attr]

        client = self._client()
        result = client.bfs_expand(
            seed_ids=["seed1"],
            query_terms=["transformer"],
            max_depth=2,
            top_k_per_hop=10,
        )
        arxiv_ids = [c.arxiv_id for c in result]
        assert arxiv_ids.count("1001") == 1

    @patch(_PATCH_REF, return_value=[])
    @patch(_PATCH_CIT)
    def test_bm25_pruning_keeps_relevant(
        self, mock_cit: object, mock_ref: object
    ) -> None:
        """top_k_per_hop should keep the higher-scoring papers."""
        candidates = [
            _make_candidate("low1", title="unrelated biology study", abstract="cells"),
            _make_candidate(
                "high1",
                title="transformer attention mechanism",
                abstract="transformer attention model",
            ),
            _make_candidate(
                "low2",
                title="cooking recipes overview",
                abstract="food preparation",
            ),
            _make_candidate(
                "high2",
                title="transformer encoder decoder",
                abstract="transformer architecture",
            ),
        ]
        mock_cit.return_value = candidates  # type: ignore[union-attr]
        mock_ref.return_value = []  # type: ignore[union-attr]

        client = self._client()
        result = client.bfs_expand(
            seed_ids=["seed1"],
            query_terms=["transformer", "attention"],
            max_depth=1,
            top_k_per_hop=2,
        )
        ids = {c.arxiv_id for c in result}
        assert "high1" in ids
        assert "high2" in ids
        assert "low1" not in ids
        assert "low2" not in ids

    @patch(_PATCH_REF)
    @patch(_PATCH_CIT)
    def test_direction_citations_only(self, mock_cit: object, mock_ref: object) -> None:
        mock_cit.return_value = [  # type: ignore[union-attr]
            _make_candidate("c1", title="transformer model")
        ]
        mock_ref.return_value = [  # type: ignore[union-attr]
            _make_candidate("r1", title="transformer reference")
        ]

        client = self._client()
        result = client.bfs_expand(
            seed_ids=["seed1"],
            query_terms=["transformer"],
            max_depth=1,
            top_k_per_hop=10,
            direction="citations",
        )
        ids = {c.arxiv_id for c in result}
        assert "c1" in ids
        assert "r1" not in ids
        mock_ref.assert_not_called()  # type: ignore[union-attr]

    @patch(_PATCH_REF)
    @patch(_PATCH_CIT)
    def test_direction_references_only(
        self, mock_cit: object, mock_ref: object
    ) -> None:
        mock_cit.return_value = [  # type: ignore[union-attr]
            _make_candidate("c1", title="transformer citation")
        ]
        mock_ref.return_value = [  # type: ignore[union-attr]
            _make_candidate("r1", title="transformer reference")
        ]

        client = self._client()
        result = client.bfs_expand(
            seed_ids=["seed1"],
            query_terms=["transformer"],
            max_depth=1,
            top_k_per_hop=10,
            direction="references",
        )
        ids = {c.arxiv_id for c in result}
        assert "r1" in ids
        assert "c1" not in ids
        mock_cit.assert_not_called()  # type: ignore[union-attr]

    @patch(_PATCH_REF, return_value=[])
    @patch(_PATCH_CIT, return_value=[])
    def test_empty_graph_terminates(self, mock_cit: object, mock_ref: object) -> None:
        """BFS should terminate early when no candidates are found."""
        client = self._client()
        result = client.bfs_expand(
            seed_ids=["seed1"],
            query_terms=["transformer"],
            max_depth=3,
            top_k_per_hop=10,
        )
        assert result == []
