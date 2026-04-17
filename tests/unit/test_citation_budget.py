"""Unit tests for citation budget stopping criteria in bfs_expand."""

from unittest.mock import MagicMock, patch

from research_pipeline.sources.citation_graph import CitationGraphClient


def _make_candidate(arxiv_id: str, title: str = "Paper", abstract: str = ""):
    """Create a minimal CandidateRecord-like object."""
    from datetime import UTC, datetime

    from research_pipeline.models.candidate import CandidateRecord

    return CandidateRecord(
        arxiv_id=arxiv_id,
        version="v1",
        title=title,
        authors=["Author"],
        published=datetime(2024, 1, 1, tzinfo=UTC),
        updated=datetime(2024, 1, 1, tzinfo=UTC),
        categories=[],
        primary_category="cs.AI",
        abstract=abstract,
        abs_url="",
        pdf_url="",
        source="semantic_scholar",
    )


def _make_hop_candidates(prefix: str, count: int, query_term: str = ""):
    """Generate a list of candidates for one BFS hop."""
    candidates = []
    for i in range(count):
        title = f"{prefix} paper {i}"
        if query_term:
            title += f" about {query_term}"
        candidates.append(_make_candidate(f"{prefix}.{i:05d}", title=title))
    return candidates


class TestBfsExpandBudget:
    """Tests for max_total_papers budget stopping."""

    @patch.object(CitationGraphClient, "get_citations")
    @patch.object(CitationGraphClient, "get_references")
    def test_budget_zero_means_no_limit(
        self, mock_refs: MagicMock, mock_cites: MagicMock
    ) -> None:
        """max_total_papers=0 should not impose any limit."""
        hop1 = _make_hop_candidates("hop1", 5, "transformer")
        hop2 = _make_hop_candidates("hop2", 5, "transformer")
        mock_cites.side_effect = [hop1, *([hop2] * 5)]
        mock_refs.side_effect = [[], *([[]] * 5)]

        client = CitationGraphClient()
        results = client.bfs_expand(
            seed_ids=["seed.001"],
            query_terms=["transformer"],
            max_depth=2,
            top_k_per_hop=5,
            direction="citations",
            max_total_papers=0,
        )
        assert len(results) >= 5

    @patch.object(CitationGraphClient, "get_citations")
    @patch.object(CitationGraphClient, "get_references")
    def test_budget_caps_total_papers(
        self, mock_refs: MagicMock, mock_cites: MagicMock
    ) -> None:
        """Should stop when budget is reached."""
        hop1 = _make_hop_candidates("hop1", 10, "transformer")
        hop2 = _make_hop_candidates("hop2", 10, "transformer")
        mock_cites.side_effect = [hop1, *([hop2] * 10)]
        mock_refs.return_value = []

        client = CitationGraphClient()
        results = client.bfs_expand(
            seed_ids=["seed.001"],
            query_terms=["transformer"],
            max_depth=3,
            top_k_per_hop=10,
            direction="citations",
            max_total_papers=3,
        )
        assert len(results) <= 3

    @patch.object(CitationGraphClient, "get_citations")
    @patch.object(CitationGraphClient, "get_references")
    def test_budget_stops_before_second_hop(
        self, mock_refs: MagicMock, mock_cites: MagicMock
    ) -> None:
        """If budget is filled in hop 1, hop 2 should not execute."""
        hop1 = _make_hop_candidates("hop1", 5, "attention")
        mock_cites.side_effect = [hop1]
        mock_refs.return_value = []

        client = CitationGraphClient()
        results = client.bfs_expand(
            seed_ids=["seed.001"],
            query_terms=["attention"],
            max_depth=3,
            top_k_per_hop=5,
            direction="citations",
            max_total_papers=5,
        )
        assert len(results) == 5
        # Only 1 call to get_citations (hop 1 only)
        assert mock_cites.call_count == 1

    @patch.object(CitationGraphClient, "get_citations")
    @patch.object(CitationGraphClient, "get_references")
    def test_budget_partial_second_hop(
        self, mock_refs: MagicMock, mock_cites: MagicMock
    ) -> None:
        """Budget should limit how many papers are taken from hop 2."""
        hop1 = _make_hop_candidates("hop1", 3, "neural")
        hop2 = _make_hop_candidates("hop2", 10, "neural")
        mock_cites.side_effect = [hop1, *([hop2] * 3)]
        mock_refs.return_value = []

        client = CitationGraphClient()
        results = client.bfs_expand(
            seed_ids=["seed.001"],
            query_terms=["neural"],
            max_depth=2,
            top_k_per_hop=10,
            direction="citations",
            max_total_papers=5,
        )
        assert len(results) == 5


class TestBfsExpandDiminishingReturns:
    """Tests for min_new_per_hop diminishing returns stopping."""

    @patch.object(CitationGraphClient, "get_citations")
    @patch.object(CitationGraphClient, "get_references")
    def test_min_new_zero_means_no_check(
        self, mock_refs: MagicMock, mock_cites: MagicMock
    ) -> None:
        """min_new_per_hop=0 should not check diminishing returns."""
        hop1 = _make_hop_candidates("hop1", 1, "graph")
        mock_cites.side_effect = [hop1, []]
        mock_refs.return_value = []

        client = CitationGraphClient()
        results = client.bfs_expand(
            seed_ids=["seed.001"],
            query_terms=["graph"],
            max_depth=2,
            top_k_per_hop=10,
            direction="citations",
            min_new_per_hop=0,
        )
        assert len(results) == 1

    @patch.object(CitationGraphClient, "get_citations")
    @patch.object(CitationGraphClient, "get_references")
    def test_stops_when_too_few_new_candidates(
        self, mock_refs: MagicMock, mock_cites: MagicMock
    ) -> None:
        """Should stop when hop yields fewer than min_new_per_hop."""
        hop1 = _make_hop_candidates("hop1", 2, "bert")
        mock_cites.side_effect = [hop1]
        mock_refs.return_value = []

        client = CitationGraphClient()
        results = client.bfs_expand(
            seed_ids=["seed.001"],
            query_terms=["bert"],
            max_depth=3,
            top_k_per_hop=10,
            direction="citations",
            min_new_per_hop=5,
        )
        # Hop 1 had only 2 candidates, less than min_new=5 → stop immediately
        assert len(results) == 0

    @patch.object(CitationGraphClient, "get_citations")
    @patch.object(CitationGraphClient, "get_references")
    def test_continues_when_enough_new_candidates(
        self, mock_refs: MagicMock, mock_cites: MagicMock
    ) -> None:
        """Should continue when hop yields at least min_new_per_hop."""
        hop1 = _make_hop_candidates("hop1", 10, "llm")
        hop2 = _make_hop_candidates("hop2", 8, "llm")
        mock_cites.side_effect = [hop1, *([hop2] * 10)]
        mock_refs.return_value = []

        client = CitationGraphClient()
        results = client.bfs_expand(
            seed_ids=["seed.001"],
            query_terms=["llm"],
            max_depth=2,
            top_k_per_hop=5,
            direction="citations",
            min_new_per_hop=3,
        )
        # Both hops should execute (10 >= 3 and 8 >= 3)
        assert len(results) >= 5

    @patch.object(CitationGraphClient, "get_citations")
    @patch.object(CitationGraphClient, "get_references")
    def test_diminishing_returns_at_second_hop(
        self, mock_refs: MagicMock, mock_cites: MagicMock
    ) -> None:
        """First hop passes threshold, second hop triggers stop."""
        hop1 = _make_hop_candidates("hop1", 10, "rag")
        hop2 = _make_hop_candidates("hop2", 1, "rag")
        mock_cites.side_effect = [hop1, *([hop2] * 10)]
        mock_refs.return_value = []

        client = CitationGraphClient()
        results = client.bfs_expand(
            seed_ids=["seed.001"],
            query_terms=["rag"],
            max_depth=3,
            top_k_per_hop=5,
            direction="citations",
            min_new_per_hop=3,
        )
        # Only hop 1 results kept (hop 2 had 1 < 3)
        assert len(results) == 5


class TestBfsExpandCombinedBudget:
    """Tests for combined budget + diminishing returns."""

    @patch.object(CitationGraphClient, "get_citations")
    @patch.object(CitationGraphClient, "get_references")
    def test_budget_and_diminishing_both_active(
        self, mock_refs: MagicMock, mock_cites: MagicMock
    ) -> None:
        """Both criteria should work together."""
        hop1 = _make_hop_candidates("hop1", 8, "memory")
        hop2 = _make_hop_candidates("hop2", 1, "memory")
        mock_cites.side_effect = [hop1, *([hop2] * 10)]
        mock_refs.return_value = []

        client = CitationGraphClient()
        results = client.bfs_expand(
            seed_ids=["seed.001"],
            query_terms=["memory"],
            max_depth=3,
            top_k_per_hop=5,
            direction="citations",
            max_total_papers=10,
            min_new_per_hop=3,
        )
        # Hop 1: 8 candidates, keeps 5 (under budget 10, above min 3)
        # Hop 2: only 1 new < min 3 → stop
        assert len(results) == 5

    @patch.object(CitationGraphClient, "get_citations")
    @patch.object(CitationGraphClient, "get_references")
    def test_empty_seeds_returns_empty(
        self, mock_refs: MagicMock, mock_cites: MagicMock
    ) -> None:
        """Empty seed list should return empty results."""
        client = CitationGraphClient()
        results = client.bfs_expand(
            seed_ids=[],
            query_terms=["test"],
            max_depth=2,
            max_total_papers=10,
            min_new_per_hop=3,
        )
        assert results == []


class TestBfsExpandBackwardCompat:
    """Tests ensuring backward compatibility with existing API."""

    @patch.object(CitationGraphClient, "get_citations")
    @patch.object(CitationGraphClient, "get_references")
    def test_existing_params_still_work(
        self, mock_refs: MagicMock, mock_cites: MagicMock
    ) -> None:
        """Calling bfs_expand without new params should work as before."""
        hop1 = _make_hop_candidates("hop1", 5, "test")
        mock_cites.side_effect = [hop1, *([[]] * 5)]
        mock_refs.return_value = []

        client = CitationGraphClient()
        results = client.bfs_expand(
            seed_ids=["seed.001"],
            query_terms=["test"],
            max_depth=2,
            top_k_per_hop=5,
            direction="citations",
        )
        assert len(results) >= 0  # Just verify it runs

    @patch.object(CitationGraphClient, "get_citations")
    @patch.object(CitationGraphClient, "get_references")
    def test_both_directions_with_budget(
        self, mock_refs: MagicMock, mock_cites: MagicMock
    ) -> None:
        """Budget should work with direction='both'."""
        cites = _make_hop_candidates("cite", 5, "agent")
        refs = _make_hop_candidates("ref", 5, "agent")
        mock_cites.return_value = cites
        mock_refs.return_value = refs

        client = CitationGraphClient()
        results = client.bfs_expand(
            seed_ids=["seed.001"],
            query_terms=["agent"],
            max_depth=1,
            top_k_per_hop=10,
            direction="both",
            max_total_papers=4,
        )
        assert len(results) <= 4
