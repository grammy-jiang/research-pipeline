"""Tests for the ArxivSource search adapter."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from research_pipeline.models.candidate import CandidateRecord
from research_pipeline.sources.arxiv_source import ArxivSource


def _make_candidate(arxiv_id: str) -> CandidateRecord:
    """Build a minimal CandidateRecord for testing."""
    return CandidateRecord(
        arxiv_id=arxiv_id,
        version="v1",
        title=f"Paper {arxiv_id}",
        authors=["Author A"],
        abstract="Abstract text.",
        categories=["cs.AI"],
        primary_category="cs.AI",
        published="2024-01-01",
        updated="2024-01-01",
        abs_url=f"https://arxiv.org/abs/{arxiv_id}",
        pdf_url=f"https://arxiv.org/pdf/{arxiv_id}",
    )


class TestArxivSourceName:
    """Tests for the name property."""

    def test_name_returns_arxiv(self) -> None:
        src = ArxivSource.__new__(ArxivSource)
        src._client = MagicMock()
        assert src.name == "arxiv"


class TestArxivSourceSearch:
    """Tests for ArxivSource.search()."""

    @patch("research_pipeline.sources.arxiv_source.dedup_across_queries")
    @patch("research_pipeline.sources.arxiv_source.build_query_from_plan")
    @patch("research_pipeline.sources.arxiv_source.date_window")
    def test_search_calls_client_per_query(
        self,
        mock_date_window: MagicMock,
        mock_build: MagicMock,
        mock_dedup: MagicMock,
    ) -> None:
        """Client.search is called once per generated query."""
        mock_date_window.return_value = ("2024-01-01", "2024-06-01")
        mock_build.return_value = ["q1", "q2", "q3"]

        c1 = _make_candidate("2401.00001")
        c2 = _make_candidate("2401.00002")
        c3 = _make_candidate("2401.00003")

        mock_client = MagicMock()
        mock_client.search.side_effect = [
            ([c1], {}),
            ([c2], {}),
            ([c3], {}),
        ]
        mock_dedup.return_value = [c1, c2, c3]

        src = ArxivSource.__new__(ArxivSource)
        src._client = mock_client

        result = src.search(
            topic="AI agents",
            must_terms=["agent"],
            nice_terms=["memory"],
        )

        assert mock_client.search.call_count == 3
        assert result == [c1, c2, c3]

    @patch("research_pipeline.sources.arxiv_source.dedup_across_queries")
    @patch("research_pipeline.sources.arxiv_source.build_query_from_plan")
    @patch("research_pipeline.sources.arxiv_source.date_window")
    def test_dedup_is_applied(
        self,
        mock_date_window: MagicMock,
        mock_build: MagicMock,
        mock_dedup: MagicMock,
    ) -> None:
        """dedup_across_queries is called with the collected candidate lists."""
        mock_date_window.return_value = ("2024-01-01", "2024-06-01")
        mock_build.return_value = ["q1"]

        c1 = _make_candidate("2401.00001")
        mock_client = MagicMock()
        mock_client.search.return_value = ([c1], {})
        mock_dedup.return_value = [c1]

        src = ArxivSource.__new__(ArxivSource)
        src._client = mock_client

        src.search(topic="t", must_terms=["a"], nice_terms=["b"])

        mock_dedup.assert_called_once()
        args = mock_dedup.call_args[0][0]
        assert len(args) == 1
        assert args[0] == [c1]

    @patch("research_pipeline.sources.arxiv_source.dedup_across_queries")
    @patch("research_pipeline.sources.arxiv_source.build_query_from_plan")
    @patch("research_pipeline.sources.arxiv_source.date_window")
    def test_explicit_dates_skip_date_window(
        self,
        mock_date_window: MagicMock,
        mock_build: MagicMock,
        mock_dedup: MagicMock,
    ) -> None:
        """Providing date_from and date_to skips date_window call."""
        mock_build.return_value = ["q1"]
        mock_client = MagicMock()
        mock_client.search.return_value = ([], {})
        mock_dedup.return_value = []

        src = ArxivSource.__new__(ArxivSource)
        src._client = mock_client

        src.search(
            topic="t",
            must_terms=[],
            nice_terms=[],
            date_from="2024-01-01",
            date_to="2024-06-01",
        )

        mock_date_window.assert_not_called()

    @patch("research_pipeline.sources.arxiv_source.dedup_across_queries")
    @patch("research_pipeline.sources.arxiv_source.build_query_from_plan")
    @patch("research_pipeline.sources.arxiv_source.date_window")
    def test_empty_queries_returns_empty(
        self,
        mock_date_window: MagicMock,
        mock_build: MagicMock,
        mock_dedup: MagicMock,
    ) -> None:
        """When build_query_from_plan returns no queries, result is empty."""
        mock_date_window.return_value = ("2024-01-01", "2024-06-01")
        mock_build.return_value = []
        mock_dedup.return_value = []

        src = ArxivSource.__new__(ArxivSource)
        src._client = MagicMock()

        result = src.search(topic="t", must_terms=[], nice_terms=[])

        assert result == []
        src._client.search.assert_not_called()
