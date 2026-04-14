"""Unit tests for backward-preference citation expansion."""

from unittest.mock import MagicMock, patch

from research_pipeline.sources.citation_graph import CitationGraphClient


def _make_s2_paper(
    paper_id: str = "abc123",
    arxiv_id: str = "2401.12345",
    title: str = "Test Paper",
) -> dict:  # type: ignore[type-arg]
    """Build a minimal S2 paper dict."""
    return {
        "paperId": paper_id,
        "externalIds": {"ArXiv": arxiv_id, "DOI": "10.1234/test"},
        "title": title,
        "abstract": "Abstract text",
        "year": 2024,
        "authors": [{"name": "Author One"}],
        "citationCount": 10,
        "influentialCitationCount": 2,
        "venue": "NeurIPS",
        "publicationDate": "2024-06-15",
        "openAccessPdf": {"url": f"https://arxiv.org/pdf/{arxiv_id}"},
    }


class TestBackwardPreference:
    """Tests for reference_boost in fetch_related."""

    @patch.object(CitationGraphClient, "get_references")
    @patch.object(CitationGraphClient, "get_citations")
    def test_default_equal_limits(
        self, mock_cit: MagicMock, mock_ref: MagicMock
    ) -> None:
        """Without reference_boost, both directions get equal limits."""
        mock_cit.return_value = []
        mock_ref.return_value = []
        client = CitationGraphClient()
        client.fetch_related(["2401.00001"], direction="both", limit_per_paper=30)

        mock_cit.assert_called_once_with("2401.00001", 30)
        mock_ref.assert_called_once_with("2401.00001", 30)

    @patch.object(CitationGraphClient, "get_references")
    @patch.object(CitationGraphClient, "get_citations")
    def test_reference_boost_doubles_ref_limit(
        self, mock_cit: MagicMock, mock_ref: MagicMock
    ) -> None:
        """reference_boost=2.0 doubles the reference limit."""
        mock_cit.return_value = []
        mock_ref.return_value = []
        client = CitationGraphClient()
        client.fetch_related(
            ["2401.00001"],
            direction="both",
            limit_per_paper=30,
            reference_boost=2.0,
        )

        mock_cit.assert_called_once_with("2401.00001", 30)
        mock_ref.assert_called_once_with("2401.00001", 60)

    @patch.object(CitationGraphClient, "get_references")
    @patch.object(CitationGraphClient, "get_citations")
    def test_boost_ignored_for_citations_only(
        self, mock_cit: MagicMock, mock_ref: MagicMock
    ) -> None:
        """reference_boost is ignored when direction='citations'."""
        mock_cit.return_value = []
        client = CitationGraphClient()
        client.fetch_related(
            ["2401.00001"],
            direction="citations",
            limit_per_paper=30,
            reference_boost=3.0,
        )

        mock_cit.assert_called_once_with("2401.00001", 30)
        mock_ref.assert_not_called()

    @patch.object(CitationGraphClient, "get_references")
    @patch.object(CitationGraphClient, "get_citations")
    def test_boost_ignored_for_references_only(
        self, mock_cit: MagicMock, mock_ref: MagicMock
    ) -> None:
        """reference_boost is ignored when direction='references'."""
        mock_ref.return_value = []
        client = CitationGraphClient()
        client.fetch_related(
            ["2401.00001"],
            direction="references",
            limit_per_paper=30,
            reference_boost=3.0,
        )

        mock_ref.assert_called_once_with("2401.00001", 30)
        mock_cit.assert_not_called()

    @patch.object(CitationGraphClient, "get_references")
    @patch.object(CitationGraphClient, "get_citations")
    def test_boost_1_0_keeps_equal(
        self, mock_cit: MagicMock, mock_ref: MagicMock
    ) -> None:
        """reference_boost=1.0 keeps both limits equal (no-op)."""
        mock_cit.return_value = []
        mock_ref.return_value = []
        client = CitationGraphClient()
        client.fetch_related(
            ["2401.00001"],
            direction="both",
            limit_per_paper=50,
            reference_boost=1.0,
        )

        mock_cit.assert_called_once_with("2401.00001", 50)
        mock_ref.assert_called_once_with("2401.00001", 50)
