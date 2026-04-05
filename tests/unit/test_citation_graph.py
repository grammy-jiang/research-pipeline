"""Unit tests for sources.citation_graph."""

from unittest.mock import MagicMock, patch

import requests

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


class TestCitationGraphClient:
    """Tests for CitationGraphClient."""

    @patch.object(CitationGraphClient, "_api_get")
    def test_get_citations(self, mock_get: MagicMock) -> None:
        mock_get.return_value = {
            "data": [
                {"citingPaper": _make_s2_paper("p1", "2401.11111", "Citing 1")},
                {"citingPaper": _make_s2_paper("p2", "2401.22222", "Citing 2")},
            ]
        }
        client = CitationGraphClient()
        results = client.get_citations("2401.00001", limit=10)

        assert len(results) == 2
        assert results[0].arxiv_id == "2401.11111"
        assert results[1].arxiv_id == "2401.22222"

    @patch.object(CitationGraphClient, "_api_get")
    def test_get_references(self, mock_get: MagicMock) -> None:
        mock_get.return_value = {
            "data": [
                {"citedPaper": _make_s2_paper("p1", "2401.33333", "Ref 1")},
            ]
        }
        client = CitationGraphClient()
        results = client.get_references("2401.00001", limit=10)

        assert len(results) == 1
        assert results[0].arxiv_id == "2401.33333"

    @patch.object(CitationGraphClient, "_api_get")
    def test_get_citations_api_error(self, mock_get: MagicMock) -> None:
        mock_get.side_effect = requests.RequestException("fail")
        client = CitationGraphClient()
        results = client.get_citations("2401.00001", limit=10)
        assert results == []

    @patch.object(CitationGraphClient, "_api_get")
    def test_get_citations_empty(self, mock_get: MagicMock) -> None:
        mock_get.return_value = {"data": []}
        client = CitationGraphClient()
        results = client.get_citations("2401.00001", limit=10)
        assert results == []

    @patch.object(CitationGraphClient, "_api_get")
    def test_fetch_related_deduplicates(self, mock_get: MagicMock) -> None:
        # Same paper appears in citations of both seeds
        mock_get.return_value = {
            "data": [
                {"citingPaper": _make_s2_paper("p1", "2401.11111", "Shared")},
                {"citedPaper": _make_s2_paper("p1", "2401.11111", "Shared")},
            ]
        }
        client = CitationGraphClient()
        results = client.fetch_related(
            ["2401.00001", "2401.00002"], direction="both", limit_per_paper=5
        )

        # Should be deduplicated
        arxiv_ids = [r.arxiv_id for r in results]
        assert arxiv_ids.count("2401.11111") == 1

    @patch.object(CitationGraphClient, "_api_get")
    def test_fetch_related_citations_only(self, mock_get: MagicMock) -> None:
        mock_get.return_value = {
            "data": [
                {"citingPaper": _make_s2_paper("p1", "2401.44444", "Citer")},
            ]
        }
        client = CitationGraphClient()
        results = client.fetch_related(
            ["2401.00001"], direction="citations", limit_per_paper=5
        )
        assert len(results) == 1

    def test_parse_paper_no_arxiv_id(self) -> None:
        client = CitationGraphClient()
        paper = _make_s2_paper()
        paper["externalIds"] = {}
        candidate = client._parse_paper(paper)
        assert candidate.arxiv_id.startswith("s2-")

    def test_parse_paper_no_abstract(self) -> None:
        client = CitationGraphClient()
        paper = _make_s2_paper()
        paper["abstract"] = None
        candidate = client._parse_paper(paper)
        assert candidate.abstract == ""
