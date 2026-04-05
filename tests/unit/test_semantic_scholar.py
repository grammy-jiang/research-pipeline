"""Unit tests for sources.semantic_scholar_source."""

from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import requests

from research_pipeline.sources.semantic_scholar_source import SemanticScholarSource


def _make_s2_paper(
    paper_id: str = "abc123",
    arxiv_id: str = "2401.12345",
    title: str = "Test Paper",
    abstract: str = "Test abstract",
    year: int = 2024,
    citation_count: int = 10,
) -> dict:  # type: ignore[type-arg]
    """Build a minimal S2 paper dict for testing."""
    return {
        "paperId": paper_id,
        "externalIds": {"ArXiv": arxiv_id, "DOI": "10.1234/test"},
        "title": title,
        "abstract": abstract,
        "year": year,
        "authors": [{"name": "Author One"}, {"name": "Author Two"}],
        "citationCount": citation_count,
        "influentialCitationCount": 3,
        "venue": "NeurIPS",
        "publicationDate": f"{year}-06-15",
        "openAccessPdf": {"url": f"https://arxiv.org/pdf/{arxiv_id}"},
    }


class TestSemanticScholarSource:
    """Tests for SemanticScholarSource."""

    def test_name(self) -> None:
        source = SemanticScholarSource()
        assert source.name == "semantic_scholar"

    def test_parse_paper_full(self) -> None:
        source = SemanticScholarSource()
        paper = _make_s2_paper()
        candidate = source._parse_paper(paper)

        assert candidate.arxiv_id == "2401.12345"
        assert candidate.title == "Test Paper"
        assert candidate.source == "semantic_scholar"
        assert candidate.doi == "10.1234/test"
        assert candidate.semantic_scholar_id == "abc123"
        assert candidate.citation_count == 10
        assert candidate.influential_citation_count == 3
        assert candidate.venue == "NeurIPS"
        assert candidate.year == 2024
        assert len(candidate.authors) == 2

    def test_parse_paper_no_arxiv_id(self) -> None:
        source = SemanticScholarSource()
        paper = _make_s2_paper(arxiv_id="")
        paper["externalIds"] = {"DOI": "10.1234/test"}
        candidate = source._parse_paper(paper)

        assert candidate.arxiv_id.startswith("s2-")
        assert candidate.doi == "10.1234/test"

    def test_parse_paper_no_abstract(self) -> None:
        source = SemanticScholarSource()
        paper = _make_s2_paper()
        paper["abstract"] = None
        candidate = source._parse_paper(paper)

        assert candidate.abstract == ""

    def test_parse_paper_no_publication_date(self) -> None:
        source = SemanticScholarSource()
        paper = _make_s2_paper()
        paper["publicationDate"] = None
        candidate = source._parse_paper(paper)

        assert candidate.published == datetime(2024, 1, 1, tzinfo=UTC)

    def test_parse_paper_no_year(self) -> None:
        source = SemanticScholarSource()
        paper = _make_s2_paper()
        paper["publicationDate"] = None
        paper["year"] = None
        candidate = source._parse_paper(paper)

        assert candidate.published.tzinfo == UTC

    @patch.object(SemanticScholarSource, "_api_get")
    def test_search_returns_candidates(self, mock_get: MagicMock) -> None:
        mock_get.return_value = {
            "total": 2,
            "data": [
                _make_s2_paper("p1", "2401.11111", "Paper One"),
                _make_s2_paper("p2", "2401.22222", "Paper Two"),
            ],
        }
        source = SemanticScholarSource()
        results = source.search(
            topic="transformers",
            must_terms=["attention", "model"],
            nice_terms=["efficient"],
            max_results=10,
        )

        assert len(results) == 2
        assert results[0].arxiv_id == "2401.11111"
        assert results[1].arxiv_id == "2401.22222"

    @patch.object(SemanticScholarSource, "_api_get")
    def test_search_empty_results(self, mock_get: MagicMock) -> None:
        mock_get.return_value = {"total": 0, "data": []}
        source = SemanticScholarSource()
        results = source.search(topic="nonexistent", must_terms=[], nice_terms=[])
        assert results == []

    @patch.object(SemanticScholarSource, "_api_get")
    def test_search_handles_api_error(self, mock_get: MagicMock) -> None:
        mock_get.side_effect = requests.RequestException("API error")
        source = SemanticScholarSource()
        results = source.search(
            topic="transformers", must_terms=["test"], nice_terms=[]
        )
        assert results == []

    @patch.object(SemanticScholarSource, "_api_get")
    def test_search_with_year_range(self, mock_get: MagicMock) -> None:
        mock_get.return_value = {"total": 0, "data": []}
        source = SemanticScholarSource()
        source.search(
            topic="test",
            must_terms=["term"],
            nice_terms=[],
            date_from="2023-01-01",
            date_to="2024-12-31",
        )

        call_params = mock_get.call_args[0][1]
        assert call_params["year"] == "2023-2024"

    def test_api_key_set_in_headers(self) -> None:
        source = SemanticScholarSource(api_key="test-key-123")
        assert source._session.headers.get("x-api-key") == "test-key-123"
