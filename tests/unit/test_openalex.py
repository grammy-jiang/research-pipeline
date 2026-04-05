"""Unit tests for sources.openalex_source."""

from unittest.mock import MagicMock, patch

import requests

from research_pipeline.sources.openalex_source import (
    OpenAlexSource,
    _reconstruct_abstract,
)


def _make_openalex_work(
    openalex_id: str = "W12345",
    title: str = "Test Paper",
    year: int = 2024,
    doi: str = "10.1234/test",
) -> dict:  # type: ignore[type-arg]
    """Build a minimal OpenAlex work dict for testing."""
    return {
        "id": f"https://openalex.org/{openalex_id}",
        "title": title,
        "doi": f"https://doi.org/{doi}",
        "publication_date": f"{year}-06-15",
        "publication_year": year,
        "authorships": [
            {"author": {"display_name": "Author One"}},
            {"author": {"display_name": "Author Two"}},
        ],
        "cited_by_count": 25,
        "abstract_inverted_index": {"This": [0], "is": [1], "a": [2], "test": [3]},
        "locations": [
            {
                "source": {"display_name": "arXiv"},
                "landing_page_url": "https://arxiv.org/abs/2401.12345",
                "pdf_url": "https://arxiv.org/pdf/2401.12345",
            }
        ],
        "primary_location": {
            "source": {"display_name": "NeurIPS"},
        },
        "best_oa_location": {
            "pdf_url": "https://arxiv.org/pdf/2401.12345",
        },
    }


class TestReconstructAbstract:
    """Tests for abstract reconstruction."""

    def test_basic(self) -> None:
        index = {"Hello": [0], "world": [1]}
        assert _reconstruct_abstract(index) == "Hello world"

    def test_empty(self) -> None:
        assert _reconstruct_abstract({}) == ""

    def test_out_of_order(self) -> None:
        index = {"world": [1], "Hello": [0]}
        assert _reconstruct_abstract(index) == "Hello world"


class TestOpenAlexSource:
    """Tests for OpenAlexSource."""

    def test_name(self) -> None:
        source = OpenAlexSource()
        assert source.name == "openalex"

    def test_parse_work_full(self) -> None:
        source = OpenAlexSource()
        work = _make_openalex_work()
        candidate = source._parse_work(work)

        assert candidate.arxiv_id == "2401.12345"
        assert candidate.title == "Test Paper"
        assert candidate.source == "openalex"
        assert candidate.doi == "10.1234/test"
        assert candidate.openalex_id == "W12345"
        assert candidate.citation_count == 25
        assert candidate.venue == "NeurIPS"
        assert candidate.year == 2024
        assert candidate.abstract == "This is a test"
        assert len(candidate.authors) == 2

    def test_parse_work_no_arxiv_id(self) -> None:
        source = OpenAlexSource()
        work = _make_openalex_work()
        work["locations"] = []
        candidate = source._parse_work(work)

        assert candidate.arxiv_id.startswith("oalex-")

    def test_parse_work_no_abstract(self) -> None:
        source = OpenAlexSource()
        work = _make_openalex_work()
        work["abstract_inverted_index"] = None
        candidate = source._parse_work(work)

        assert candidate.abstract == ""

    @patch.object(OpenAlexSource, "_api_get")
    def test_search_returns_candidates(self, mock_get: MagicMock) -> None:
        mock_get.return_value = {
            "results": [
                _make_openalex_work("W1", "Paper One"),
                _make_openalex_work("W2", "Paper Two"),
            ],
        }
        source = OpenAlexSource()
        results = source.search(
            topic="transformers",
            must_terms=["attention"],
            nice_terms=[],
            max_results=10,
        )

        assert len(results) == 2

    @patch.object(OpenAlexSource, "_api_get")
    def test_search_handles_api_error(self, mock_get: MagicMock) -> None:
        mock_get.side_effect = requests.RequestException("fail")
        source = OpenAlexSource()
        results = source.search(topic="test", must_terms=["x"], nice_terms=[])
        assert results == []

    @patch.object(OpenAlexSource, "_api_get")
    def test_search_with_date_filter(self, mock_get: MagicMock) -> None:
        mock_get.return_value = {"results": []}
        source = OpenAlexSource()
        source.search(
            topic="test",
            must_terms=["term"],
            nice_terms=[],
            date_from="2023-01-01",
            date_to="2024-12-31",
        )

        call_params = mock_get.call_args[0][1]
        assert "from_publication_date:2023-01-01" in call_params["filter"]
        assert "to_publication_date:2024-12-31" in call_params["filter"]
