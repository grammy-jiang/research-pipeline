"""Unit tests for HuggingFaceSource."""

from datetime import UTC, datetime
from unittest.mock import MagicMock

from research_pipeline.sources.huggingface_source import (
    HuggingFaceSource,
    _matches_terms,
    _parse_date,
)


def _make_hf_entry(
    paper_id: str = "2401.12345",
    title: str = "Memory-Augmented Neural Networks",
    summary: str = "We present a novel memory architecture.",
    published_at: str = "2024-01-15T00:00:00Z",
    authors: list[dict[str, str]] | None = None,
) -> dict[str, object]:
    """Create a mock HuggingFace daily paper entry."""
    if authors is None:
        authors = [{"name": "Alice"}, {"name": "Bob"}]
    return {
        "paper": {
            "id": paper_id,
            "title": title,
            "summary": summary,
            "publishedAt": published_at,
            "authors": authors,
        },
        "numComments": 5,
    }


class TestHuggingFaceSource:
    """Tests for HuggingFaceSource."""

    def test_name(self) -> None:
        """Source name is 'huggingface'."""
        source = HuggingFaceSource()
        assert source.name == "huggingface"

    def test_search_parses_entries(self) -> None:
        """Valid entries are parsed into CandidateRecords."""
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.json.return_value = [
            _make_hf_entry(
                paper_id="2401.12345",
                title="Transformer Memory Systems",
                summary="A memory system using transformers.",
            ),
            _make_hf_entry(
                paper_id="2401.67890",
                title="Attention Is All You Need",
                summary="We propose a new network architecture.",
            ),
        ]
        mock_response.raise_for_status = MagicMock()
        mock_session.get.return_value = mock_response

        source = HuggingFaceSource(min_interval=0.0, session=mock_session)
        results = source.search(
            topic="memory systems",
            must_terms=["memory"],
            nice_terms=["transformer"],
            max_results=10,
        )

        assert len(results) == 1  # Only "memory" match
        assert results[0].arxiv_id == "2401.12345"
        assert results[0].source == "huggingface"
        assert results[0].title == "Transformer Memory Systems"
        assert results[0].authors == ["Alice", "Bob"]

    def test_search_returns_empty_on_api_error(self) -> None:
        """API failure returns empty list."""
        mock_session = MagicMock()
        import requests

        mock_session.get.side_effect = requests.RequestException("timeout")

        source = HuggingFaceSource(min_interval=0.0, session=mock_session)
        results = source.search(
            topic="anything",
            must_terms=["test"],
            nice_terms=[],
        )
        assert results == []

    def test_search_filters_by_date(self) -> None:
        """Entries outside date range are excluded."""
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.json.return_value = [
            _make_hf_entry(
                paper_id="2024.old",
                title="Old Memory Paper",
                summary="About memory.",
                published_at="2023-01-01T00:00:00Z",
            ),
            _make_hf_entry(
                paper_id="2024.new",
                title="New Memory Paper",
                summary="About memory.",
                published_at="2024-06-15T00:00:00Z",
            ),
        ]
        mock_response.raise_for_status = MagicMock()
        mock_session.get.return_value = mock_response

        source = HuggingFaceSource(min_interval=0.0, session=mock_session)
        results = source.search(
            topic="memory",
            must_terms=["memory"],
            nice_terms=[],
            date_from="2024-01-01",
            date_to="2024-12-31",
        )

        assert len(results) == 1
        assert results[0].arxiv_id == "hf-2024.new"

    def test_search_no_must_terms_returns_all(self) -> None:
        """When no must_terms, all entries are returned."""
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.json.return_value = [
            _make_hf_entry(title="Paper A"),
            _make_hf_entry(paper_id="2401.99999", title="Paper B"),
        ]
        mock_response.raise_for_status = MagicMock()
        mock_session.get.return_value = mock_response

        source = HuggingFaceSource(min_interval=0.0, session=mock_session)
        results = source.search(
            topic="anything",
            must_terms=[],
            nice_terms=[],
        )
        assert len(results) == 2

    def test_search_max_results_respected(self) -> None:
        """Results are capped at max_results."""
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.json.return_value = [
            _make_hf_entry(paper_id=f"2401.{i:05d}", title=f"Memory Paper {i}")
            for i in range(10)
        ]
        mock_response.raise_for_status = MagicMock()
        mock_session.get.return_value = mock_response

        source = HuggingFaceSource(min_interval=0.0, session=mock_session)
        results = source.search(
            topic="memory",
            must_terms=["memory"],
            nice_terms=[],
            max_results=3,
        )
        assert len(results) == 3

    def test_parse_entry_non_arxiv_id(self) -> None:
        """Non-arXiv paper IDs get hf- prefix."""
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.json.return_value = [
            _make_hf_entry(paper_id="some-custom-id", title="Custom Memory Paper"),
        ]
        mock_response.raise_for_status = MagicMock()
        mock_session.get.return_value = mock_response

        source = HuggingFaceSource(min_interval=0.0, session=mock_session)
        results = source.search(
            topic="memory",
            must_terms=["memory"],
            nice_terms=[],
        )
        assert len(results) == 1
        assert results[0].arxiv_id.startswith("hf-")
        assert results[0].pdf_url == ""


class TestMatchesTerms:
    """Tests for the _matches_terms helper."""

    def test_matches_in_title(self) -> None:
        """Term in title matches."""
        from research_pipeline.models.candidate import CandidateRecord

        c = CandidateRecord(
            arxiv_id="test",
            version="v1",
            title="Neural Memory Architecture",
            authors=[],
            published=datetime.now(UTC),
            updated=datetime.now(UTC),
            categories=[],
            primary_category="",
            abstract="No relevant keywords here.",
            abs_url="",
            pdf_url="",
            source="test",
        )
        assert _matches_terms(c, ["memory"]) is True
        assert _matches_terms(c, ["quantum"]) is False

    def test_matches_in_abstract(self) -> None:
        """Term in abstract matches."""
        from research_pipeline.models.candidate import CandidateRecord

        c = CandidateRecord(
            arxiv_id="test",
            version="v1",
            title="Boring Title",
            authors=[],
            published=datetime.now(UTC),
            updated=datetime.now(UTC),
            categories=[],
            primary_category="",
            abstract="This paper introduces a memory system.",
            abs_url="",
            pdf_url="",
            source="test",
        )
        assert _matches_terms(c, ["memory"]) is True

    def test_case_insensitive(self) -> None:
        """Matching is case-insensitive."""
        from research_pipeline.models.candidate import CandidateRecord

        c = CandidateRecord(
            arxiv_id="test",
            version="v1",
            title="MEMORY SYSTEMS",
            authors=[],
            published=datetime.now(UTC),
            updated=datetime.now(UTC),
            categories=[],
            primary_category="",
            abstract="",
            abs_url="",
            pdf_url="",
            source="test",
        )
        assert _matches_terms(c, ["memory"]) is True
        assert _matches_terms(c, ["Memory"]) is True


class TestParseDate:
    """Tests for the _parse_date helper."""

    def test_iso8601_with_z(self) -> None:
        """Parse ISO 8601 with Z suffix."""
        dt = _parse_date("2024-01-15T10:30:00Z")
        assert dt.year == 2024
        assert dt.month == 1
        assert dt.day == 15
        assert dt.tzinfo is not None

    def test_yyyy_mm_dd(self) -> None:
        """Parse simple date string."""
        dt = _parse_date("2024-06-20")
        assert dt.year == 2024
        assert dt.month == 6
        assert dt.day == 20
        assert dt.tzinfo is not None

    def test_invalid_returns_now(self) -> None:
        """Invalid date returns current time."""
        dt = _parse_date("not-a-date")
        assert isinstance(dt, datetime)
        assert dt.tzinfo is not None
