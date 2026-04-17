"""Tests for sources/enrichment.py — abstract enrichment pipeline."""

from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

from research_pipeline.models.candidate import CandidateRecord
from research_pipeline.sources.enrichment import (
    _s2_lookup_by_doi,
    _s2_lookup_by_title,
    enrich_candidates,
)


def _make_candidate(
    arxiv_id: str = "2401.00001",
    title: str = "Test Paper",
    abstract: str = "",
    doi: str | None = None,
    citation_count: int | None = None,
    semantic_scholar_id: str | None = None,
    venue: str | None = None,
) -> CandidateRecord:
    """Helper to create test candidates."""
    return CandidateRecord(
        arxiv_id=arxiv_id,
        version="v1",
        title=title,
        authors=["Author One"],
        published=datetime(2024, 1, 1, tzinfo=UTC),
        updated=datetime(2024, 1, 1, tzinfo=UTC),
        categories=["cs.AI"],
        primary_category="cs.AI",
        abstract=abstract,
        abs_url=f"https://arxiv.org/abs/{arxiv_id}",
        pdf_url=f"https://arxiv.org/pdf/{arxiv_id}",
        doi=doi,
        citation_count=citation_count,
        semantic_scholar_id=semantic_scholar_id,
        venue=venue,
    )


def _mock_s2_response(
    abstract: str = "Enriched abstract",
    citation_count: int = 42,
    influential_count: int = 5,
    paper_id: str = "s2-abc123",
    venue: str = "NeurIPS 2024",
    title: str = "Test Paper",
) -> dict:
    """Helper to create mock S2 API responses."""
    return {
        "paperId": paper_id,
        "title": title,
        "abstract": abstract,
        "citationCount": citation_count,
        "influentialCitationCount": influential_count,
        "venue": venue,
    }


# --- DOI lookup tests ---


class TestS2LookupByDoi:
    """Tests for _s2_lookup_by_doi."""

    def test_successful_lookup(self) -> None:
        """DOI lookup returns paper data."""
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = _mock_s2_response()
        mock_session.get.return_value = mock_response

        mock_limiter = MagicMock()

        result = _s2_lookup_by_doi("10.1234/test", mock_session, mock_limiter)

        assert result is not None
        assert result["abstract"] == "Enriched abstract"
        mock_limiter.wait.assert_called_once()

    def test_not_found_returns_none(self) -> None:
        """404 response returns None."""
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_session.get.return_value = mock_response

        result = _s2_lookup_by_doi("10.1234/missing", mock_session, MagicMock())

        assert result is None


# --- Title lookup tests ---


class TestS2LookupByTitle:
    """Tests for _s2_lookup_by_title."""

    def test_exact_title_match(self) -> None:
        """Title search returns paper when title matches exactly."""
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [_mock_s2_response(title="Test Paper")]
        }
        mock_session.get.return_value = mock_response

        result = _s2_lookup_by_title("Test Paper", mock_session, MagicMock())

        assert result is not None
        assert result["title"] == "Test Paper"

    def test_no_results_returns_none(self) -> None:
        """Empty search results return None."""
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": []}
        mock_session.get.return_value = mock_response

        result = _s2_lookup_by_title("Unknown Paper", mock_session, MagicMock())

        assert result is None

    def test_mismatched_title_returns_none(self) -> None:
        """Title that doesn't match returns None."""
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [_mock_s2_response(title="Completely Different Paper About Cats")]
        }
        mock_session.get.return_value = mock_response

        result = _s2_lookup_by_title(
            "Quantum Computing Survey", mock_session, MagicMock()
        )

        assert result is None

    def test_404_returns_none(self) -> None:
        """404 response returns None."""
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_session.get.return_value = mock_response

        result = _s2_lookup_by_title("Any Title", mock_session, MagicMock())

        assert result is None


# --- enrich_candidates tests ---


class TestEnrichCandidates:
    """Tests for the main enrich_candidates function."""

    @patch("research_pipeline.sources.enrichment._s2_lookup_by_doi")
    def test_enriches_missing_abstract_via_doi(self, mock_doi: MagicMock) -> None:
        """Candidate missing abstract gets enriched via DOI lookup."""
        mock_doi.return_value = _mock_s2_response(abstract="New abstract")

        candidates = [_make_candidate(abstract="", doi="10.1234/test")]

        count = enrich_candidates(candidates)

        assert count == 1
        assert candidates[0].abstract == "New abstract"

    @patch("research_pipeline.sources.enrichment._s2_lookup_by_doi")
    def test_enriches_missing_citations(self, mock_doi: MagicMock) -> None:
        """Candidate missing citation count gets enriched."""
        mock_doi.return_value = _mock_s2_response(citation_count=100)

        candidates = [
            _make_candidate(abstract="Has abstract", doi="10.1/x", citation_count=None)
        ]

        count = enrich_candidates(candidates)

        assert count == 1
        assert candidates[0].citation_count == 100

    @patch("research_pipeline.sources.enrichment._s2_lookup_by_doi")
    def test_skips_already_complete(self, mock_doi: MagicMock) -> None:
        """Already-complete candidates are not looked up."""
        candidates = [
            _make_candidate(
                abstract="Full abstract",
                doi="10.1/x",
                citation_count=50,
            )
        ]

        count = enrich_candidates(candidates)

        assert count == 0
        mock_doi.assert_not_called()

    @patch("research_pipeline.sources.enrichment._s2_lookup_by_title")
    @patch("research_pipeline.sources.enrichment._s2_lookup_by_doi")
    def test_falls_back_to_title_when_no_doi(
        self, mock_doi: MagicMock, mock_title: MagicMock
    ) -> None:
        """Candidate without DOI falls back to title-based lookup."""
        mock_title.return_value = _mock_s2_response(abstract="From title search")

        candidates = [_make_candidate(abstract="", doi=None)]

        count = enrich_candidates(candidates)

        assert count == 1
        assert candidates[0].abstract == "From title search"
        mock_doi.assert_not_called()  # No DOI, so DOI lookup skipped
        mock_title.assert_called_once()

    @patch("research_pipeline.sources.enrichment._s2_lookup_by_title")
    @patch("research_pipeline.sources.enrichment._s2_lookup_by_doi")
    def test_falls_back_to_title_when_doi_returns_none(
        self, mock_doi: MagicMock, mock_title: MagicMock
    ) -> None:
        """DOI lookup returns None → falls back to title search."""
        mock_doi.return_value = None
        mock_title.return_value = _mock_s2_response(abstract="Fallback abstract")

        candidates = [_make_candidate(abstract="", doi="10.1/missing")]

        count = enrich_candidates(candidates)

        assert count == 1
        assert candidates[0].abstract == "Fallback abstract"

    @patch("research_pipeline.sources.enrichment._s2_lookup_by_doi")
    def test_enriches_venue_and_s2_id(self, mock_doi: MagicMock) -> None:
        """Enrichment fills in venue and Semantic Scholar ID."""
        mock_doi.return_value = _mock_s2_response(venue="ICML 2024", paper_id="s2-xyz")

        candidates = [_make_candidate(abstract="", doi="10.1/x")]

        enrich_candidates(candidates)

        assert candidates[0].venue == "ICML 2024"
        assert candidates[0].semantic_scholar_id == "s2-xyz"

    @patch("research_pipeline.sources.enrichment._s2_lookup_by_doi")
    def test_preserves_existing_venue(self, mock_doi: MagicMock) -> None:
        """Existing venue is not overwritten."""
        mock_doi.return_value = _mock_s2_response(venue="New Venue")

        candidates = [
            _make_candidate(abstract="", doi="10.1/x", venue="Original Venue")
        ]

        enrich_candidates(candidates)

        assert candidates[0].venue == "Original Venue"

    @patch("research_pipeline.sources.enrichment._s2_lookup_by_doi")
    def test_handles_api_error_gracefully(self, mock_doi: MagicMock) -> None:
        """API errors don't crash enrichment."""
        import requests

        mock_doi.side_effect = requests.RequestException("timeout")

        candidates = [_make_candidate(abstract="", doi="10.1/x")]

        count = enrich_candidates(candidates)

        assert count == 0

    @patch("research_pipeline.sources.enrichment._s2_lookup_by_doi")
    def test_multiple_candidates_partial_enrichment(self, mock_doi: MagicMock) -> None:
        """Mix of enrichable and non-enrichable candidates."""
        mock_doi.side_effect = [
            _mock_s2_response(abstract="Enriched 1"),
            None,  # Second candidate not found
        ]

        candidates = [
            _make_candidate(arxiv_id="2401.00001", abstract="", doi="10.1/a"),
            _make_candidate(arxiv_id="2401.00002", abstract="", doi="10.1/b"),
            _make_candidate(
                arxiv_id="2401.00003", abstract="Already here", citation_count=10
            ),
        ]

        count = enrich_candidates(candidates)

        assert count == 1  # Only first one enriched
        assert candidates[0].abstract == "Enriched 1"
        assert candidates[1].abstract == ""  # Not found

    @patch("research_pipeline.sources.enrichment._s2_lookup_by_doi")
    def test_enriches_influential_citation_count(self, mock_doi: MagicMock) -> None:
        """Influential citation count is filled in."""
        mock_doi.return_value = _mock_s2_response(influential_count=12)

        candidates = [_make_candidate(abstract="", doi="10.1/x")]

        enrich_candidates(candidates)

        assert candidates[0].influential_citation_count == 12

    def test_empty_candidates_list(self) -> None:
        """Empty list returns 0 enriched."""
        count = enrich_candidates([])
        assert count == 0

    @patch("research_pipeline.sources.enrichment._s2_lookup_by_doi")
    def test_s2_api_key_set_in_session(self, mock_doi: MagicMock) -> None:
        """When API key provided, it's set on session headers."""
        mock_doi.return_value = None

        candidates = [_make_candidate(abstract="", doi="10.1/x")]

        # Capture the session used internally
        with patch("requests.Session") as mock_session_cls:
            mock_session = MagicMock()
            mock_session_cls.return_value = mock_session

            enrich_candidates(candidates, s2_api_key="test-key-123")

            mock_session.headers.__setitem__.assert_called_with(
                "x-api-key", "test-key-123"
            )
