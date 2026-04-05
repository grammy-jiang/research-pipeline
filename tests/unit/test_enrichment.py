"""Unit tests for sources.enrichment."""

from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import requests

from research_pipeline.infra.rate_limit import RateLimiter
from research_pipeline.models.candidate import CandidateRecord
from research_pipeline.sources.enrichment import enrich_candidates


def _make_candidate(
    arxiv_id: str = "2401.12345",
    abstract: str = "",
    doi: str | None = "10.1234/test",
    citation_count: int | None = None,
) -> CandidateRecord:
    """Build a minimal CandidateRecord for testing."""
    return CandidateRecord(
        arxiv_id=arxiv_id,
        version="v1",
        title="Test Paper",
        authors=["Author One"],
        published=datetime(2024, 1, 1, tzinfo=UTC),
        updated=datetime(2024, 1, 1, tzinfo=UTC),
        categories=["cs.IR"],
        primary_category="cs.IR",
        abstract=abstract,
        abs_url=f"https://arxiv.org/abs/{arxiv_id}",
        pdf_url=f"https://arxiv.org/pdf/{arxiv_id}",
        source="dblp",
        doi=doi,
        citation_count=citation_count,
    )


class TestEnrichCandidates:
    """Tests for enrich_candidates function."""

    @patch("research_pipeline.sources.enrichment._s2_lookup_by_doi")
    def test_enriches_missing_abstract(self, mock_lookup: MagicMock) -> None:
        mock_lookup.return_value = {
            "paperId": "s2-abc",
            "abstract": "Enriched abstract text",
            "citationCount": 50,
            "influentialCitationCount": 5,
            "venue": "ICML",
        }
        candidates = [_make_candidate(abstract="", citation_count=None)]
        count = enrich_candidates(
            candidates,
            s2_rate_limiter=RateLimiter(min_interval=0.0),
        )

        assert count == 1
        assert candidates[0].abstract == "Enriched abstract text"
        assert candidates[0].citation_count == 50
        assert candidates[0].semantic_scholar_id == "s2-abc"

    @patch("research_pipeline.sources.enrichment._s2_lookup_by_doi")
    def test_skips_already_enriched(self, mock_lookup: MagicMock) -> None:
        candidates = [
            _make_candidate(abstract="Already has abstract", citation_count=10)
        ]
        count = enrich_candidates(
            candidates,
            s2_rate_limiter=RateLimiter(min_interval=0.0),
        )

        assert count == 0
        mock_lookup.assert_not_called()

    @patch("research_pipeline.sources.enrichment._s2_lookup_by_doi")
    def test_skips_no_doi(self, mock_lookup: MagicMock) -> None:
        candidates = [_make_candidate(doi=None)]
        count = enrich_candidates(
            candidates,
            s2_rate_limiter=RateLimiter(min_interval=0.0),
        )

        assert count == 0
        mock_lookup.assert_not_called()

    @patch("research_pipeline.sources.enrichment._s2_lookup_by_doi")
    def test_handles_lookup_failure(self, mock_lookup: MagicMock) -> None:
        mock_lookup.side_effect = requests.RequestException("fail")
        candidates = [_make_candidate()]
        count = enrich_candidates(
            candidates,
            s2_rate_limiter=RateLimiter(min_interval=0.0),
        )

        assert count == 0

    @patch("research_pipeline.sources.enrichment._s2_lookup_by_doi")
    def test_handles_not_found(self, mock_lookup: MagicMock) -> None:
        mock_lookup.return_value = None
        candidates = [_make_candidate()]
        count = enrich_candidates(
            candidates,
            s2_rate_limiter=RateLimiter(min_interval=0.0),
        )

        assert count == 0
