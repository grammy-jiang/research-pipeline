"""Semantic Scholar search source.

Uses the Semantic Scholar Academic Graph API ``/paper/search`` endpoint.
Free tier supports 1 request/sec without an API key; with a free key,
rate limits are more generous (~100 req/5min).
"""

import logging
from datetime import UTC, datetime

import requests

from research_pipeline.infra.rate_limit import RateLimiter
from research_pipeline.infra.retry import retry
from research_pipeline.models.candidate import CandidateRecord

logger = logging.getLogger(__name__)

_S2_API_BASE = "https://api.semanticscholar.org/graph/v1"

_S2_SEARCH_FIELDS = ",".join(
    [
        "paperId",
        "externalIds",
        "title",
        "abstract",
        "year",
        "authors",
        "citationCount",
        "influentialCitationCount",
        "venue",
        "publicationDate",
        "openAccessPdf",
    ]
)


class SemanticScholarSource:
    """Search source backed by the Semantic Scholar API.

    Implements the ``SearchSource`` protocol.
    """

    def __init__(
        self,
        api_key: str = "",
        min_interval: float = 1.0,
        session: requests.Session | None = None,
    ) -> None:
        self._api_key = api_key
        self._rate_limiter = RateLimiter(
            min_interval=min_interval, name="semantic_scholar"
        )
        self._session = session or requests.Session()
        if api_key:
            self._session.headers["x-api-key"] = api_key

    @property
    def name(self) -> str:
        return "semantic_scholar"

    @retry(
        max_attempts=3,
        backoff_base=2.0,
        retryable_exceptions=(requests.RequestException,),
    )
    def _api_get(self, url: str, params: dict[str, str | int]) -> dict:  # type: ignore[type-arg]
        """Execute a rate-limited, retried GET request to the S2 API.

        Args:
            url: API endpoint URL.
            params: Query parameters.

        Returns:
            Parsed JSON response.
        """
        self._rate_limiter.wait()
        response = self._session.get(url, params=params, timeout=30)
        response.raise_for_status()
        return response.json()  # type: ignore[no-any-return]

    def search(
        self,
        topic: str,
        must_terms: list[str],
        nice_terms: list[str],
        max_results: int = 100,
        date_from: str | None = None,
        date_to: str | None = None,
    ) -> list[CandidateRecord]:
        """Search Semantic Scholar for papers.

        Args:
            topic: Raw topic string.
            must_terms: Terms that must appear.
            nice_terms: Boost terms.
            max_results: Maximum number of results (API cap: 100 per page).
            date_from: Date filter start (YYYY-MM-DD or YYYY).
            date_to: Date filter end (YYYY-MM-DD or YYYY).

        Returns:
            List of CandidateRecords with ``source="semantic_scholar"``.
        """
        query_parts = must_terms[:3]
        if nice_terms:
            query_parts.extend(nice_terms[:2])
        query = " ".join(query_parts) if query_parts else topic

        logger.info("Semantic Scholar search: %s (max_results=%d)", query, max_results)

        params: dict[str, str | int] = {
            "query": query,
            "limit": min(max_results, 100),
            "fields": _S2_SEARCH_FIELDS,
        }

        # Add year range if provided
        if date_from or date_to:
            year_from = date_from[:4] if date_from else ""
            year_to = date_to[:4] if date_to else ""
            if year_from and year_to:
                params["year"] = f"{year_from}-{year_to}"
            elif year_from:
                params["year"] = f"{year_from}-"
            elif year_to:
                params["year"] = f"-{year_to}"

        url = f"{_S2_API_BASE}/paper/search"

        candidates: list[CandidateRecord] = []
        offset = 0

        while offset < max_results:
            params["offset"] = offset
            try:
                data = self._api_get(url, params)
            except requests.RequestException as exc:
                logger.error("Semantic Scholar search failed: %s", exc)
                break

            papers = data.get("data", [])
            if not papers:
                break

            for paper in papers:
                try:
                    candidate = self._parse_paper(paper)
                    candidates.append(candidate)
                except Exception as exc:
                    logger.warning(
                        "Failed to parse S2 paper %s: %s",
                        paper.get("paperId", "?"),
                        exc,
                    )

            total = data.get("total", 0)
            offset += len(papers)
            if offset >= total or offset >= max_results:
                break

        logger.info("Semantic Scholar returned %d candidates", len(candidates))
        return candidates

    def _parse_paper(self, paper: dict) -> CandidateRecord:  # type: ignore[type-arg]
        """Parse a Semantic Scholar paper dict into a CandidateRecord.

        Args:
            paper: Raw paper dict from S2 API.

        Returns:
            A CandidateRecord populated with S2 data.
        """
        external_ids = paper.get("externalIds") or {}
        arxiv_id = external_ids.get("ArXiv", "")
        doi = external_ids.get("DOI")

        if not arxiv_id:
            arxiv_id = f"s2-{paper.get('paperId', '')[:10]}"

        title = paper.get("title", "")
        abstract = paper.get("abstract") or ""
        year = paper.get("year")

        # Parse authors
        authors_raw = paper.get("authors") or []
        authors = [a.get("name", "") for a in authors_raw if a.get("name")]

        # Parse publication date
        pub_date_str = paper.get("publicationDate")
        if pub_date_str:
            try:
                published = datetime.fromisoformat(pub_date_str).replace(tzinfo=UTC)
            except ValueError:
                published = datetime(year or 2024, 1, 1, tzinfo=UTC)
        elif year:
            published = datetime(year, 1, 1, tzinfo=UTC)
        else:
            published = datetime.now(UTC)

        # Build PDF URL
        open_access = paper.get("openAccessPdf") or {}
        pdf_url = open_access.get("url", "")
        if not pdf_url and arxiv_id and not arxiv_id.startswith("s2-"):
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"

        abs_url = ""
        if arxiv_id and not arxiv_id.startswith("s2-"):
            abs_url = f"https://arxiv.org/abs/{arxiv_id}"
        elif doi:
            abs_url = f"https://doi.org/{doi}"

        venue = paper.get("venue") or ""

        return CandidateRecord(
            arxiv_id=arxiv_id,
            version="v1",
            title=title,
            authors=authors,
            published=published,
            updated=published,
            categories=[],
            primary_category="",
            abstract=abstract,
            abs_url=abs_url,
            pdf_url=pdf_url,
            source="semantic_scholar",
            doi=doi,
            semantic_scholar_id=paper.get("paperId"),
            citation_count=paper.get("citationCount"),
            influential_citation_count=paper.get("influentialCitationCount"),
            venue=venue if venue else None,
            year=year,
        )
