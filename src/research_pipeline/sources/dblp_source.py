"""DBLP search source.

Uses the DBLP API ``/search/publ/api`` endpoint.  No API key required.
Rate limit: 1 request per 2 seconds (polite, fair-use policy).

Note: DBLP does not provide abstracts.  Papers returned from DBLP have
``abstract=""``.  Use cross-source enrichment (Phase 2.5) to fill in
abstracts from Semantic Scholar or OpenAlex.
"""

import logging
import re
from datetime import UTC, datetime

import requests

from research_pipeline.infra.rate_limit import RateLimiter
from research_pipeline.infra.retry import retry
from research_pipeline.models.candidate import CandidateRecord

logger = logging.getLogger(__name__)

_DBLP_API_BASE = "https://dblp.org/search/publ/api"
_ARXIV_ID_RE = re.compile(r"(\d{4}\.\d{4,5})")


class DBLPSource:
    """Search source backed by the DBLP API.

    Implements the ``SearchSource`` protocol.
    """

    def __init__(
        self,
        min_interval: float = 2.0,
        session: requests.Session | None = None,
    ) -> None:
        self._rate_limiter = RateLimiter(min_interval=min_interval, name="dblp")
        self._session = session or requests.Session()

    @property
    def name(self) -> str:
        return "dblp"

    @retry(
        max_attempts=3,
        backoff_base=2.0,
        retryable_exceptions=(requests.RequestException,),
    )
    def _api_get(self, params: dict[str, str | int]) -> dict:  # type: ignore[type-arg]
        """Execute a rate-limited, retried GET request to the DBLP API.

        Args:
            params: Query parameters.

        Returns:
            Parsed JSON response.
        """
        self._rate_limiter.wait()
        response = self._session.get(_DBLP_API_BASE, params=params, timeout=30)
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
        """Search DBLP for publications.

        Args:
            topic: Raw topic string.
            must_terms: Terms that must appear.
            nice_terms: Boost terms.
            max_results: Maximum number of results (API cap: 1000).
            date_from: Not directly supported by DBLP (ignored).
            date_to: Not directly supported by DBLP (ignored).

        Returns:
            List of CandidateRecords with ``source="dblp"``.
        """
        query_parts = must_terms[:3]
        if nice_terms:
            query_parts.extend(nice_terms[:2])
        query = " ".join(query_parts) if query_parts else topic

        logger.info("DBLP search: %s (max_results=%d)", query, max_results)

        params: dict[str, str | int] = {
            "q": query,
            "format": "json",
            "h": min(max_results, 1000),
        }

        try:
            data = self._api_get(params)
        except requests.RequestException as exc:
            logger.error("DBLP search failed: %s", exc)
            return []

        result = data.get("result", {})
        hits = result.get("hits", {})
        hit_list = hits.get("hit", [])

        candidates: list[CandidateRecord] = []
        for hit in hit_list[:max_results]:
            try:
                info = hit.get("info", {})
                candidate = self._parse_hit(info)
                candidates.append(candidate)
            except Exception as exc:
                logger.warning("Failed to parse DBLP hit: %s", exc)

        logger.info("DBLP returned %d candidates", len(candidates))
        return candidates

    def _parse_hit(self, info: dict) -> CandidateRecord:  # type: ignore[type-arg]
        """Parse a DBLP hit info dict into a CandidateRecord.

        Args:
            info: The ``info`` dict from a DBLP search hit.

        Returns:
            A CandidateRecord populated with DBLP data.
        """
        title = info.get("title") or ""
        # DBLP sometimes appends a trailing period
        if title.endswith("."):
            title = title[:-1]

        # Authors
        authors_data = info.get("authors", {})
        author_list = authors_data.get("author", [])
        if isinstance(author_list, dict):
            author_list = [author_list]
        authors = []
        for a in author_list:
            if isinstance(a, dict):
                authors.append(a.get("text", ""))
            elif isinstance(a, str):
                authors.append(a)

        # Year
        year_str = info.get("year")
        year = int(year_str) if year_str else None
        published = datetime(year, 1, 1, tzinfo=UTC) if year else datetime.now(UTC)

        # DOI and URLs
        doi = info.get("doi") or None
        url = info.get("url") or ""
        ee = info.get("ee") or ""

        # Try to extract arXiv ID
        arxiv_id = ""
        for candidate_url in [ee, url]:
            match = _ARXIV_ID_RE.search(candidate_url)
            if match and "arxiv" in candidate_url.lower():
                arxiv_id = match.group(1)
                break

        if not arxiv_id:
            key = info.get("key", "")
            arxiv_id = f"dblp-{abs(hash(key or title.lower())) % 10**10}"

        # Venue
        venue = info.get("venue") or ""

        abs_url = ee or url
        pdf_url = ""
        if arxiv_id and not arxiv_id.startswith("dblp-"):
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"
            abs_url = abs_url or f"https://arxiv.org/abs/{arxiv_id}"

        return CandidateRecord(
            arxiv_id=arxiv_id,
            version="v1",
            title=title,
            authors=authors,
            published=published,
            updated=published,
            categories=[],
            primary_category="",
            abstract="",  # DBLP has no abstracts
            abs_url=abs_url,
            pdf_url=pdf_url,
            source="dblp",
            doi=doi,
            venue=venue if venue else None,
            year=year,
        )
