"""OpenAlex search source.

Uses the OpenAlex API ``/works`` endpoint.  Requires an API key (free,
mandatory since Feb 2026).  Rate limit: 5 req/sec (well within the
100K credits/day free tier).
"""

import logging
import re
from datetime import UTC, datetime

import requests

from research_pipeline.infra.rate_limit import RateLimiter
from research_pipeline.infra.retry import retry
from research_pipeline.models.candidate import CandidateRecord

logger = logging.getLogger(__name__)

_OPENALEX_API_BASE = "https://api.openalex.org"
_ARXIV_ID_RE = re.compile(r"(\d{4}\.\d{4,5})")


class OpenAlexSource:
    """Search source backed by the OpenAlex API.

    Implements the ``SearchSource`` protocol.
    """

    def __init__(
        self,
        api_key: str = "",
        min_interval: float = 0.2,
        session: requests.Session | None = None,
    ) -> None:
        self._api_key = api_key
        self._rate_limiter = RateLimiter(min_interval=min_interval, name="openalex")
        self._session = session or requests.Session()

    @property
    def name(self) -> str:
        return "openalex"

    @retry(
        max_attempts=3,
        backoff_base=2.0,
        retryable_exceptions=(requests.RequestException,),
    )
    def _api_get(self, url: str, params: dict[str, str | int]) -> dict:  # type: ignore[type-arg]
        """Execute a rate-limited, retried GET request.

        Args:
            url: API endpoint URL.
            params: Query parameters.

        Returns:
            Parsed JSON response.
        """
        self._rate_limiter.wait()
        if self._api_key:
            params["api_key"] = self._api_key
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
        """Search OpenAlex for works.

        Args:
            topic: Raw topic string.
            must_terms: Terms that must appear.
            nice_terms: Boost terms.
            max_results: Maximum number of results.
            date_from: Date filter start (YYYY-MM-DD or YYYY).
            date_to: Date filter end (YYYY-MM-DD or YYYY).

        Returns:
            List of CandidateRecords with ``source="openalex"``.
        """
        query_parts = must_terms[:3]
        if nice_terms:
            query_parts.extend(nice_terms[:2])
        query = " ".join(query_parts) if query_parts else topic

        logger.info("OpenAlex search: %s (max_results=%d)", query, max_results)

        params: dict[str, str | int] = {
            "search": query,
            "per_page": min(max_results, 200),
        }

        # Date filter
        filters: list[str] = []
        if date_from:
            filters.append(f"from_publication_date:{date_from[:10]}")
        if date_to:
            filters.append(f"to_publication_date:{date_to[:10]}")
        if filters:
            params["filter"] = ",".join(filters)

        url = f"{_OPENALEX_API_BASE}/works"
        candidates: list[CandidateRecord] = []

        try:
            data = self._api_get(url, params)
        except requests.RequestException as exc:
            logger.error("OpenAlex search failed: %s", exc)
            return []

        results = data.get("results", [])
        for work in results[:max_results]:
            try:
                candidate = self._parse_work(work)
                candidates.append(candidate)
            except Exception as exc:
                logger.warning(
                    "Failed to parse OpenAlex work %s: %s",
                    work.get("id", "?"),
                    exc,
                )

        logger.info("OpenAlex returned %d candidates", len(candidates))
        return candidates

    def _parse_work(self, work: dict) -> CandidateRecord:  # type: ignore[type-arg]
        """Parse an OpenAlex work dict into a CandidateRecord.

        Args:
            work: Raw work dict from OpenAlex API.

        Returns:
            A CandidateRecord populated with OpenAlex data.
        """
        title = work.get("title") or ""
        doi = work.get("doi") or ""
        if doi.startswith("https://doi.org/"):
            doi = doi[len("https://doi.org/") :]

        # Extract OpenAlex ID
        openalex_id = work.get("id") or ""
        if openalex_id.startswith("https://openalex.org/"):
            openalex_id = openalex_id[len("https://openalex.org/") :]

        # Try to extract arXiv ID from locations
        arxiv_id = ""
        locations = work.get("locations") or []
        for loc in locations:
            source = loc.get("source") or {}
            landing_url = loc.get("landing_page_url") or ""
            pdf_url_loc = loc.get("pdf_url") or ""
            for url in [landing_url, pdf_url_loc]:
                match = _ARXIV_ID_RE.search(url)
                if match and (
                    "arxiv" in url.lower()
                    or "arxiv" in source.get("display_name", "").lower()
                ):
                    arxiv_id = match.group(1)
                    break
            if arxiv_id:
                break

        if not arxiv_id:
            arxiv_id = (
                f"oalex-{openalex_id}"
                if openalex_id
                else f"oalex-{abs(hash(title.lower())) % 10**10}"
            )

        # Abstract from inverted index
        abstract = ""
        abstract_inverted = work.get("abstract_inverted_index")
        if abstract_inverted and isinstance(abstract_inverted, dict):
            abstract = _reconstruct_abstract(abstract_inverted)

        # Authors
        authorships = work.get("authorships") or []
        authors = []
        for authorship in authorships:
            author = authorship.get("author", {})
            name = author.get("display_name", "")
            if name:
                authors.append(name)

        # Publication date
        pub_date_str = work.get("publication_date")
        year = work.get("publication_year")
        if pub_date_str:
            try:
                published = datetime.fromisoformat(pub_date_str).replace(tzinfo=UTC)
            except ValueError:
                published = datetime(year or 2024, 1, 1, tzinfo=UTC)
        elif year:
            published = datetime(year, 1, 1, tzinfo=UTC)
        else:
            published = datetime.now(UTC)

        # Venue
        primary_location = work.get("primary_location") or {}
        venue_source = primary_location.get("source") or {}
        venue = venue_source.get("display_name") or ""

        # PDF URL
        pdf_url = ""
        best_oa = work.get("best_oa_location") or {}
        pdf_url = best_oa.get("pdf_url") or ""
        if not pdf_url and arxiv_id and not arxiv_id.startswith("oalex-"):
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"

        abs_url = ""
        if arxiv_id and not arxiv_id.startswith("oalex-"):
            abs_url = f"https://arxiv.org/abs/{arxiv_id}"
        elif doi:
            abs_url = f"https://doi.org/{doi}"

        citation_count = work.get("cited_by_count")

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
            source="openalex",
            doi=doi if doi else None,
            openalex_id=openalex_id if openalex_id else None,
            citation_count=citation_count,
            venue=venue if venue else None,
            year=year,
        )


def _reconstruct_abstract(inverted_index: dict[str, list[int]]) -> str:
    """Reconstruct abstract text from OpenAlex inverted index format.

    Args:
        inverted_index: Dict mapping words to their position indices.

    Returns:
        Reconstructed abstract string.
    """
    if not inverted_index:
        return ""
    # Build position → word mapping
    positions: list[tuple[int, str]] = []
    for word, indices in inverted_index.items():
        for idx in indices:
            positions.append((idx, word))
    positions.sort(key=lambda x: x[0])
    return " ".join(word for _, word in positions)
