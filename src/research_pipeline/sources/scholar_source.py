"""Google Scholar search source using scholarly (free) or SerpAPI (paid)."""

import contextlib
import logging
import re
import time
from datetime import UTC, datetime

from research_pipeline.models.candidate import CandidateRecord

logger = logging.getLogger(__name__)

_ARXIV_ID_PATTERN = re.compile(r"(\d{4}\.\d{4,5})")


def _extract_arxiv_id(url: str) -> tuple[str, str]:
    """Try to extract an arXiv ID and version from a URL.

    Args:
        url: Any URL that might contain an arXiv ID.

    Returns:
        Tuple of (arxiv_id, version). Empty strings if not found.
    """
    match = _ARXIV_ID_PATTERN.search(url)
    if match:
        return match.group(1), "v1"
    return "", ""


class ScholarlySource:
    """Google Scholar search via the scholarly library (free, scraping-based).

    Rate-limited to avoid captchas. Results may be less structured
    than arXiv API but cover conferences, journals, and preprints.
    """

    def __init__(
        self,
        min_interval: float = 10.0,
    ) -> None:
        self._min_interval = min_interval
        self._last_request: float = 0.0

    @property
    def name(self) -> str:
        return "scholar"

    def _rate_wait(self) -> None:
        """Wait to respect rate limits."""
        elapsed = time.monotonic() - self._last_request
        if elapsed < self._min_interval:
            wait = self._min_interval - elapsed
            logger.debug("Scholar rate limiter: waiting %.1fs", wait)
            time.sleep(wait)
        self._last_request = time.monotonic()

    def search(
        self,
        topic: str,
        must_terms: list[str],
        nice_terms: list[str],
        max_results: int = 20,
        date_from: str | None = None,
        date_to: str | None = None,
    ) -> list[CandidateRecord]:
        """Search Google Scholar using the scholarly library.

        Args:
            topic: Raw topic for query construction.
            must_terms: Key terms (used in query).
            nice_terms: Additional terms.
            max_results: Max papers to retrieve.
            date_from: Not directly supported by scholarly (ignored).
            date_to: Not directly supported by scholarly (ignored).

        Returns:
            List of CandidateRecords from Scholar.
        """
        try:
            from scholarly import scholarly  # type: ignore[import-not-found]
        except ImportError:
            logger.error(
                "scholarly is not installed. "
                "Install with: pip install 'research-pipeline[scholar]'"
            )
            return []

        # Build query from terms
        query_parts = must_terms[:3]
        if nice_terms:
            query_parts.extend(nice_terms[:2])
        query = " ".join(query_parts)
        logger.info("Scholar query: %s (max_results=%d)", query, max_results)

        candidates: list[CandidateRecord] = []
        try:
            self._rate_wait()
            search_query = scholarly.search_pubs(query)

            for i, result in enumerate(search_query):
                if i >= max_results:
                    break
                try:
                    candidate = self._parse_result(result)
                    candidates.append(candidate)
                except Exception as exc:
                    logger.warning("Failed to parse Scholar result %d: %s", i, exc)

                if i > 0 and i % 5 == 0:
                    self._rate_wait()

        except Exception as exc:
            logger.error("Scholar search failed: %s", exc)

        logger.info("Scholar returned %d candidates", len(candidates))
        return candidates

    def _parse_result(self, result: dict) -> CandidateRecord:  # type: ignore[type-arg]
        """Parse a scholarly result dict into a CandidateRecord."""
        bib = result.get("bib", {})

        title = bib.get("title", "")
        abstract = bib.get("abstract", "")
        authors = bib.get("author", [])
        if isinstance(authors, str):
            authors = [a.strip() for a in authors.split(" and ")]

        pub_year = bib.get("pub_year", "")
        if pub_year:
            try:
                published = datetime(int(pub_year), 1, 1, tzinfo=UTC)
            except ValueError:
                published = datetime.now(UTC)
        else:
            published = datetime.now(UTC)

        # Try to extract arXiv ID from URLs
        pub_url = result.get("pub_url", "")
        eprint_url = result.get("eprint_url", "")
        arxiv_id, version = _extract_arxiv_id(pub_url)
        if not arxiv_id:
            arxiv_id, version = _extract_arxiv_id(eprint_url)
        if not arxiv_id:
            # Use a hash of the title as fallback ID
            arxiv_id = f"scholar-{abs(hash(title.lower())) % 10**10}"
            version = ""

        # Categories from venue
        venue = bib.get("venue", "")
        categories = [venue] if venue else []

        return CandidateRecord(
            arxiv_id=arxiv_id,
            version=version or "v1",
            title=title,
            authors=authors,
            published=published,
            updated=published,
            categories=categories,
            primary_category=categories[0] if categories else "",
            abstract=abstract,
            abs_url=pub_url or eprint_url,
            pdf_url=eprint_url or pub_url,
        )


class SerpAPISource:
    """Google Scholar search via SerpAPI (paid, reliable).

    Provides structured JSON responses. Requires a SerpAPI key.
    """

    def __init__(
        self,
        api_key: str = "",
        min_interval: float = 5.0,
    ) -> None:
        self._api_key = api_key
        self._min_interval = min_interval
        self._last_request: float = 0.0

    @property
    def name(self) -> str:
        return "serpapi"

    def _rate_wait(self) -> None:
        """Wait to respect rate limits."""
        elapsed = time.monotonic() - self._last_request
        if elapsed < self._min_interval:
            wait = self._min_interval - elapsed
            time.sleep(wait)
        self._last_request = time.monotonic()

    def search(
        self,
        topic: str,
        must_terms: list[str],
        nice_terms: list[str],
        max_results: int = 20,
        date_from: str | None = None,
        date_to: str | None = None,
    ) -> list[CandidateRecord]:
        """Search Google Scholar via SerpAPI.

        Args:
            topic: Raw topic for query.
            must_terms: Key terms.
            nice_terms: Additional terms.
            max_results: Max papers.
            date_from: Year filter start (e.g. "2024").
            date_to: Year filter end.

        Returns:
            List of CandidateRecords.
        """
        if not self._api_key:
            logger.error(
                "SerpAPI key not set. Set RESEARCH_PIPELINE_SERPAPI_KEY "
                "or configure sources.serpapi.api_key in config.toml"
            )
            return []

        try:
            from serpapi import GoogleSearch  # type: ignore[import-not-found]
        except ImportError:
            logger.error(
                "serpapi is not installed. "
                "Install with: pip install 'research-pipeline[serpapi]'"
            )
            return []

        query_parts = must_terms[:3]
        if nice_terms:
            query_parts.extend(nice_terms[:2])
        query = " ".join(query_parts)

        params = {
            "engine": "google_scholar",
            "q": query,
            "api_key": self._api_key,
            "num": min(max_results, 20),
        }

        # Add year filter if available
        if date_from:
            with contextlib.suppress(ValueError, IndexError):
                params["as_ylo"] = str(int(date_from[:4]))
        if date_to:
            with contextlib.suppress(ValueError, IndexError):
                params["as_yhi"] = str(int(date_to[:4]))

        logger.info("SerpAPI Scholar query: %s", query)
        candidates: list[CandidateRecord] = []

        try:
            self._rate_wait()
            search = GoogleSearch(params)
            results = search.get_dict()

            for result in results.get("organic_results", []):
                try:
                    candidate = self._parse_result(result)
                    candidates.append(candidate)
                except Exception as exc:
                    logger.warning("Failed to parse SerpAPI result: %s", exc)

        except Exception as exc:
            logger.error("SerpAPI search failed: %s", exc)

        logger.info("SerpAPI returned %d candidates", len(candidates))
        return candidates

    def _parse_result(self, result: dict) -> CandidateRecord:  # type: ignore[type-arg]
        """Parse a SerpAPI result dict into a CandidateRecord."""
        title = result.get("title", "")
        snippet = result.get("snippet", "")
        link = result.get("link", "")

        # Extract authors from publication_info
        pub_info = result.get("publication_info", {})
        authors_str = pub_info.get("summary", "")
        # Format: "Author1, Author2 - Journal, Year"
        authors_part = authors_str.split(" - ")[0] if " - " in authors_str else ""
        authors = [a.strip() for a in authors_part.split(",") if a.strip()]

        # Extract year
        year_match = re.search(r"\b(20\d{2})\b", authors_str)
        if year_match:
            published = datetime(int(year_match.group(1)), 1, 1, tzinfo=UTC)
        else:
            published = datetime.now(UTC)

        # Try to find arXiv ID
        arxiv_id, version = _extract_arxiv_id(link)
        resources = result.get("resources", [])
        if not arxiv_id and resources:
            for res in resources:
                arxiv_id, version = _extract_arxiv_id(res.get("link", ""))
                if arxiv_id:
                    break
        if not arxiv_id:
            arxiv_id = f"scholar-{abs(hash(title.lower())) % 10**10}"
            version = ""

        pdf_url = ""
        if resources:
            pdf_url = resources[0].get("link", "")

        return CandidateRecord(
            arxiv_id=arxiv_id,
            version=version or "v1",
            title=title,
            authors=authors,
            published=published,
            updated=published,
            categories=[],
            primary_category="",
            abstract=snippet,
            abs_url=link,
            pdf_url=pdf_url or link,
        )
