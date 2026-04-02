"""ArxivClient: rate-limited, single-session HTTP client for the arXiv API."""

import logging
from pathlib import Path

import requests

from arxiv_paper_pipeline.arxiv.parser import (
    CandidateRecord,
    parse_atom_response,
    parse_total_results,
)
from arxiv_paper_pipeline.arxiv.query_builder import (
    build_api_url,
    canonical_cache_key,
)
from arxiv_paper_pipeline.arxiv.rate_limit import ArxivRateLimiter
from arxiv_paper_pipeline.infra.cache import FileCache
from arxiv_paper_pipeline.infra.http import create_session

logger = logging.getLogger(__name__)


class ArxivClient:
    """Rate-limited client for the arXiv search API.

    Enforces single-connection, rate-limited access per arXiv TOU.
    Supports same-day caching to avoid redundant requests.
    """

    def __init__(
        self,
        rate_limiter: ArxivRateLimiter | None = None,
        cache: FileCache | None = None,
        session: requests.Session | None = None,
        base_url: str = "https://export.arxiv.org/api/query",
        contact_email: str = "",
        request_timeout: int = 60,
    ) -> None:
        self.rate_limiter = rate_limiter or ArxivRateLimiter()
        self.cache = cache
        self.session = session or create_session(contact_email)
        self.base_url = base_url
        self.request_timeout = request_timeout

    def _fetch_page(self, url: str, cache_key: str) -> str:
        """Fetch a single page, using cache if available.

        Args:
            url: Full API URL.
            cache_key: Cache key for this request.

        Returns:
            Raw XML response text.

        Raises:
            requests.HTTPError: On 4xx/5xx responses (after retries for 5xx).
        """
        if self.cache:
            cached = self.cache.get(cache_key)
            if cached is not None:
                logger.info("Cache hit, skipping arXiv request")
                return cached

        self.rate_limiter.wait()
        logger.info("Fetching: %s", url[:120])
        response = self.session.get(url, timeout=self.request_timeout)
        response.raise_for_status()

        xml_text = response.text

        if self.cache:
            self.cache.put(cache_key, xml_text)

        return xml_text

    def search(
        self,
        query: str,
        max_results: int = 100,
        sort_by: str = "submittedDate",
        sort_order: str = "descending",
        date_from: str | None = None,
        date_to: str | None = None,
        save_raw_dir: Path | None = None,
    ) -> tuple[list[CandidateRecord], list[str]]:
        """Execute a paginated arXiv search.

        Args:
            query: arXiv search query string.
            max_results: Total maximum results to fetch.
            sort_by: Sort field.
            sort_order: Sort direction.
            date_from: Date window start (arXiv format).
            date_to: Date window end (arXiv format).
            save_raw_dir: If set, save raw Atom XML pages to this directory.

        Returns:
            Tuple of (candidates list, raw XML page paths).
        """
        page_size = min(max_results, 100)
        all_candidates: list[CandidateRecord] = []
        raw_paths: list[str] = []
        start = 0

        while start < max_results:
            current_page_size = min(page_size, max_results - start)
            url = build_api_url(
                query=query,
                start=start,
                max_results=current_page_size,
                sort_by=sort_by,
                sort_order=sort_order,
                date_from=date_from,
                date_to=date_to,
                base_url=self.base_url,
            )
            cache_key = canonical_cache_key(
                query,
                start,
                current_page_size,
                sort_by,
                sort_order,
                date_from,
                date_to,
            )

            xml_text = self._fetch_page(url, cache_key)

            if save_raw_dir:
                save_raw_dir.mkdir(parents=True, exist_ok=True)
                raw_file = save_raw_dir / f"q_start{start}.xml"
                raw_file.write_text(xml_text, encoding="utf-8")
                raw_paths.append(str(raw_file))

            candidates = parse_atom_response(xml_text)
            total = parse_total_results(xml_text)

            all_candidates.extend(candidates)
            logger.info(
                "Page start=%d: got %d candidates (total reported: %d)",
                start,
                len(candidates),
                total,
            )

            if len(candidates) < current_page_size:
                break
            if total > 0 and start + len(candidates) >= total:
                break

            start += len(candidates)

        logger.info(
            "Search complete: %d total candidates for query",
            len(all_candidates),
        )
        return all_candidates, raw_paths
