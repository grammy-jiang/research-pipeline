"""ArxivClient: rate-limited, single-session HTTP client for the arXiv API."""

import logging
import time
from pathlib import Path

import requests

from research_pipeline.arxiv.parser import (
    CandidateRecord,
    parse_atom_response,
    parse_total_results,
)
from research_pipeline.arxiv.query_builder import (
    build_api_url,
    canonical_cache_key,
)
from research_pipeline.arxiv.rate_limit import ArxivRateLimiter
from research_pipeline.infra.cache import FileCache
from research_pipeline.infra.http import create_session

logger = logging.getLogger(__name__)

_DEFAULT_MAX_RETRIES = 4
_DEFAULT_BACKOFF_BASE = 5


class ArxivClient:
    """Rate-limited client for the arXiv search API.

    Enforces single-connection, rate-limited access per arXiv TOU.
    Supports same-day caching to avoid redundant requests.
    Handles HTTP 429 with exponential backoff.
    """

    def __init__(
        self,
        rate_limiter: ArxivRateLimiter | None = None,
        cache: FileCache | None = None,
        session: requests.Session | None = None,
        base_url: str = "https://export.arxiv.org/api/query",
        contact_email: str = "",
        request_timeout: int = 60,
        max_retries: int = _DEFAULT_MAX_RETRIES,
        backoff_base: int = _DEFAULT_BACKOFF_BASE,
    ) -> None:
        self.rate_limiter = rate_limiter or ArxivRateLimiter()
        self.cache = cache
        self.session = session or create_session(contact_email)
        self.base_url = base_url
        self.request_timeout = request_timeout
        self.max_retries = max_retries
        self.backoff_base = backoff_base

    def _fetch_page(self, url: str, cache_key: str) -> str:
        """Fetch a single page, using cache if available.

        Handles HTTP 429 (rate exceeded) with exponential backoff.

        Args:
            url: Full API URL.
            cache_key: Cache key for this request.

        Returns:
            Raw XML response text.

        Raises:
            requests.HTTPError: On non-retryable HTTP errors.
            requests.ReadTimeout: If all retries are exhausted.
        """
        if self.cache:
            cached = self.cache.get(cache_key)
            if cached is not None:
                logger.info("Cache hit, skipping arXiv request")
                return cached

        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            self.rate_limiter.wait()
            logger.info("Fetching: %s", url[:120])
            try:
                response = self.session.get(url, timeout=self.request_timeout)
            except requests.ReadTimeout as exc:
                last_exc = exc
                wait = self.backoff_base * (2**attempt)
                logger.warning(
                    "Request timed out (attempt %d/%d). " "Retrying in %ds...",
                    attempt + 1,
                    self.max_retries + 1,
                    wait,
                )
                time.sleep(wait)
                continue

            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After")
                if retry_after and retry_after.isdigit():
                    wait = int(retry_after)
                else:
                    wait = self.backoff_base * (2**attempt)
                logger.warning(
                    "arXiv rate limit (429) hit (attempt %d/%d). " "Backing off %ds...",
                    attempt + 1,
                    self.max_retries + 1,
                    wait,
                )
                time.sleep(wait)
                continue

            response.raise_for_status()

            xml_text = response.text
            if self.cache:
                self.cache.put(cache_key, xml_text)
            return xml_text

        if last_exc is not None:
            raise last_exc
        msg = (
            f"arXiv API returned 429 after {self.max_retries + 1} attempts. "
            "Try again later or reduce request frequency."
        )
        raise requests.HTTPError(msg)

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
