"""arXiv search source adapter wrapping the existing ArxivClient."""

import logging

import requests

from research_pipeline.arxiv.client import ArxivClient
from research_pipeline.arxiv.dedup import dedup_across_queries
from research_pipeline.arxiv.query_builder import build_query_from_plan
from research_pipeline.arxiv.rate_limit import ArxivRateLimiter
from research_pipeline.infra.cache import FileCache
from research_pipeline.infra.clock import date_window
from research_pipeline.models.candidate import CandidateRecord
from research_pipeline.models.query_plan import QueryPlan

logger = logging.getLogger(__name__)


class ArxivSource:
    """Search source backed by the arXiv API."""

    def __init__(
        self,
        rate_limiter: ArxivRateLimiter | None = None,
        cache: FileCache | None = None,
        session: requests.Session | None = None,
        base_url: str = "https://export.arxiv.org/api/query",
        contact_email: str = "",
        request_timeout: int = 60,
        max_retries: int = 4,
        backoff_base: int = 5,
    ) -> None:
        self._client = ArxivClient(
            rate_limiter=rate_limiter,
            cache=cache,
            session=session,
            base_url=base_url,
            contact_email=contact_email,
            request_timeout=request_timeout,
            max_retries=max_retries,
            backoff_base=backoff_base,
        )

    @property
    def name(self) -> str:
        return "arxiv"

    def search(
        self,
        topic: str,
        must_terms: list[str],
        nice_terms: list[str],
        max_results: int = 100,
        date_from: str | None = None,
        date_to: str | None = None,
    ) -> list[CandidateRecord]:
        """Search arXiv using the standard query builder and client.

        Args:
            topic: Raw topic string.
            must_terms: AND-ed terms (capped at 3).
            nice_terms: Boost terms.
            max_results: Max results per query.
            date_from: Date window start.
            date_to: Date window end.

        Returns:
            Deduplicated list of candidates from arXiv.
        """
        plan = QueryPlan(
            topic_raw=topic,
            topic_normalized=topic.lower().strip(),
            must_terms=must_terms,
            nice_terms=nice_terms,
        )
        queries = build_query_from_plan(plan)

        if not date_from or not date_to:
            date_from, date_to = date_window(plan.primary_months)

        all_lists: list[list[CandidateRecord]] = []
        for q in queries:
            candidates, _ = self._client.search(
                query=q,
                max_results=max_results,
                date_from=date_from,
                date_to=date_to,
            )
            all_lists.append(candidates)

        return dedup_across_queries(all_lists)
