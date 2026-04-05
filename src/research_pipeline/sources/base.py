"""Abstract base for paper search sources."""

from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

from research_pipeline.models.candidate import CandidateRecord

logger = logging.getLogger(__name__)


@runtime_checkable
class SearchSource(Protocol):
    """Protocol for paper search backends.

    Each source must provide a name, a way to build queries from
    a query plan, and a search method that returns CandidateRecords.
    """

    @property
    def name(self) -> str:
        """Human-readable source name (e.g. 'arxiv', 'scholar')."""
        ...

    def search(
        self,
        topic: str,
        must_terms: list[str],
        nice_terms: list[str],
        max_results: int = 100,
        date_from: str | None = None,
        date_to: str | None = None,
    ) -> list[CandidateRecord]:
        """Execute a search and return candidate records.

        Args:
            topic: The raw topic string.
            must_terms: Terms that must appear (capped at 3).
            nice_terms: Terms that should boost relevance.
            max_results: Maximum number of results.
            date_from: Date window start.
            date_to: Date window end.

        Returns:
            List of deduplicated candidate records from this source.
        """
        ...


def dedup_cross_source(
    candidates: list[CandidateRecord],
) -> list[CandidateRecord]:
    """Deduplicate candidates from multiple sources by arxiv_id, DOI, and title.

    Uses exact arxiv_id match first, then DOI match, then normalized title
    match.  When merging duplicates, prefers the record with richer metadata
    (priority: arXiv > Semantic Scholar > OpenAlex > DBLP).

    Args:
        candidates: Flat list of candidates from all sources.

    Returns:
        Deduplicated list preserving first-seen order.
    """
    seen_ids: set[str] = set()
    seen_dois: set[str] = set()
    seen_titles: set[str] = set()
    result: list[CandidateRecord] = []

    for c in candidates:
        # Dedup by arxiv_id if available and real (not placeholder)
        if c.arxiv_id and not c.arxiv_id.startswith(
            ("scholar-", "s2-", "oalex-", "dblp-")
        ):
            if c.arxiv_id in seen_ids:
                continue
            seen_ids.add(c.arxiv_id)

        # Dedup by DOI
        if c.doi:
            doi_norm = c.doi.lower().strip()
            if doi_norm in seen_dois:
                logger.debug("Cross-source dedup (DOI): skipping '%s'", c.title[:60])
                continue
            seen_dois.add(doi_norm)

        # Dedup by normalized title
        norm_title = c.title.lower().strip()
        if norm_title in seen_titles:
            logger.debug("Cross-source dedup: skipping '%s'", c.title[:60])
            continue
        seen_titles.add(norm_title)
        result.append(c)

    removed = len(candidates) - len(result)
    if removed > 0:
        logger.info(
            "Cross-source dedup: %d candidates → %d (removed %d duplicates)",
            len(candidates),
            len(result),
            removed,
        )
    return result
