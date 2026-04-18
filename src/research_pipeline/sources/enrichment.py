"""Cross-source candidate enrichment.

After multi-source search and dedup, some candidates (e.g. from DBLP) may
be missing abstracts or other metadata.  This module fills in gaps by
looking up the same paper on other sources (Semantic Scholar, OpenAlex)
using DOI or title matching.
"""

import logging

import requests

from research_pipeline.infra.rate_limit import RateLimiter
from research_pipeline.infra.retry import retry
from research_pipeline.models.candidate import CandidateRecord

logger = logging.getLogger(__name__)

_S2_API_BASE = "https://api.semanticscholar.org/graph/v1"

_S2_FIELDS = (
    "paperId,externalIds,title,abstract,year,"
    "citationCount,influentialCitationCount,venue"
)


@retry(
    max_attempts=3,
    backoff_base=2.0,
    retryable_exceptions=(requests.RequestException,),
)
def _s2_lookup_by_doi(
    doi: str,
    session: requests.Session,
    rate_limiter: RateLimiter,
) -> dict | None:  # type: ignore[type-arg]
    """Look up a paper on Semantic Scholar by DOI.

    Args:
        doi: Digital Object Identifier.
        session: HTTP session.
        rate_limiter: Rate limiter for S2 API.

    Returns:
        Paper dict from S2 API, or None if not found.
    """
    rate_limiter.wait()
    url = f"{_S2_API_BASE}/paper/DOI:{doi}"
    response = session.get(url, params={"fields": _S2_FIELDS}, timeout=30)
    if response.status_code == 404:
        return None
    response.raise_for_status()
    return response.json()  # type: ignore[no-any-return]


@retry(
    max_attempts=3,
    backoff_base=2.0,
    retryable_exceptions=(requests.RequestException,),
)
def _s2_lookup_by_title(
    title: str,
    session: requests.Session,
    rate_limiter: RateLimiter,
) -> dict | None:  # type: ignore[type-arg]
    """Look up a paper on Semantic Scholar by title search.

    Uses the S2 search endpoint and returns the first match (if the
    title is a close enough match).

    Args:
        title: Paper title to search for.
        session: HTTP session.
        rate_limiter: Rate limiter for S2 API.

    Returns:
        Paper dict from S2 API, or None if no good match found.
    """
    rate_limiter.wait()
    url = f"{_S2_API_BASE}/paper/search"
    params = {"query": title, "fields": _S2_FIELDS, "limit": "3"}
    response = session.get(url, params=params, timeout=30)
    if response.status_code == 404:
        return None
    response.raise_for_status()
    data = response.json()
    papers = data.get("data", [])
    if not papers:
        return None
    # Return first result only if title is a close match
    top = papers[0]
    top_title = (top.get("title") or "").lower().strip()
    query_title = title.lower().strip()
    if top_title == query_title or top_title.startswith(query_title[:50]):
        return top  # type: ignore[no-any-return]
    return None


def enrich_candidates(
    candidates: list[CandidateRecord],
    session: requests.Session | None = None,
    s2_api_key: str = "",
    s2_rate_limiter: RateLimiter | None = None,
) -> int:
    """Enrich candidates with missing metadata from external sources.

    Currently enriches via Semantic Scholar when a DOI is available.
    Fills in missing abstracts, citation counts, and S2 IDs.

    Args:
        candidates: List of candidates to enrich (modified in-place via
            reconstruction since Pydantic models may be frozen).
        session: HTTP session for API calls.
        s2_api_key: Optional Semantic Scholar API key.
        s2_rate_limiter: Rate limiter for S2 API calls.

    Returns:
        Number of candidates that were enriched.
    """
    if session is None:
        session = requests.Session()
    if s2_api_key:
        session.headers["x-api-key"] = s2_api_key
    if s2_rate_limiter is None:
        s2_rate_limiter = RateLimiter(min_interval=1.0, name="s2_enrichment")

    enriched_count = 0

    for i, candidate in enumerate(candidates):
        needs_enrichment = not candidate.abstract or candidate.citation_count is None

        if not needs_enrichment:
            continue

        paper: dict | None = None  # type: ignore[type-arg]

        if candidate.doi:
            try:
                paper = _s2_lookup_by_doi(candidate.doi, session, s2_rate_limiter)
            except requests.RequestException as exc:
                logger.warning(
                    "Enrichment DOI lookup failed for %s (DOI: %s): %s",
                    candidate.arxiv_id,
                    candidate.doi,
                    exc,
                )

        if paper is None and candidate.title:
            try:
                paper = _s2_lookup_by_title(candidate.title, session, s2_rate_limiter)
            except requests.RequestException as exc:
                logger.warning(
                    "Enrichment title lookup failed for %s: %s",
                    candidate.arxiv_id,
                    exc,
                )

        if paper is None:
            continue

        updates: dict[str, object] = {}
        if not candidate.abstract and paper.get("abstract"):
            updates["abstract"] = paper["abstract"]
        if candidate.citation_count is None and paper.get("citationCount") is not None:
            updates["citation_count"] = paper["citationCount"]
        if (
            candidate.influential_citation_count is None
            and paper.get("influentialCitationCount") is not None
        ):
            updates["influential_citation_count"] = paper["influentialCitationCount"]
        if not candidate.semantic_scholar_id and paper.get("paperId"):
            updates["semantic_scholar_id"] = paper["paperId"]
        if not candidate.venue and paper.get("venue"):
            updates["venue"] = paper["venue"]

        if updates:
            candidates[i] = candidate.model_copy(update=updates)
            enriched_count += 1
            logger.debug(
                "Enriched %s with %d fields from S2",
                candidate.arxiv_id,
                len(updates),
            )

    logger.info("Enriched %d/%d candidates", enriched_count, len(candidates))
    return enriched_count
