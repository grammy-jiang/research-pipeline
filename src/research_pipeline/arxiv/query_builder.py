"""Query builder: convert topics and parameters to arXiv query strings."""

import logging

from research_pipeline.models.query_plan import QueryPlan

logger = logging.getLogger(__name__)


def _escape_term(term: str) -> str:
    """Escape special characters in a search term for arXiv query syntax."""
    return term.replace('"', "").strip()


def build_field_query(field: str, terms: list[str], operator: str = "AND") -> str:
    """Build a field-qualified arXiv query fragment.

    Args:
        field: arXiv field prefix (ti, abs, au, cat, all).
        terms: Search terms.
        operator: Boolean operator to join terms.

    Returns:
        Query fragment, e.g. ``ti:neural AND ti:network``.
    """
    parts = [f'{field}:"{_escape_term(t)}"' for t in terms if t.strip()]
    return f" {operator} ".join(parts)


def build_category_filter(categories: list[str]) -> str:
    """Build a category filter clause.

    Args:
        categories: List of arXiv category codes (e.g. ``cs.IR``).

    Returns:
        Query fragment, e.g. ``cat:cs.IR OR cat:cs.CL``.
    """
    if not categories:
        return ""
    return " OR ".join(f"cat:{cat}" for cat in categories)


def build_negative_filter(negative_terms: list[str]) -> str:
    """Build ANDNOT exclusion clauses.

    Args:
        negative_terms: Terms to exclude.

    Returns:
        Query fragment, e.g. ``ANDNOT all:survey ANDNOT all:tutorial``.
    """
    if not negative_terms:
        return ""
    return " ".join(
        f'ANDNOT all:"{_escape_term(t)}"' for t in negative_terms if t.strip()
    )


def build_query_from_plan(plan: QueryPlan) -> list[str]:
    """Generate arXiv query strings from a QueryPlan.

    If the plan already has ``query_variants``, returns those directly.
    Otherwise generates queries from must/nice terms and categories.

    Args:
        plan: The query plan.

    Returns:
        List of arXiv query strings ready for API use.
    """
    if plan.query_variants:
        logger.info("Using %d pre-defined query variants", len(plan.query_variants))
        return list(plan.query_variants)

    queries: list[str] = []

    # Cap must_terms to avoid overly specific AND chains
    must_terms = plan.must_terms[:3]
    if len(plan.must_terms) > 3:
        logger.info(
            "Capping must_terms from %d to 3 to avoid zero-result queries",
            len(plan.must_terms),
        )

    # Variant 1: must terms in title + abstract
    if must_terms:
        title_q = build_field_query("ti", must_terms, "AND")
        abs_q = build_field_query("abs", must_terms, "AND")
        q = f"({title_q}) OR ({abs_q})"
        if plan.candidate_categories:
            cat_q = build_category_filter(plan.candidate_categories)
            q = f"({q}) AND ({cat_q})"
        neg = build_negative_filter(plan.negative_terms)
        if neg:
            q = f"({q}) {neg}"
        queries.append(q)

    # Variant 2: must + nice terms in all fields
    if must_terms and plan.nice_terms:
        combined = must_terms + plan.nice_terms[:2]
        all_q = build_field_query("all", combined[:3], "AND")
        q = all_q
        if plan.candidate_categories:
            cat_q = build_category_filter(plan.candidate_categories)
            q = f"({q}) AND ({cat_q})"
        neg = build_negative_filter(plan.negative_terms)
        if neg:
            q = f"({q}) {neg}"
        queries.append(q)

    # Variant 3: broader search with overflow must + nice terms (OR)
    overflow_terms = plan.must_terms[3:] + plan.nice_terms[:3]
    if overflow_terms:
        nice_q = build_field_query("all", overflow_terms[:3], "OR")
        q = nice_q
        if plan.candidate_categories:
            cat_q = build_category_filter(plan.candidate_categories)
            q = f"({q}) AND ({cat_q})"
        queries.append(q)

    if not queries:
        fallback = f'all:"{_escape_term(plan.topic_normalized)}"'
        logger.warning("No terms available; using fallback query: %s", fallback)
        queries.append(fallback)

    logger.info("Generated %d query variants from plan", len(queries))
    return queries


def build_api_url(
    query: str,
    start: int = 0,
    max_results: int = 100,
    sort_by: str = "submittedDate",
    sort_order: str = "descending",
    date_from: str | None = None,
    date_to: str | None = None,
    base_url: str = "https://export.arxiv.org/api/query",
) -> str:
    """Build a full arXiv API URL from query parameters.

    Args:
        query: The search query string.
        start: Start index for pagination.
        max_results: Maximum results per page.
        sort_by: Sort field (relevance, lastUpdatedDate, submittedDate).
        sort_order: Sort direction (ascending, descending).
        date_from: Start of date window (arXiv format).
        date_to: End of date window (arXiv format).
        base_url: arXiv API base URL.

    Returns:
        Full API URL string.
    """
    search_query = query
    if date_from and date_to:
        search_query = f"({query}) AND submittedDate:[{date_from} TO {date_to}]"

    params = (
        f"search_query={search_query}"
        f"&start={start}"
        f"&max_results={max_results}"
        f"&sortBy={sort_by}"
        f"&sortOrder={sort_order}"
    )
    url = f"{base_url}?{params}"
    logger.debug("Built API URL: %s", url)
    return url


def canonical_cache_key(
    query: str,
    start: int,
    max_results: int,
    sort_by: str,
    sort_order: str,
    date_from: str | None,
    date_to: str | None,
) -> str:
    """Generate a canonical cache key for a search request.

    Args:
        query: Search query.
        start: Start index.
        max_results: Page size.
        sort_by: Sort field.
        sort_order: Sort direction.
        date_from: Date window start.
        date_to: Date window end.

    Returns:
        Deterministic cache key string.
    """
    parts = [
        f"q={query}",
        f"s={start}",
        f"m={max_results}",
        f"sb={sort_by}",
        f"so={sort_order}",
    ]
    if date_from and date_to:
        parts.append(f"df={date_from}")
        parts.append(f"dt={date_to}")
    return "|".join(parts)
