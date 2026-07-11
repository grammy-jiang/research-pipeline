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


_FIELD_PREFIXES = ("ti:", "abs:", "cat:", "au:", "all:", "co:", "jr:", "id:")


def _is_field_scoped(query: str) -> bool:
    """True if *query* already contains arXiv field syntax (operator-crafted)."""
    lowered = query.lower()
    return any(prefix in lowered for prefix in _FIELD_PREFIXES)


def _scope_variant(variant: str, plan: QueryPlan) -> str:
    """Field-scope a plain-language variant so arXiv does not treat it as a
    match-everything query (#16).

    A variant that already carries arXiv field syntax is assumed to be a
    deliberately-crafted raw query and is passed through unchanged. Otherwise
    the variant's words are AND-ed within title and abstract, then constrained
    by the plan's categories and negative terms.
    """
    if _is_field_scoped(variant):
        return variant
    terms = [t for t in variant.split() if t.strip()]
    if not terms:
        return variant
    title_q = build_field_query("ti", terms, "AND")
    abs_q = build_field_query("abs", terms, "AND")
    query = f"({title_q}) OR ({abs_q})"
    if plan.candidate_categories:
        query = f"({query}) AND ({build_category_filter(plan.candidate_categories)})"
    neg = build_negative_filter(plan.negative_terms)
    if neg:
        query = f"({query}) {neg}"
    return query


def build_query_from_plan(plan: QueryPlan) -> list[str]:
    """Generate arXiv query strings from a QueryPlan.

    If the plan already has ``query_variants``, each is field-scoped (unless it
    already contains arXiv field syntax). Otherwise generates queries from
    must/nice terms and categories.

    Args:
        plan: The query plan.

    Returns:
        List of arXiv query strings ready for API use.
    """
    if plan.query_variants:
        logger.info("Field-scoping %d query variants", len(plan.query_variants))
        return [_scope_variant(v, plan) for v in plan.query_variants]

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


# ---------------------------------------------------------------------------
# Topic → terms → query-variant planning (moved from cli/cmd_plan.py, #109)
# ---------------------------------------------------------------------------

# Common English stop words that degrade arXiv query precision.
_STOP_WORDS: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "but",
        "by",
        "do",
        "for",
        "from",
        "had",
        "has",
        "have",
        "he",
        "her",
        "his",
        "how",
        "i",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "just",
        "my",
        "no",
        "nor",
        "not",
        "of",
        "on",
        "or",
        "our",
        "out",
        "own",
        "so",
        "such",
        "than",
        "that",
        "the",
        "their",
        "them",
        "then",
        "there",
        "these",
        "they",
        "this",
        "through",
        "to",
        "too",
        "very",
        "was",
        "we",
        "were",
        "what",
        "when",
        "where",
        "which",
        "while",
        "who",
        "whom",
        "why",
        "will",
        "with",
        "would",
        "you",
        "your",
    }
)

_MAX_MUST_TERMS: int = 3


def filter_stop_words(terms: list[str]) -> list[str]:
    """Remove common English stop words from a list of terms.

    Args:
        terms: Raw search terms (may contain stop words).

    Returns:
        Filtered list with stop words removed, preserving order.
    """
    return [t for t in terms if t.lower() not in _STOP_WORDS]


def split_topic_terms(topic: str) -> tuple[list[str], list[str]]:
    """Split a topic string into must_terms and nice_terms.

    Tokenizes the topic, removes stop words, then assigns the first
    ``_MAX_MUST_TERMS`` content words to *must_terms* and the remainder
    to *nice_terms*.

    Args:
        topic: Raw topic string.

    Returns:
        Tuple of (must_terms, nice_terms).
    """
    raw_tokens = topic.lower().split()
    content_terms = filter_stop_words(raw_tokens)
    must_terms = content_terms[:_MAX_MUST_TERMS]
    nice_terms = content_terms[_MAX_MUST_TERMS:]
    return must_terms, nice_terms


def generate_query_variants(
    must_terms: list[str],
    nice_terms: list[str],
    max_variants: int = 5,
) -> list[str]:
    """Auto-generate query variants from must and nice terms.

    Strategies:
    1. All must + all nice terms (full breadth query)
    2. Must terms only (core concepts, no qualifiers)
    3. Subsets pairing each must term with all nice terms
    4. Nice terms grouped in pairs with must terms

    Args:
        must_terms: High-priority content terms.
        nice_terms: Lower-priority content terms.
        max_variants: Maximum number of variants to generate.

    Returns:
        List of query variant strings.
    """
    variants: list[str] = []
    seen: set[str] = set()

    def _add(terms: list[str]) -> None:
        q = " ".join(terms).strip()
        if q and q not in seen:
            seen.add(q)
            variants.append(q)

    # Variant 1: all terms
    _add(must_terms + nice_terms)

    # Variant 2: must terms only
    if nice_terms:
        _add(must_terms)

    # Variant 3: each must term paired with all nice terms
    if len(must_terms) > 1:
        for term in must_terms:
            _add([term, *nice_terms])
            if len(variants) >= max_variants:
                return variants[:max_variants]

    # Variant 4: must terms with subsets of nice terms
    if len(nice_terms) > 1:
        for i in range(0, len(nice_terms), 2):
            subset = nice_terms[i : i + 2]
            _add(must_terms + subset)
            if len(variants) >= max_variants:
                return variants[:max_variants]

    # Variant 5: reversed order for diversity
    _add(list(reversed(must_terms)) + nice_terms)

    # Variant 6+: Q2D (Query-to-Document) augmentation.
    # Wraps the topic in academic phrasing to match how paper abstracts
    # are written, boosting recall via "augment-don't-replace" strategy.
    topic_phrase = " ".join(must_terms + nice_terms)
    _Q2D_TEMPLATES = [
        "this paper presents {topic}",
        "we propose a method for {topic}",
        "a survey of {topic}",
    ]
    for template in _Q2D_TEMPLATES:
        _add(template.format(topic=topic_phrase).split())
        if len(variants) >= max_variants:
            return variants[:max_variants]

    return variants[:max_variants]
