"""Query noise removal for improved search precision.

Implements the SiRe (Suppress-reduce-Expand) strategy from deep research:
1. Suppress: remove academic boilerplate terms common across all papers
2. Reduce: filter low-signal terms (too generic to discriminate)
3. Expand: (handled by Q2D augmentation in plan stage)

This module can be applied to must_terms and nice_terms before BM25 scoring
or arXiv query construction.
"""

import logging
import re

logger = logging.getLogger(__name__)

# Academic boilerplate terms that appear in most papers and provide
# no discriminative signal for topic-specific search.
_ACADEMIC_BOILERPLATE: frozenset[str] = frozenset(
    {
        "paper",
        "study",
        "propose",
        "proposed",
        "approach",
        "method",
        "methods",
        "result",
        "results",
        "experiment",
        "experiments",
        "experimental",
        "performance",
        "analysis",
        "evaluate",
        "evaluation",
        "demonstrate",
        "show",
        "shows",
        "novel",
        "new",
        "state-of-the-art",
        "sota",
        "benchmark",
        "benchmarks",
        "dataset",
        "datasets",
        "existing",
        "previous",
        "recent",
        "various",
        "several",
        "different",
        "based",
        "using",
        "used",
        "use",
        "framework",
        "system",
        "model",
        "models",
        "task",
        "tasks",
        "problem",
        "problems",
        "technique",
        "techniques",
        "present",
        "introduce",
        "introduced",
        "achieve",
        "achieved",
        "obtain",
        "obtained",
        "significant",
        "significantly",
        "improve",
        "improved",
        "improvement",
        "effective",
        "efficiently",
        "compared",
        "comparison",
    }
)

# Minimum term length for a term to be considered signal-bearing
_MIN_TERM_LENGTH = 2


def remove_academic_boilerplate(terms: list[str]) -> list[str]:
    """Remove academic boilerplate terms that add no search specificity.

    Terms like 'propose', 'method', 'result', 'novel' appear in nearly
    every academic paper abstract and degrade search precision.

    Args:
        terms: List of query terms.

    Returns:
        Filtered list with boilerplate removed.
    """
    cleaned = [t for t in terms if t.lower() not in _ACADEMIC_BOILERPLATE]
    removed = len(terms) - len(cleaned)
    if removed:
        logger.debug("Removed %d academic boilerplate terms", removed)
    return cleaned


def remove_short_terms(
    terms: list[str], min_length: int = _MIN_TERM_LENGTH
) -> list[str]:
    """Remove terms shorter than the minimum length.

    Very short terms (1-2 chars) rarely carry semantic meaning
    in academic search contexts.

    Args:
        terms: List of query terms.
        min_length: Minimum character length. Default is 2.

    Returns:
        Filtered list with short terms removed.
    """
    return [t for t in terms if len(t) >= min_length]


def deduplicate_substrings(terms: list[str]) -> list[str]:
    """Remove terms that are substrings of other terms in the list.

    If "neural" and "neural network" are both present, "neural" is
    redundant because "neural network" is more specific.

    Args:
        terms: List of query terms (may contain multi-word phrases).

    Returns:
        Deduplicated list, preferring longer (more specific) terms.
    """
    if len(terms) <= 1:
        return terms

    # Sort by length descending so longer terms are checked first
    sorted_terms = sorted(terms, key=len, reverse=True)
    result: list[str] = []

    for term in sorted_terms:
        lower_term = term.lower()
        # Check if this term is a substring of any already-kept term
        is_substring = any(
            lower_term in kept.lower() and lower_term != kept.lower() for kept in result
        )
        if not is_substring:
            result.append(term)

    # Restore original ordering
    original_order = {t.lower(): i for i, t in enumerate(terms)}
    result.sort(key=lambda t: original_order.get(t.lower(), len(terms)))
    return result


def normalize_terms(terms: list[str]) -> list[str]:
    """Normalize query terms: lowercase, strip whitespace, remove duplicates.

    Args:
        terms: Raw query terms.

    Returns:
        Normalized, deduplicated list preserving first-occurrence order.
    """
    seen: set[str] = set()
    result: list[str] = []
    for t in terms:
        normalized = t.lower().strip()
        normalized = re.sub(r"\s+", " ", normalized)
        if normalized and normalized not in seen:
            seen.add(normalized)
            result.append(normalized)
    return result


def clean_query_terms(
    terms: list[str],
    remove_boilerplate: bool = True,
    min_length: int = _MIN_TERM_LENGTH,
    deduplicate: bool = True,
) -> list[str]:
    """Apply full query noise removal pipeline.

    Steps:
    1. Normalize (lowercase, strip, dedup)
    2. Remove academic boilerplate
    3. Remove short terms
    4. Deduplicate substrings

    If all terms would be removed, returns the original normalized terms
    to avoid empty queries.

    Args:
        terms: Raw query terms.
        remove_boilerplate: Whether to remove academic boilerplate.
        min_length: Minimum term length.
        deduplicate: Whether to remove substring duplicates.

    Returns:
        Cleaned query terms.
    """
    if not terms:
        return []

    cleaned = normalize_terms(terms)

    if remove_boilerplate:
        cleaned = remove_academic_boilerplate(cleaned)

    cleaned = remove_short_terms(cleaned, min_length)

    if deduplicate:
        cleaned = deduplicate_substrings(cleaned)

    # Safety: never return empty if we started with terms
    if not cleaned:
        logger.warning(
            "Query noise removal would remove all %d terms; keeping originals",
            len(terms),
        )
        return normalize_terms(terms)

    if len(cleaned) < len(terms):
        logger.info(
            "Query noise removal: %d → %d terms",
            len(terms),
            len(cleaned),
        )

    return cleaned
