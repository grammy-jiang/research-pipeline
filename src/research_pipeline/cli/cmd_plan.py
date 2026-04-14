"""CLI handler for the 'plan' command."""

import logging
from pathlib import Path

import typer

from research_pipeline.config.loader import load_config
from research_pipeline.models.query_plan import QueryPlan
from research_pipeline.storage.workspace import get_stage_dir, init_run

logger = logging.getLogger(__name__)

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


def _filter_stop_words(terms: list[str]) -> list[str]:
    """Remove common English stop words from a list of terms.

    Args:
        terms: Raw search terms (may contain stop words).

    Returns:
        Filtered list with stop words removed, preserving order.
    """
    return [t for t in terms if t.lower() not in _STOP_WORDS]


def _split_topic_terms(topic: str) -> tuple[list[str], list[str]]:
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
    content_terms = _filter_stop_words(raw_tokens)
    must_terms = content_terms[:_MAX_MUST_TERMS]
    nice_terms = content_terms[_MAX_MUST_TERMS:]
    return must_terms, nice_terms


def _generate_query_variants(
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
            _add([term] + nice_terms)
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


def run_plan(
    topic: str,
    config_path: Path | None = None,
    workspace: Path | None = None,
    run_id: str | None = None,
) -> None:
    """Execute the plan stage: normalize topic → query plan.

    Args:
        topic: Raw topic string.
        config_path: Path to config TOML.
        workspace: Workspace directory.
        run_id: Optional run ID.
    """
    config = load_config(config_path)
    ws = workspace or Path(config.workspace)
    run_id, run_root = init_run(ws, run_id)

    must_terms, nice_terms = _split_topic_terms(topic)
    query_variants = _generate_query_variants(
        must_terms, nice_terms, max_variants=config.search.max_query_variants
    )

    plan = QueryPlan(
        topic_raw=topic,
        topic_normalized=topic.lower().strip(),
        must_terms=must_terms,
        nice_terms=nice_terms,
        negative_terms=[],
        candidate_categories=[],
        query_variants=query_variants,
        primary_months=config.search.primary_months,
        fallback_months=config.search.fallback_months,
    )

    plan_dir = get_stage_dir(run_root, "plan")
    plan_path = plan_dir / "query_plan.json"
    plan_path.write_text(plan.model_dump_json(indent=2), encoding="utf-8")

    typer.echo(f"Run ID: {run_id}")
    typer.echo(f"Query plan saved: {plan_path}")
    typer.echo(f"Must terms: {plan.must_terms}")
    typer.echo(f"Nice terms: {plan.nice_terms}")
    typer.echo(f"Query variants ({len(plan.query_variants)}): {plan.query_variants}")
    logger.info("Plan stage complete for run %s", run_id)
