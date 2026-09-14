"""CLI handler for the 'plan' command."""

from research_pipeline.arxiv.query_builder import (
    generate_query_variants,
    split_topic_terms,
)
from research_pipeline.config.models import PipelineConfig
from research_pipeline.models.query_plan import QueryPlan
from research_pipeline.screening.q2d_augmentation import augment_query_plan
from research_pipeline.screening.query_cleanup import clean_query_terms


def build_query_plan(topic: str, config: PipelineConfig) -> QueryPlan:
    """Build the same normalized, expanded query plan for CLI and MCP."""
    must_terms, nice_terms = split_topic_terms(topic)

    # Apply query noise removal (SiRe strategy: suppress academic boilerplate)
    must_terms = clean_query_terms(must_terms, remove_boilerplate=True)
    nice_terms = clean_query_terms(nice_terms, remove_boilerplate=True)

    query_variants = generate_query_variants(
        must_terms, nice_terms, max_variants=config.search.max_query_variants
    )

    # Q2D augmentation: expand domain synonyms + generate pseudo-abstract queries
    query_variants = augment_query_plan(
        must_terms,
        nice_terms,
        existing_variants=query_variants,
        max_total_variants=config.search.max_query_variants,
    )

    return QueryPlan(
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
