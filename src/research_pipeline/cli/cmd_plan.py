"""CLI handler for the 'plan' command."""

import logging
from pathlib import Path

import typer

from research_pipeline.arxiv.query_builder import (
    generate_query_variants,
    split_topic_terms,
)
from research_pipeline.config.loader import load_config
from research_pipeline.models.query_plan import QueryPlan
from research_pipeline.screening.q2d_augmentation import augment_query_plan
from research_pipeline.screening.query_cleanup import clean_query_terms
from research_pipeline.storage.workspace import get_stage_dir, init_run

logger = logging.getLogger(__name__)


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
