"""CLI handler for the 'plan' command."""

import logging
from pathlib import Path

import typer

from research_pipeline.config.loader import load_config
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

    from research_pipeline.pipeline.query_planning import build_query_plan

    plan = build_query_plan(topic, config)

    plan_dir = get_stage_dir(run_root, "plan")
    plan_path = plan_dir / "query_plan.json"
    plan_path.write_text(plan.model_dump_json(indent=2), encoding="utf-8")

    typer.echo(f"Run ID: {run_id}")
    typer.echo(f"Query plan saved: {plan_path}")
    typer.echo(f"Must terms: {plan.must_terms}")
    typer.echo(f"Nice terms: {plan.nice_terms}")
    typer.echo(f"Query variants ({len(plan.query_variants)}): {plan.query_variants}")
    logger.info("Plan stage complete for run %s", run_id)
