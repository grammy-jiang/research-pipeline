"""CLI handler for the 'run' command (full pipeline)."""

import logging
from pathlib import Path

import typer

from arxiv_paper_pipeline.config.loader import load_config
from arxiv_paper_pipeline.pipeline.orchestrator import run_pipeline

logger = logging.getLogger(__name__)


def run_full(
    topic: str,
    resume: bool = False,
    config_path: Path | None = None,
    workspace: Path | None = None,
    run_id: str | None = None,
) -> None:
    """Execute the full pipeline end-to-end.

    Args:
        topic: Research topic.
        resume: Resume from last checkpoint.
        config_path: Path to config TOML.
        workspace: Workspace directory.
        run_id: Optional run ID.
    """
    config = load_config(config_path)
    ws = workspace or Path(config.workspace)

    manifest = run_pipeline(
        topic=topic,
        config=config,
        run_id=run_id,
        resume=resume,
        workspace=ws,
    )

    typer.echo("Pipeline complete!")
    typer.echo(f"Run ID: {manifest.run_id}")

    completed = [s for s, r in manifest.stages.items() if r.status == "completed"]
    failed = [s for s, r in manifest.stages.items() if r.status == "failed"]

    typer.echo(f"Completed stages: {', '.join(completed)}")
    if failed:
        typer.echo(f"Failed stages: {', '.join(failed)}")

    logger.info("Full pipeline run complete: %s", manifest.run_id)
