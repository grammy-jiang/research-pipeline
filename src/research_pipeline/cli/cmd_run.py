"""CLI handler for the 'run' command (full pipeline)."""

import logging
from pathlib import Path

import typer

from research_pipeline.config.loader import load_config
from research_pipeline.pipeline.orchestrator import run_pipeline

logger = logging.getLogger(__name__)


def run_full(
    topic: str,
    resume: bool = False,
    config_path: Path | None = None,
    workspace: Path | None = None,
    run_id: str | None = None,
    source: str | None = None,
    profile: str | None = None,
) -> None:
    """Execute the full pipeline end-to-end.

    Args:
        topic: Research topic.
        resume: Resume from last checkpoint.
        config_path: Path to config TOML.
        workspace: Workspace directory.
        run_id: Optional run ID.
        source: Source override (arxiv, scholar, all).
        profile: Pipeline profile override (quick, standard, deep, auto).
    """
    config = load_config(config_path)
    ws = workspace or Path(config.workspace)

    # Apply source override to config
    if source:
        if source.lower() == "all":
            config.sources.enabled = ["arxiv", "scholar"]
        else:
            config.sources.enabled = [s.strip() for s in source.split(",")]

    # Apply profile override to config
    if profile:
        config.profile = profile

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
