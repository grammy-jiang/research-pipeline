"""CLI handler for the 'score-claims' command.

Thin presentation wrapper (#109): the stage logic lives in
``research_pipeline.confidence.runner``; this maps its missing-input error to a
``typer.Exit`` so the Core layer stays typer-free.
"""

from __future__ import annotations

from pathlib import Path

import typer

from research_pipeline.confidence.runner import run_score_claims as _run_score_claims


def run_score_claims(
    config_path: Path | None = None,
    workspace: Path | None = None,
    run_id: str | None = None,
) -> None:
    """Score confidence for decomposed claims (CLI wrapper).

    Args:
        config_path: Path to config TOML.
        workspace: Workspace directory.
        run_id: Run ID with claim decompositions.
    """
    try:
        _run_score_claims(config_path=config_path, workspace=workspace, run_id=run_id)
    except FileNotFoundError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc
