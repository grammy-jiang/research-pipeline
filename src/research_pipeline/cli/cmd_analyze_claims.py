"""CLI handler for the 'analyze-claims' command.

Thin presentation wrapper (#109): the stage logic lives in
``research_pipeline.analysis.runner``; this maps its missing-input error to a
``typer.Exit`` so the Core layer stays typer-free.
"""

from __future__ import annotations

from pathlib import Path

import typer

from research_pipeline.analysis.runner import run_analyze_claims as _run_analyze_claims


def run_analyze_claims(
    config_path: Path | None = None,
    workspace: Path | None = None,
    run_id: str | None = None,
) -> None:
    """Execute claim decomposition on paper summaries (CLI wrapper).

    Args:
        config_path: Path to config TOML.
        workspace: Workspace directory.
        run_id: Run ID with summaries.
    """
    try:
        _run_analyze_claims(config_path=config_path, workspace=workspace, run_id=run_id)
    except FileNotFoundError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc
