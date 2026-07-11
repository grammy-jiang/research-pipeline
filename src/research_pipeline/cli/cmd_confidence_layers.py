"""CLI handler for the 'confidence-layers' command.

Thin presentation wrapper (#109): the stage logic lives in
``research_pipeline.confidence.layers_runner``; this maps its missing-input error
to a ``typer.Exit`` so the Core layer stays typer-free.
"""

from __future__ import annotations

from pathlib import Path

import typer

from research_pipeline.confidence.layers_runner import (
    run_confidence_layers as _run_confidence_layers,
)


def run_confidence_layers(
    config_path: Path | None = None,
    workspace: Path | None = None,
    run_id: str | None = None,
    l4_threshold: float = 0.50,
    damping: float = 0.8,
    calibrate: bool = False,
) -> None:
    """Score claims through the 4-layer confidence architecture (CLI wrapper).

    Args:
        config_path: Path to config TOML.
        workspace: Workspace directory.
        run_id: Run ID with claim decompositions.
        l4_threshold: Confidence below which L4 verification triggers.
        damping: Fusion damping exponent (0-1).
        calibrate: Whether to fit Platt scaling from prior scored data.
    """
    try:
        _run_confidence_layers(
            config_path=config_path,
            workspace=workspace,
            run_id=run_id,
            l4_threshold=l4_threshold,
            damping=damping,
            calibrate=calibrate,
        )
    except FileNotFoundError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc
