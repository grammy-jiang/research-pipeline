"""CLI command for the Unified Horizon Metric (A3-5)."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import typer

from research_pipeline.evaluation.horizon import (
    HorizonInputs,
    compute_unified_horizon_metric,
)

logger = logging.getLogger(__name__)


def horizon_cmd(
    normalized_score: float = typer.Option(
        ..., "--score", help="Normalized task quality in [0, 1]."
    ),
    difficulty: float = typer.Option(
        0.5, "--difficulty", help="Task difficulty in [0, 1]."
    ),
    achieved_steps: int = typer.Option(
        ..., "--achieved", help="Trajectory length actually completed."
    ),
    target_steps: int = typer.Option(..., "--target", help="Benchmark target horizon."),
    entropy_trend: float = typer.Option(
        0.0,
        "--entropy-trend",
        help="Token-entropy slope across trajectory (neg=locking).",
    ),
    reliability: float = typer.Option(
        1.0, "--reliability", help="Optional Pass[k] reliability floor in [0, 1]."
    ),
    output: Path | None = typer.Option(
        None, "--output", "-o", help="Write JSON result to this path."
    ),
) -> None:
    """Compute the Unified Horizon Metric (UHM) for a run.

    Combines difficulty-weighted score, horizon efficiency, stability
    (entropy trend), and reliability into a single scalar in [0, 1].
    Resolves A3-5 from the Deep Research Report.
    """
    inputs = HorizonInputs(
        normalized_score=normalized_score,
        difficulty=difficulty,
        achieved_steps=achieved_steps,
        target_steps=target_steps,
        entropy_trend=entropy_trend,
        reliability=reliability,
    )
    result = compute_unified_horizon_metric(inputs)
    payload = result.to_dict()
    text = json.dumps(payload, indent=2)
    if output:
        output.write_text(text, encoding="utf-8")
        typer.echo(f"Wrote UHM result to {output}")
    else:
        typer.echo(text)
    typer.echo(f"\nUHM = {result.uhm:.4f}")
