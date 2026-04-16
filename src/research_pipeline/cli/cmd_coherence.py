"""CLI handler for the 'coherence' command.

Evaluates multi-session knowledge coherence across pipeline runs.
"""

import logging
from pathlib import Path

import typer

from research_pipeline.config.loader import load_config
from research_pipeline.pipeline.coherence import run_coherence

logger = logging.getLogger(__name__)


def run_coherence_cmd(
    run_ids: list[str],
    config_path: Path | None = None,
    workspace: Path | None = None,
    output: Path | None = None,
) -> None:
    """Evaluate coherence across multiple pipeline runs.

    Args:
        run_ids: List of run IDs to evaluate (minimum 2).
        config_path: Path to config TOML file.
        workspace: Workspace root directory override.
        output: Output path for coherence report JSON.
    """
    if len(run_ids) < 2:
        logger.error("At least 2 run IDs required. Got %d.", len(run_ids))
        raise typer.Exit(code=1)

    config = load_config(config_path)
    ws = workspace or Path(config.workspace)

    logger.info("Evaluating coherence across %d runs: %s", len(run_ids), run_ids)

    report = run_coherence(
        run_ids=run_ids,
        workspace=ws,
        output=output,
    )

    # Print summary
    s = report.score
    typer.echo(f"\nCoherence Report ({len(run_ids)} runs)")
    typer.echo("=" * 40)
    typer.echo(f"Overall Score:              {s.overall:.2f}")
    typer.echo(f"Factual Consistency:        {s.factual_consistency:.2f}")
    typer.echo(f"Temporal Ordering:          {s.temporal_ordering:.2f}")
    typer.echo(f"Knowledge Update Fidelity:  {s.knowledge_update_fidelity:.2f}")
    typer.echo(f"Contradiction Rate:         {s.contradiction_rate:.3f}")
    typer.echo(f"Topic Overlap:              {report.topic_overlap:.2f}")
    typer.echo(f"Total Findings:             {report.finding_count}")
    typer.echo(f"Common Findings:            {report.common_finding_count}")
    typer.echo(f"Contradictions Found:       {len(report.contradictions)}")
    typer.echo(f"Knowledge Updates:          {len(report.knowledge_updates)}")
