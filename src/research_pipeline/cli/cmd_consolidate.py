"""CLI handler for the 'consolidate' command.

Runs memory consolidation across pipeline runs — compresses old
episodes into thematic rules, prunes stale entries, tracks drift.
"""

import logging
from pathlib import Path

import typer

from research_pipeline.config.loader import load_config
from research_pipeline.pipeline.consolidation import run_consolidation

logger = logging.getLogger(__name__)


def run_consolidate_cmd(
    run_ids: list[str] | None = None,
    config_path: Path | None = None,
    workspace: Path | None = None,
    output: Path | None = None,
    dry_run: bool = False,
    capacity: int = 100,
    threshold: float = 0.8,
    min_support: int = 2,
    staleness_days: int = 90,
) -> None:
    """Run memory consolidation across pipeline runs.

    Args:
        run_ids: List of run IDs to ingest. If None, scans workspace.
        config_path: Path to config TOML file.
        workspace: Workspace root directory override.
        output: Output path for consolidation report JSON.
        dry_run: If True, compute metrics but don't modify store.
        capacity: Episode capacity before triggering consolidation.
        threshold: Fraction of capacity triggering consolidation.
        min_support: Minimum run appearances for rule promotion.
        staleness_days: Age threshold for pruning stale episodes.
    """
    config = load_config(config_path)
    ws = workspace or Path(config.workspace)

    mode = "dry-run" if dry_run else "live"
    logger.info("Running memory consolidation (%s) on workspace %s", mode, ws)

    result = run_consolidation(
        workspace=ws,
        run_ids=run_ids if run_ids else None,
        capacity=capacity,
        threshold=threshold,
        min_support=min_support,
        staleness_days=staleness_days,
        dry_run=dry_run,
        output=output,
    )

    # Print summary
    typer.echo(f"\nMemory Consolidation Report {'(DRY RUN)' if dry_run else ''}")
    typer.echo("=" * 45)
    typer.echo(f"Episodes before:   {result.episodes_before}")
    typer.echo(f"Episodes after:    {result.episodes_after}")
    typer.echo(f"Rules created:     {result.rules_created}")
    typer.echo(f"Rules updated:     {result.rules_updated}")
    typer.echo(f"Entries pruned:    {result.entries_pruned}")
    typer.echo(f"Drift measurements: {len(result.drift_metrics)}")

    if result.drift_metrics:
        avg_drift = sum(d.drift_score for d in result.drift_metrics) / len(
            result.drift_metrics
        )
        typer.echo(f"Average drift:     {avg_drift:.3f}")
