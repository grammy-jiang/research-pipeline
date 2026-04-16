"""CLI command: aggregate — evidence-only aggregation of synthesis results."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import typer

from research_pipeline.config.loader import load_config
from research_pipeline.models.summary import SynthesisReport
from research_pipeline.storage.workspace import get_stage_dir
from research_pipeline.summarization.evidence_aggregation import (
    aggregate_evidence,
    format_aggregation_text,
)

logger = logging.getLogger(__name__)


def aggregate_cmd(
    run_id: str = typer.Option(..., "--run-id", help="Pipeline run ID."),
    min_pointers: int = typer.Option(
        0,
        "--min-pointers",
        help="Minimum evidence pointers per statement.",
    ),
    max_words: int = typer.Option(
        50,
        "--max-words",
        help="Maximum words per statement.",
    ),
    similarity_threshold: float = typer.Option(
        0.7,
        "--similarity-threshold",
        help="Threshold for merging similar statements (0-1).",
    ),
    no_strip_rhetoric: bool = typer.Option(
        False,
        "--no-strip-rhetoric",
        help="Disable rhetoric stripping.",
    ),
    output_format: str = typer.Option(
        "text",
        "--format",
        help="Output format: text or json.",
    ),
    config_path: str = typer.Option(
        "config.toml",
        "--config",
        help="Path to config file.",
    ),
) -> None:
    """Aggregate evidence from synthesis results, stripping rhetoric."""
    cfg = load_config(Path(config_path))
    runs_dir = Path(cfg.workspace)

    # Locate synthesis report
    summary_dir = get_stage_dir(runs_dir / run_id, "summarize")
    report_path = summary_dir / "synthesis.json"

    if not report_path.exists():
        logger.error("Synthesis report not found: %s", report_path)
        raise typer.Exit(code=1)

    # Load synthesis report
    raw = json.loads(report_path.read_text(encoding="utf-8"))
    report = SynthesisReport.model_validate(raw)

    logger.info(
        "Aggregating evidence from %d papers in run %s",
        report.paper_count,
        run_id,
    )

    # Run aggregation
    result = aggregate_evidence(
        report,
        min_pointers=min_pointers,
        max_words=max_words,
        similarity_threshold=similarity_threshold,
        strip_rhetoric_enabled=not no_strip_rhetoric,
    )

    # Save result
    output_path = summary_dir / "evidence_aggregation.json"
    output_path.write_text(
        result.model_dump_json(indent=2),
        encoding="utf-8",
    )
    logger.info("Saved evidence aggregation to %s", output_path)

    # Display
    if output_format == "json":
        typer.echo(result.model_dump_json(indent=2))
    else:
        typer.echo(format_aggregation_text(result))

    typer.echo(
        f"\n✓ Aggregated {result.stats.input_statements} → "
        f"{result.stats.output_statements} evidence statements"
    )
