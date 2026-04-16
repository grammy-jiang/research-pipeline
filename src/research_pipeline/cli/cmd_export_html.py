"""CLI command: export-html — render synthesis report as HTML."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import typer

from research_pipeline.config.loader import load_config
from research_pipeline.models.summary import SynthesisReport
from research_pipeline.storage.workspace import get_stage_dir
from research_pipeline.summarization.html_export import (
    render_html_from_markdown,
    render_html_report,
)

logger = logging.getLogger(__name__)


def export_html_cmd(
    run_id: str = typer.Option(
        "",
        "--run-id",
        help="Pipeline run ID (reads synthesis.json).",
    ),
    markdown_file: str = typer.Option(
        "",
        "--markdown",
        help="Path to a Markdown report to convert.",
    ),
    output: str = typer.Option(
        "",
        "--output",
        "-o",
        help="Output HTML file path (default: auto in run dir or CWD).",
    ),
    title: str = typer.Option(
        "Research Report",
        "--title",
        help="Report title (used with --markdown mode).",
    ),
    config_path: str = typer.Option(
        "config.toml",
        "--config",
        help="Path to config file.",
    ),
) -> None:
    """Export synthesis report as a self-contained HTML document.

    Supports two modes:
    1. --run-id: Render from structured SynthesisReport JSON.
    2. --markdown: Render from a raw Markdown file.
    """
    if not run_id and not markdown_file:
        logger.error("Provide either --run-id or --markdown")
        raise typer.Exit(code=1)

    if markdown_file:
        md_path = Path(markdown_file)
        if not md_path.exists():
            logger.error("Markdown file not found: %s", md_path)
            raise typer.Exit(code=1)

        out_path = Path(output) if output else md_path.with_suffix(".html")
        html_str = render_html_from_markdown(md_path, out_path, title=title)
        typer.echo(f"✓ HTML report written to {out_path}")
        logger.info("Exported Markdown → HTML (%d bytes)", len(html_str))
        return

    # Run-ID mode: load structured report
    cfg = load_config(Path(config_path))
    runs_dir = Path(cfg.workspace)

    summary_dir = get_stage_dir(runs_dir / run_id, "summarize")
    report_path = summary_dir / "synthesis.json"

    if not report_path.exists():
        logger.error("Synthesis report not found: %s", report_path)
        raise typer.Exit(code=1)

    raw = json.loads(report_path.read_text(encoding="utf-8"))
    report = SynthesisReport.model_validate(raw)

    out_path = Path(output) if output else summary_dir / "synthesis_report.html"
    html_str = render_html_report(report, out_path)

    typer.echo(f"✓ HTML report written to {out_path}")
    logger.info(
        "Exported %d-paper synthesis → HTML (%d bytes)",
        report.paper_count,
        len(html_str),
    )
