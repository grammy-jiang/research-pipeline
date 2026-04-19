"""CLI command: report — render synthesis as templated Markdown."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import typer

from research_pipeline.config.loader import load_config
from research_pipeline.models.summary import CrossPaperSynthesisRecord, SynthesisReport
from research_pipeline.storage.workspace import get_stage_dir, init_run
from research_pipeline.summarization.report_templates import (
    list_templates,
    render_report_to_file,
)

logger = logging.getLogger(__name__)


def report_cmd(
    run_id: str = typer.Option(
        ...,
        "--run-id",
        help="Pipeline run ID.",
    ),
    template: str = typer.Option(
        "structured_synthesis",
        "--template",
        "-t",
        help=(
            "Report template: structured_synthesis, survey, gap_analysis, "
            "lit_review, executive."
        ),
    ),
    custom_template: str = typer.Option(
        "",
        "--custom-template",
        help="Path to a custom Jinja2 template file.",
    ),
    output: str = typer.Option(
        "",
        "-o",
        "--output",
        help="Output Markdown file path (default: auto in run dir).",
    ),
) -> None:
    """Render a synthesis report using a configurable template.

    Reads synthesis_report.json or synthesis.json from the summarize stage
    and renders it through a Jinja2 template to produce a formatted Markdown
    report.

    Available templates: structured_synthesis, survey, gap_analysis, lit_review,
    executive.

    Example::

        research-pipeline report --run-id <RUN_ID>
        research-pipeline report --run-id <RUN_ID> -t gap_analysis
        research-pipeline report --run-id <RUN_ID> --custom-template my.j2
    """
    available = list_templates()
    if (
        template not in available
        and template != "structured_synthesis"
        and not custom_template
    ):
        logger.error(
            "Unknown template %r. Available: %s",
            template,
            ", ".join(available),
        )
        raise typer.Exit(1)

    cfg = load_config()
    ws = Path(cfg.workspace)
    _, run_root = init_run(ws, run_id)
    stage_dir = get_stage_dir(run_root, "summarize")

    structured_json = stage_dir / "synthesis_report.json"
    legacy_json = stage_dir / "synthesis.json"
    candidates = (
        [structured_json, legacy_json]
        if template == "structured_synthesis"
        else [legacy_json, structured_json]
    )
    synthesis_json = next((path for path in candidates if path.exists()), None)
    if synthesis_json is None:
        logger.error(
            "No synthesis_report.json or synthesis.json in %s. "
            "Run the summarize stage first.",
            stage_dir,
        )
        raise typer.Exit(1)

    data = json.loads(synthesis_json.read_text(encoding="utf-8"))
    # Handle wrapped format (export.py wraps in {"report": ...})
    if "report" in data and "topic" in data["report"]:
        data = data["report"]
    if "corpus" in data and "taxonomy" in data:
        report: SynthesisReport | CrossPaperSynthesisRecord = (
            CrossPaperSynthesisRecord.model_validate(data)
        )
        if template != "structured_synthesis" and not custom_template:
            template = "structured_synthesis"
    else:
        report = SynthesisReport.model_validate(data)

    custom_tmpl: str | None = None
    if custom_template:
        tmpl_path = Path(custom_template)
        if not tmpl_path.exists():
            logger.error("Custom template not found: %s", tmpl_path)
            raise typer.Exit(1)
        custom_tmpl = tmpl_path.read_text(encoding="utf-8")

    out_path = Path(output) if output else stage_dir / f"report_{template}.md"

    render_report_to_file(
        report, out_path, template_name=template, custom_template=custom_tmpl
    )
    logger.info("Report written to %s", out_path)
