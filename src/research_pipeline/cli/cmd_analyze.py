"""CLI handler for the 'analyze' command.

Prepares per-paper analysis tasks and validates collected analysis results.
The actual LLM analysis is performed by sub-agents (paper-analyzer);
this command handles the workspace setup, prompt generation, and result
validation.
"""

import json
import logging
from pathlib import Path

from research_pipeline.analysis.tasks import (
    discover_papers,
    generate_prompt,
    load_research_topic,
    validate_analysis_json,
)
from research_pipeline.config.loader import load_config
from research_pipeline.storage.workspace import get_stage_dir, init_run

logger = logging.getLogger(__name__)


def run_analyze(
    config_path: Path | None = None,
    workspace: Path | None = None,
    run_id: str | None = None,
    collect: bool = False,
    paper_ids: list[str] | None = None,
) -> None:
    """Execute analyze stage: prepare prompts or validate collected results.

    Args:
        config_path: Path to config TOML file.
        workspace: Workspace root directory.
        run_id: Pipeline run ID.
        collect: If True, validate collected analysis JSON files.
        paper_ids: Optional list of specific paper IDs to analyze.
    """
    if not run_id:
        logger.error("--run-id is required for the analyze command.")
        return

    config = load_config(config_path)
    ws = workspace or Path(config.workspace)
    _run_id, run_root = init_run(ws, run_id)

    analysis_dir = get_stage_dir(run_root, "analysis")
    analysis_dir.mkdir(parents=True, exist_ok=True)

    if collect:
        _collect_and_validate(analysis_dir)
        return

    # Discover papers and generate prompts
    papers = discover_papers(run_root)
    if paper_ids:
        papers = [p for p in papers if p["arxiv_id"] in paper_ids]

    if not papers:
        logger.error("No converted papers found in run %s", run_id)
        return

    topic = load_research_topic(run_root)
    logger.info("Preparing analysis for %d papers, topic: '%s'", len(papers), topic)

    prompts = []
    for paper in papers:
        prompt = generate_prompt(paper, topic, run_root)
        prompts.append(prompt)

    # Write prompts manifest
    prompts_path = analysis_dir / "analysis_tasks.json"
    prompts_path.write_text(json.dumps(prompts, indent=2))
    logger.info("Wrote %d analysis tasks to %s", len(prompts), prompts_path)

    # Also check if any analyses already exist
    existing = list(analysis_dir.glob("*_analysis.json"))
    if existing:
        logger.info(
            "Found %d existing analysis JSON files in %s",
            len(existing),
            analysis_dir,
        )

    logger.info(
        "Analysis preparation complete. "
        "Launch paper-analyzer sub-agents for each task, "
        "then run 'analyze --collect' to validate results."
    )


def _collect_and_validate(analysis_dir: Path) -> None:
    """Collect and validate analysis JSON files."""
    json_files = sorted(analysis_dir.glob("*_analysis.json"))
    if not json_files:
        logger.error("No analysis JSON files found in %s", analysis_dir)
        return

    logger.info("Validating %d analysis files...", len(json_files))

    valid_count = 0
    total_errors = 0
    results = []

    for json_file in json_files:
        errors = validate_analysis_json(json_file)
        result = {
            "file": json_file.name,
            "valid": len(errors) == 0,
            "errors": errors,
        }
        results.append(result)

        if errors:
            logger.warning(
                "Validation errors in %s: %s", json_file.name, "; ".join(errors)
            )
            total_errors += len(errors)
        else:
            valid_count += 1

    # Write validation report
    report_path = analysis_dir / "validation_report.json"
    report_path.write_text(
        json.dumps(
            {
                "total_files": len(json_files),
                "valid": valid_count,
                "invalid": len(json_files) - valid_count,
                "total_errors": total_errors,
                "results": results,
            },
            indent=2,
        )
    )

    logger.info(
        "Validation complete: %d/%d valid, %d errors. Report: %s",
        valid_count,
        len(json_files),
        total_errors,
        report_path,
    )
