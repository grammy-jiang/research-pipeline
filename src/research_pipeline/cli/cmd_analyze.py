"""CLI handler for the 'analyze' command.

Prepares per-paper analysis tasks and validates collected analysis results.
The actual LLM analysis is performed by sub-agents (paper-analyzer);
this command handles the workspace setup, prompt generation, and result
validation.
"""

import json
import logging
from pathlib import Path

from research_pipeline.config.loader import load_config
from research_pipeline.storage.workspace import get_stage_dir, init_run

logger = logging.getLogger(__name__)

ANALYSIS_SCHEMA_REQUIRED_FIELDS = {
    "arxiv_id",
    "title",
    "ratings",
    "methodology_assessment",
    "key_findings",
    "strengths",
    "weaknesses",
    "limitations",
    "evidence_quotes",
    "key_contributions",
    "reproducibility",
    "relevance_to_topic",
}

RATING_DIMENSIONS = {
    "methodology",
    "experimental_rigor",
    "novelty",
    "practical_value",
    "overall",
}


def _discover_papers(run_root: Path) -> list[dict[str, str]]:
    """Find converted markdown papers in the run directory.

    Returns:
        List of dicts with 'arxiv_id' and 'path' keys.
    """
    papers = []
    convert_dir = get_stage_dir(run_root, "convert")
    if convert_dir.exists():
        for md_file in sorted(convert_dir.glob("*.md")):
            papers.append(
                {
                    "arxiv_id": md_file.stem,
                    "path": str(md_file),
                }
            )

    # Also check convert_rough and convert_fine
    for stage in ("convert_rough", "convert_fine"):
        stage_dir = get_stage_dir(run_root, stage)
        if stage_dir.exists():
            for md_file in sorted(stage_dir.glob("*.md")):
                if not any(p["arxiv_id"] == md_file.stem for p in papers):
                    papers.append(
                        {
                            "arxiv_id": md_file.stem,
                            "path": str(md_file),
                        }
                    )

    return papers


def _load_research_topic(run_root: Path) -> str:
    """Load the research topic from query plan."""
    plan_path = get_stage_dir(run_root, "plan") / "query_plan.json"
    if plan_path.exists():
        plan = json.loads(plan_path.read_text())
        return plan.get("topic", plan.get("normalized_topic", ""))  # type: ignore[no-any-return]
    return ""


def _generate_prompt(
    paper: dict[str, str], topic: str, run_root: Path
) -> dict[str, object]:
    """Generate an analysis prompt for a single paper.

    Returns:
        Dict with prompt metadata and content.
    """
    return {
        "arxiv_id": paper["arxiv_id"],
        "paper_path": paper["path"],
        "research_topic": topic,
        "run_directory": str(run_root),
        "output_markdown": str(
            get_stage_dir(run_root, "analysis") / f"{paper['arxiv_id']}_analysis.md"
        ),
        "output_json": str(
            get_stage_dir(run_root, "analysis") / f"{paper['arxiv_id']}_analysis.json"
        ),
        "prompt": (
            f"Analyze the following paper for the research topic: {topic}\n\n"
            f"PAPER FILE: {paper['path']}\n"
            f"ARXIV ID: {paper['arxiv_id']}\n\n"
            "Produce both a Markdown analysis and a structured JSON output "
            "following the paper-analyzer schema. Write them to:\n"
            f"  Markdown: {paper['arxiv_id']}_analysis.md\n"
            f"  JSON: {paper['arxiv_id']}_analysis.json"
        ),
    }


def _validate_analysis_json(path: Path) -> list[str]:
    """Validate a single analysis JSON file against the schema.

    Returns:
        List of validation errors (empty if valid).
    """
    errors = []
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        return [f"Cannot read {path.name}: {exc}"]

    # Check required fields
    missing = ANALYSIS_SCHEMA_REQUIRED_FIELDS - set(data.keys())
    if missing:
        errors.append(f"Missing required fields: {sorted(missing)}")

    # Check ratings structure
    ratings = data.get("ratings", {})
    if isinstance(ratings, dict):
        missing_dims = RATING_DIMENSIONS - set(ratings.keys())
        if missing_dims:
            errors.append(f"Missing rating dimensions: {sorted(missing_dims)}")

        for dim, val in ratings.items():
            if dim in RATING_DIMENSIONS and isinstance(val, dict):
                score = val.get("score")
                if score is not None and (
                    not isinstance(score, int) or score < 1 or score > 5
                ):
                    errors.append(f"ratings.{dim}.score must be int 1-5, got {score}")
                justification = val.get("justification", "")
                if isinstance(justification, str) and len(justification.split()) < 5:
                    errors.append(
                        f"ratings.{dim}.justification too short (need ≥5 words)"
                    )
    else:
        errors.append("'ratings' must be a dict")

    # Check key_findings have confidence
    for i, finding in enumerate(data.get("key_findings", [])):
        if isinstance(finding, dict):
            conf = finding.get("confidence")
            if conf not in ("high", "medium", "low"):
                errors.append(f"key_findings[{i}].confidence must be high/medium/low")

    # Check evidence_quotes are present
    quotes = data.get("evidence_quotes", [])
    if not quotes:
        errors.append("evidence_quotes should not be empty")

    return errors


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
    papers = _discover_papers(run_root)
    if paper_ids:
        papers = [p for p in papers if p["arxiv_id"] in paper_ids]

    if not papers:
        logger.error("No converted papers found in run %s", run_id)
        return

    topic = _load_research_topic(run_root)
    logger.info("Preparing analysis for %d papers, topic: '%s'", len(papers), topic)

    prompts = []
    for paper in papers:
        prompt = _generate_prompt(paper, topic, run_root)
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
        errors = _validate_analysis_json(json_file)
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
