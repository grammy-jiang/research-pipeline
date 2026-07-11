"""Per-paper analysis task helpers (#109).

Core helpers for the `analyze` stage — paper discovery, prompt generation,
research-topic loading, and analysis-JSON validation. Shared by the CLI
(`cli/cmd_analyze.py`) and the MCP `analyze_papers` tool, so they live in Core
instead of being reached out of the presentation layer.
"""

from __future__ import annotations

import json
from pathlib import Path

from research_pipeline.storage.workspace import get_stage_dir

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


def discover_papers(run_root: Path) -> list[dict[str, str]]:
    """Find converted markdown papers in the run directory.

    Returns:
        List of dicts with 'arxiv_id' and 'path' keys.
    """
    papers = []
    seen: set[str] = set()

    def add_markdown_files(directory: Path) -> None:
        if not directory.exists():
            return
        for md_file in sorted(directory.glob("*.md")):
            if md_file.stem in seen:
                continue
            seen.add(md_file.stem)
            papers.append(
                {
                    "arxiv_id": md_file.stem,
                    "path": str(md_file),
                }
            )

    add_markdown_files(get_stage_dir(run_root, "convert"))
    add_markdown_files(run_root / "convert")

    # Also check convert_rough and convert_fine
    for stage in ("convert_rough", "convert_fine"):
        add_markdown_files(get_stage_dir(run_root, stage))

    return papers


def load_research_topic(run_root: Path) -> str:
    """Load the research topic from query plan."""
    plan_path = get_stage_dir(run_root, "plan") / "query_plan.json"
    if plan_path.exists():
        plan = json.loads(plan_path.read_text())
        return plan.get("topic", plan.get("normalized_topic", ""))  # type: ignore[no-any-return]
    return ""


def generate_prompt(
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


def validate_analysis_json(path: Path) -> list[str]:
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
