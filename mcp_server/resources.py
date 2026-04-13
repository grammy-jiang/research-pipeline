"""MCP resource handlers for research-pipeline.

Exposes pipeline artifacts (runs, manifests, papers, markdown, config)
as MCP resources with URI templates.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_WORKSPACE = "./workspace"
DEFAULT_RUNS_DIR = "./runs"


def _find_runs_root() -> Path:
    """Find the runs directory, checking workspace/ and runs/ locations."""
    for candidate in [DEFAULT_RUNS_DIR, DEFAULT_WORKSPACE]:
        p = Path(candidate).resolve()
        if p.is_dir():
            return p
    return Path(DEFAULT_RUNS_DIR).resolve()


def _get_run_root(run_id: str) -> Path | None:
    """Find the root directory for a specific run."""
    roots = [Path(DEFAULT_RUNS_DIR), Path(DEFAULT_WORKSPACE)]
    for root in roots:
        run_path = root.resolve() / run_id
        if run_path.is_dir():
            return run_path
    return None


def list_runs() -> str:
    """List all pipeline runs as JSON."""
    runs_root = _find_runs_root()
    if not runs_root.is_dir():
        return json.dumps({"runs": [], "runs_dir": str(runs_root)})

    runs = []
    for entry in sorted(runs_root.iterdir()):
        if entry.is_dir() and not entry.name.startswith("."):
            manifest_path = entry / "run_manifest.json"
            info: dict = {"run_id": entry.name, "path": str(entry)}
            if manifest_path.exists():
                try:
                    manifest = json.loads(manifest_path.read_text())
                    info["topic"] = manifest.get("topic", "")
                    info["stages"] = list(manifest.get("stages", {}).keys())
                except (json.JSONDecodeError, OSError):
                    pass
            runs.append(info)

    return json.dumps({"runs": runs, "runs_dir": str(runs_root)})


def get_run_manifest(run_id: str) -> str:
    """Read a run's manifest as JSON."""
    run_root = _get_run_root(run_id)
    if run_root is None:
        return json.dumps({"error": f"Run '{run_id}' not found"})
    manifest_path = run_root / "run_manifest.json"
    if not manifest_path.exists():
        return json.dumps({"error": f"No manifest for run '{run_id}'"})
    return manifest_path.read_text()


def get_run_plan(run_id: str) -> str:
    """Read a run's query plan as JSON."""
    run_root = _get_run_root(run_id)
    if run_root is None:
        return json.dumps({"error": f"Run '{run_id}' not found"})
    plan_path = run_root / "plan" / "query_plan.json"
    if not plan_path.exists():
        return json.dumps({"error": f"No plan for run '{run_id}'"})
    return plan_path.read_text()


def get_run_candidates(run_id: str) -> str:
    """Read a run's search candidates as JSONL."""
    run_root = _get_run_root(run_id)
    if run_root is None:
        return json.dumps({"error": f"Run '{run_id}' not found"})
    candidates_path = run_root / "search" / "candidates.jsonl"
    if not candidates_path.exists():
        return json.dumps({"error": f"No candidates for run '{run_id}'"})
    return candidates_path.read_text()


def get_run_shortlist(run_id: str) -> str:
    """Read a run's screened shortlist as JSON."""
    run_root = _get_run_root(run_id)
    if run_root is None:
        return json.dumps({"error": f"Run '{run_id}' not found"})
    shortlist_path = run_root / "screen" / "shortlist.json"
    if not shortlist_path.exists():
        return json.dumps({"error": f"No shortlist for run '{run_id}'"})
    return shortlist_path.read_text()


def get_paper_pdf(run_id: str, paper_id: str) -> bytes:
    """Read a downloaded paper's PDF as bytes."""
    run_root = _get_run_root(run_id)
    if run_root is None:
        return b""
    pdf_dir = run_root / "download" / "pdf"
    # Try exact filename, then glob
    for pattern in [f"{paper_id}.pdf", f"*{paper_id}*.pdf"]:
        matches = list(pdf_dir.glob(pattern))
        if matches:
            return matches[0].read_bytes()
    return b""


def get_paper_markdown(run_id: str, paper_id: str) -> str:
    """Read a paper's converted markdown."""
    run_root = _get_run_root(run_id)
    if run_root is None:
        return json.dumps({"error": f"Run '{run_id}' not found"})

    # Check convert/markdown, convert_rough, convert_fine
    for subdir in ["convert/markdown", "convert_fine", "convert_rough"]:
        md_dir = run_root / subdir
        if not md_dir.is_dir():
            continue
        for pattern in [f"{paper_id}.md", f"*{paper_id}*.md"]:
            matches = list(md_dir.glob(pattern))
            if matches:
                return matches[0].read_text()

    return json.dumps(
        {"error": f"No markdown for paper '{paper_id}' in run '{run_id}'"}
    )


def get_paper_summary(run_id: str, paper_id: str) -> str:
    """Read a paper's summary as JSON."""
    run_root = _get_run_root(run_id)
    if run_root is None:
        return json.dumps({"error": f"Run '{run_id}' not found"})
    summary_dir = run_root / "summarize"
    for pattern in [f"{paper_id}_summary.json", f"*{paper_id}*.json"]:
        matches = list(summary_dir.glob(pattern))
        if matches:
            return matches[0].read_text()
    return json.dumps({"error": f"No summary for paper '{paper_id}' in run '{run_id}'"})


def get_synthesis_report(run_id: str) -> str:
    """Read a run's cross-paper synthesis report."""
    run_root = _get_run_root(run_id)
    if run_root is None:
        return json.dumps({"error": f"Run '{run_id}' not found"})
    synthesis_path = run_root / "summarize" / "synthesis_report.md"
    if not synthesis_path.exists():
        # Try JSON variant
        synthesis_path = run_root / "summarize" / "synthesis_report.json"
    if not synthesis_path.exists():
        return json.dumps({"error": f"No synthesis report for run '{run_id}'"})
    return synthesis_path.read_text()


def get_quality_scores(run_id: str) -> str:
    """Read a run's quality evaluation scores as JSON."""
    run_root = _get_run_root(run_id)
    if run_root is None:
        return json.dumps({"error": f"Run '{run_id}' not found"})
    quality_path = run_root / "quality" / "quality_scores.json"
    if not quality_path.exists():
        # Try JSONL variant
        quality_path = run_root / "quality" / "quality_scores.jsonl"
    if not quality_path.exists():
        return json.dumps({"error": f"No quality scores for run '{run_id}'"})
    return quality_path.read_text()


def get_current_config() -> str:
    """Read the current pipeline configuration as TOML."""
    config_path = Path("config.toml")
    if not config_path.exists():
        config_path = Path("config.example.toml")
    if not config_path.exists():
        return "# No configuration file found"
    return config_path.read_text()


def get_global_index() -> str:
    """Read the global paper index summary as JSON."""
    try:
        from research_pipeline.storage.global_index import GlobalPaperIndex

        idx = GlobalPaperIndex()
        papers = idx.list_papers()
        return json.dumps(
            {
                "count": len(papers),
                "papers": [p.model_dump(mode="json") for p in papers[:100]],
            }
        )
    except Exception as exc:
        return json.dumps({"error": str(exc), "count": 0, "papers": []})
