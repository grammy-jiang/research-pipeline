"""CLI handler for the 'compare' command.

Compares two pipeline runs to produce a structured diff of papers,
findings, gaps, and confidence-level changes.
"""

import json
import logging
from pathlib import Path

from research_pipeline.config.loader import load_config
from research_pipeline.storage.manifests import read_jsonl
from research_pipeline.storage.workspace import get_stage_dir, init_run

logger = logging.getLogger(__name__)


def _load_paper_ids(run_root: Path) -> set[str]:
    """Load paper IDs from a run's screen shortlist or search candidates."""
    for stage, filename in [
        ("screen", "shortlist.jsonl"),
        ("search", "candidates.jsonl"),
    ]:
        path = get_stage_dir(run_root, stage) / filename
        if path.exists():
            records = read_jsonl(path)
            return {
                r.get("arxiv_id", r.get("paper", {}).get("arxiv_id", ""))
                for r in records
            }
    return set()


def _load_candidates_map(run_root: Path) -> dict[str, dict[str, object]]:
    """Load candidate records keyed by arxiv_id."""
    result = {}
    for stage, filename in [
        ("screen", "shortlist.jsonl"),
        ("search", "candidates.jsonl"),
    ]:
        path = get_stage_dir(run_root, stage) / filename
        if path.exists():
            for r in read_jsonl(path):
                aid = r.get("arxiv_id", r.get("paper", {}).get("arxiv_id", ""))
                if aid:
                    result[aid] = r
            break
    return result


def _load_synthesis_json(run_root: Path) -> dict[str, object] | None:
    """Load synthesis_results.json if it exists."""
    for candidate_path in [
        run_root / "synthesis" / "synthesis_results.json",
        get_stage_dir(run_root, "summarize") / "synthesis_results.json",
    ]:
        if candidate_path.exists():
            try:
                return json.loads(candidate_path.read_text())
            except (json.JSONDecodeError, OSError):
                continue
    return None


def _load_quality_scores(run_root: Path) -> dict[str, float]:
    """Load quality scores keyed by paper_id."""
    path = get_stage_dir(run_root, "quality") / "quality_scores.jsonl"
    if not path.exists():
        return {}
    scores = {}
    for r in read_jsonl(path):
        pid = r.get("paper_id", "")
        if pid:
            scores[pid] = r.get("composite_score", 0.0)
    return scores


def _diff_paper_sets(ids_a: set[str], ids_b: set[str]) -> dict[str, list[str]]:
    """Diff two sets of paper IDs."""
    return {
        "only_in_run_a": sorted(ids_a - ids_b),
        "only_in_run_b": sorted(ids_b - ids_a),
        "in_both": sorted(ids_a & ids_b),
    }


def _diff_gaps(
    synth_a: dict[str, object] | None,
    synth_b: dict[str, object] | None,
) -> dict[str, object]:
    """Compare gaps between two synthesis results."""
    gaps_a = {
        g.get("description", ""): g
        for g in (synth_a or {}).get("gaps", [])
        if isinstance(g, dict)
    }
    gaps_b = {
        g.get("description", ""): g
        for g in (synth_b or {}).get("gaps", [])
        if isinstance(g, dict)
    }

    resolved = []
    for desc in gaps_a:
        if desc not in gaps_b:
            resolved.append({"description": desc, "was": gaps_a[desc]})

    new_gaps = []
    for desc in gaps_b:
        if desc not in gaps_a:
            new_gaps.append(gaps_b[desc])

    persistent = []
    for desc in gaps_a:
        if desc in gaps_b:
            persistent.append(
                {
                    "description": desc,
                    "run_a": gaps_a[desc],
                    "run_b": gaps_b[desc],
                }
            )

    return {
        "resolved_gaps": resolved,
        "new_gaps": new_gaps,
        "persistent_gaps": persistent,
    }


def _diff_confidence(
    synth_a: dict[str, object] | None,
    synth_b: dict[str, object] | None,
) -> list[dict[str, object]]:
    """Track confidence-level changes for findings across runs."""
    changes = []

    findings_a = {}
    for level in ("high", "medium", "low"):
        for f in (synth_a or {}).get("confidence_graded_findings", {}).get(level, []):
            if isinstance(f, dict):
                findings_a[f.get("finding", "")] = level

    findings_b = {}
    for level in ("high", "medium", "low"):
        for f in (synth_b or {}).get("confidence_graded_findings", {}).get(level, []):
            if isinstance(f, dict):
                findings_b[f.get("finding", "")] = level

    for finding, level_a in findings_a.items():
        level_b = findings_b.get(finding)
        if level_b and level_b != level_a:
            changes.append(
                {
                    "finding": finding,
                    "confidence_run_a": level_a,
                    "confidence_run_b": level_b,
                    "direction": (
                        "upgraded"
                        if ["low", "medium", "high"].index(level_b)
                        > ["low", "medium", "high"].index(level_a)
                        else "downgraded"
                    ),
                }
            )

    return changes


def _diff_readiness(
    synth_a: dict[str, object] | None,
    synth_b: dict[str, object] | None,
) -> dict[str, object]:
    """Compare readiness assessments."""
    ra = (synth_a or {}).get("readiness", {})
    rb = (synth_b or {}).get("readiness", {})

    if not isinstance(ra, dict):
        ra = {}
    if not isinstance(rb, dict):
        rb = {}

    return {
        "verdict_run_a": ra.get("verdict", "UNKNOWN"),
        "verdict_run_b": rb.get("verdict", "UNKNOWN"),
        "criteria_changes": {
            dim: {
                "run_a": ra.get("criteria_met", {}).get(dim, "unknown"),
                "run_b": rb.get("criteria_met", {}).get(dim, "unknown"),
            }
            for dim in set(
                list(ra.get("criteria_met", {}).keys())
                + list(rb.get("criteria_met", {}).keys())
            )
            if ra.get("criteria_met", {}).get(dim)
            != rb.get("criteria_met", {}).get(dim)
        },
    }


def compare_runs(
    run_root_a: Path,
    run_root_b: Path,
    run_id_a: str,
    run_id_b: str,
) -> dict[str, object]:
    """Compare two pipeline runs and produce a structured diff.

    Args:
        run_root_a: Root directory for run A.
        run_root_b: Root directory for run B.
        run_id_a: Run ID for run A.
        run_id_b: Run ID for run B.

    Returns:
        Comparison result dict.
    """
    ids_a = _load_paper_ids(run_root_a)
    ids_b = _load_paper_ids(run_root_b)
    paper_diff = _diff_paper_sets(ids_a, ids_b)

    synth_a = _load_synthesis_json(run_root_a)
    synth_b = _load_synthesis_json(run_root_b)

    scores_a = _load_quality_scores(run_root_a)
    scores_b = _load_quality_scores(run_root_b)

    gap_diff = _diff_gaps(synth_a, synth_b)
    confidence_changes = _diff_confidence(synth_a, synth_b)
    readiness_diff = _diff_readiness(synth_a, synth_b)

    return {
        "run_a": run_id_a,
        "run_b": run_id_b,
        "paper_diff": {
            **paper_diff,
            "count_a": len(ids_a),
            "count_b": len(ids_b),
            "overlap": len(paper_diff["in_both"]),
            "new_in_b": len(paper_diff["only_in_run_b"]),
            "dropped_from_a": len(paper_diff["only_in_run_a"]),
        },
        "gap_analysis": {
            **gap_diff,
            "resolved_count": len(gap_diff["resolved_gaps"]),
            "new_count": len(gap_diff["new_gaps"]),
            "persistent_count": len(gap_diff["persistent_gaps"]),
        },
        "confidence_changes": confidence_changes,
        "readiness": readiness_diff,
        "quality_score_changes": {
            pid: {
                "run_a": scores_a.get(pid),
                "run_b": scores_b.get(pid),
            }
            for pid in set(list(scores_a.keys()) + list(scores_b.keys()))
            if scores_a.get(pid) != scores_b.get(pid)
        },
        "has_synthesis_a": synth_a is not None,
        "has_synthesis_b": synth_b is not None,
    }


def run_compare(
    run_id_a: str | None = None,
    run_id_b: str | None = None,
    config_path: Path | None = None,
    workspace: Path | None = None,
    output: Path | None = None,
) -> None:
    """Compare two pipeline runs and produce a diff report.

    Args:
        run_id_a: First run ID (baseline).
        run_id_b: Second run ID (latest).
        config_path: Path to config TOML file.
        workspace: Workspace root directory.
        output: Output path for comparison report.
    """
    if not run_id_a or not run_id_b:
        logger.error("Both --run-a and --run-b are required.")
        return

    config = load_config(config_path)
    ws = workspace or Path(config.workspace)

    _, run_root_a = init_run(ws, run_id_a)
    _, run_root_b = init_run(ws, run_id_b)

    logger.info("Comparing runs: %s vs %s", run_id_a, run_id_b)

    result = compare_runs(run_root_a, run_root_b, run_id_a, run_id_b)

    # Log summary
    pd = result["paper_diff"]
    logger.info(
        "Papers: %d in A, %d in B, %d overlap, %d new, %d dropped",
        pd["count_a"],
        pd["count_b"],
        pd["overlap"],
        pd["new_in_b"],
        pd["dropped_from_a"],
    )

    ga = result["gap_analysis"]
    logger.info(
        "Gaps: %d resolved, %d new, %d persistent",
        ga["resolved_count"],
        ga["new_count"],
        ga["persistent_count"],
    )

    if result["confidence_changes"]:
        logger.info(
            "Confidence changes: %d findings changed level",
            len(result["confidence_changes"]),
        )

    rd = result["readiness"]
    logger.info("Readiness: %s → %s", rd["verdict_run_a"], rd["verdict_run_b"])

    # Write output
    out_path = output or Path(f"comparison_{run_id_a}_vs_{run_id_b}.json")
    out_path.write_text(json.dumps(result, indent=2, default=str))
    logger.info("Comparison report written to %s", out_path)
