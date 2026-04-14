"""CLI handler for the 'compare' command.

Compares two pipeline runs to produce a structured diff of papers,
findings, gaps, and confidence-level changes.
"""

import json
import logging
import statistics
from pathlib import Path

from research_pipeline.config.loader import load_config
from research_pipeline.storage.manifests import read_jsonl
from research_pipeline.storage.workspace import get_stage_dir, init_run

logger = logging.getLogger(__name__)


def _read_records(path: Path) -> list[dict]:  # type: ignore[type-arg]
    """Read records from either a JSON array file or a JSONL file."""
    if path.suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))  # type: ignore[no-any-return]
    return read_jsonl(path)


def _load_paper_ids(run_root: Path) -> set[str]:
    """Load paper IDs from a run's screen shortlist or search candidates."""
    for stage, filename in [
        ("screen", "shortlist.json"),
        ("search", "candidates.jsonl"),
    ]:
        path = get_stage_dir(run_root, stage) / filename
        if path.exists():
            records = _read_records(path)
            return {
                r.get("arxiv_id", r.get("paper", {}).get("arxiv_id", ""))
                for r in records
            }
    return set()


def _load_candidates_map(run_root: Path) -> dict[str, dict[str, object]]:
    """Load candidate records keyed by arxiv_id."""
    result = {}
    for stage, filename in [
        ("screen", "shortlist.json"),
        ("search", "candidates.jsonl"),
    ]:
        path = get_stage_dir(run_root, stage) / filename
        if path.exists():
            for r in _read_records(path):
                aid = r.get("arxiv_id", r.get("paper", {}).get("arxiv_id", ""))
                if aid:
                    result[aid] = r
            break
    return result


def _compute_jaccard(ids_a: set[str], ids_b: set[str]) -> float:
    """Compute Jaccard similarity between two paper ID sets.

    Args:
        ids_a: Paper IDs from run A.
        ids_b: Paper IDs from run B.

    Returns:
        Jaccard index in [0.0, 1.0]. Returns 0.0 when both sets are empty.
    """
    union = ids_a | ids_b
    if not union:
        return 0.0
    return len(ids_a & ids_b) / len(union)


def _quality_score_summary(
    scores_a: dict[str, float],
    scores_b: dict[str, float],
    common_ids: set[str],
) -> dict[str, float]:
    """Compute summary statistics for quality score deltas of common papers.

    Args:
        scores_a: Quality scores keyed by paper ID for run A.
        scores_b: Quality scores keyed by paper ID for run B.
        common_ids: Paper IDs present in both runs.

    Returns:
        Dict with mean_delta, median_delta, improved_count, declined_count,
        and unchanged_count.
    """
    deltas: list[float] = []
    improved = 0
    declined = 0
    unchanged = 0

    for pid in common_ids:
        sa = scores_a.get(pid)
        sb = scores_b.get(pid)
        if sa is None or sb is None:
            continue
        delta = sb - sa
        deltas.append(delta)
        if delta > 0:
            improved += 1
        elif delta < 0:
            declined += 1
        else:
            unchanged += 1

    return {
        "mean_delta": statistics.mean(deltas) if deltas else 0.0,
        "median_delta": statistics.median(deltas) if deltas else 0.0,
        "std_delta": statistics.stdev(deltas) if len(deltas) >= 2 else 0.0,
        "improved_count": float(improved),
        "declined_count": float(declined),
        "unchanged_count": float(unchanged),
    }


def _load_source_distribution(run_root: Path) -> dict[str, int]:
    """Count papers per source for a run.

    Args:
        run_root: Root directory of the pipeline run.

    Returns:
        Dict mapping source name to paper count.
    """
    candidates = _load_candidates_map(run_root)
    dist: dict[str, int] = {}
    for record in candidates.values():
        src = str(record.get("source", "unknown"))
        dist[src] = dist.get(src, 0) + 1
    return dist


def _diff_source_distributions(
    dist_a: dict[str, int],
    dist_b: dict[str, int],
) -> dict[str, object]:
    """Compare source distributions between two runs.

    Args:
        dist_a: Source distribution from run A.
        dist_b: Source distribution from run B.

    Returns:
        Dict with per-source counts for each run and sources added/removed.
    """
    all_sources = sorted(set(dist_a.keys()) | set(dist_b.keys()))
    per_source: dict[str, dict[str, int]] = {}
    for src in all_sources:
        per_source[src] = {
            "run_a": dist_a.get(src, 0),
            "run_b": dist_b.get(src, 0),
        }
    added = sorted(set(dist_b.keys()) - set(dist_a.keys()))
    removed = sorted(set(dist_a.keys()) - set(dist_b.keys()))
    return {
        "per_source": per_source,
        "added_sources": added,
        "removed_sources": removed,
    }


def _load_query_plan(run_root: Path) -> dict[str, object]:
    """Load a run's query_plan.json from the plan stage directory.

    Args:
        run_root: Root directory of the pipeline run.

    Returns:
        Parsed query plan dict, or empty dict if not found.
    """
    path = get_stage_dir(run_root, "plan") / "query_plan.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))  # type: ignore[no-any-return]
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to load query plan from %s: %s", path, exc)
        return {}


def _diff_query_plans(
    plan_a: dict[str, object],
    plan_b: dict[str, object],
) -> dict[str, object]:
    """Detect topic drift by diffing query plans.

    Args:
        plan_a: Query plan dict from run A.
        plan_b: Query plan dict from run B.

    Returns:
        Dict with raw topics and added/removed must_terms and nice_terms.
    """
    topic_a = plan_a.get("topic_raw", plan_a.get("topic_normalized", ""))
    topic_b = plan_b.get("topic_raw", plan_b.get("topic_normalized", ""))

    must_a = set(plan_a.get("must_terms", []) or [])
    must_b = set(plan_b.get("must_terms", []) or [])

    nice_a = set(plan_a.get("nice_terms", []) or [])
    nice_b = set(plan_b.get("nice_terms", []) or [])

    return {
        "topic_a": topic_a,
        "topic_b": topic_b,
        "added_must_terms": sorted(must_b - must_a),
        "removed_must_terms": sorted(must_a - must_b),
        "added_nice_terms": sorted(nice_b - nice_a),
        "removed_nice_terms": sorted(nice_a - nice_b),
    }


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

    common_ids = ids_a & ids_b
    jaccard = _compute_jaccard(ids_a, ids_b)
    quality_summary = _quality_score_summary(scores_a, scores_b, common_ids)

    plan_a = _load_query_plan(run_root_a)
    plan_b = _load_query_plan(run_root_b)
    topic_drift = _diff_query_plans(plan_a, plan_b)

    dist_a = _load_source_distribution(run_root_a)
    dist_b = _load_source_distribution(run_root_b)
    source_diff = _diff_source_distributions(dist_a, dist_b)

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
        "jaccard_similarity": jaccard,
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
        "quality_summary": quality_summary,
        "topic_drift": topic_drift,
        "source_distribution": source_diff,
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
