"""CLI handler for the dual-metrics command."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from research_pipeline.evaluation.dual_metrics import (
    aggregate_metrics,
    evaluate_runs,
)

logger = logging.getLogger(__name__)


def handle_dual_metrics(
    workspace: Path,
    *,
    query: str,
    run_ids: list[str] | None = None,
    k: int = 5,
    store_results: bool = True,
    output_json: bool = False,
) -> None:
    """Evaluate pipeline runs using Pass@k + Pass[k] dual metrics.

    Args:
        workspace: Path to the workspace directory.
        query: Research query these runs address.
        run_ids: Specific run IDs to evaluate (auto-discover if None).
        k: Number of samples for Pass@k / Pass[k] computation.
        store_results: Whether to persist results to SQLite.
        output_json: Whether to output raw JSON instead of summary.
    """
    if run_ids is None:
        runs_dir = workspace / "runs"
        if runs_dir.exists():
            run_ids = sorted(d.name for d in runs_dir.iterdir() if d.is_dir())
        else:
            run_ids = []

    if not run_ids:
        logger.warning("No runs found in workspace %s", workspace)
        return

    result = evaluate_runs(
        workspace,
        query,
        run_ids,
        k=k,
        store_results=store_results,
    )

    if output_json:
        logger.info(json.dumps(result.to_dict(), indent=2))
        return

    logger.info("Pass@k / Pass[k] Dual Metrics — Query: %s", result.query)
    logger.info("Runs evaluated (n): %d", result.n)
    logger.info("Correct runs (c): %d", result.c)
    logger.info("k: %d", result.k)
    logger.info("")
    logger.info("Pass@%d (capability ceiling):  %.4f", result.k, result.pass_at_k)
    logger.info("Pass[%d] (reliability floor):  %.4f", result.k, result.pass_bracket_k)
    logger.info(
        "Reliability gap:              %.4f",
        result.pass_at_k - result.pass_bracket_k,
    )
    logger.info("")
    logger.info("Safety gate: %.1f", result.safety_gate)
    logger.info("Gated Pass@%d: %.4f", result.k, result.gated_pass_at_k)
    logger.info("Gated Pass[%d]: %.4f", result.k, result.gated_pass_bracket_k)
    logger.info("Fabrications detected: %d", result.fabrication_count)

    if result.samples:
        logger.info("")
        logger.info("Per-run results:")
        for sample in result.samples:
            status = "✓" if sample.correct else "✗"
            fab = " ⚠️FAB" if sample.fabrication_detected else ""
            logger.info(
                "  %s %s: quality=%.3f%s",
                status,
                sample.run_id,
                sample.quality_score,
                fab,
            )

    # Show aggregate if multiple queries stored
    agg = aggregate_metrics([result])
    if agg.total_queries > 0:
        logger.info("")
        logger.info("Aggregate: gap=%.4f", agg.reliability_gap)
