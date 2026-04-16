"""CLI handler for the blinding-audit command."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from research_pipeline.evaluation.blinding import (
    HIGH_CONTAMINATION_THRESHOLD,
    run_blinding_audit_for_workspace,
)

logger = logging.getLogger(__name__)


def handle_blinding_audit(
    workspace: Path,
    *,
    run_id: str | None = None,
    threshold: float = HIGH_CONTAMINATION_THRESHOLD,
    store_results: bool = True,
    output_json: bool = False,
) -> None:
    """Run epistemic blinding audit and display results.

    Args:
        workspace: Path to the workspace directory.
        run_id: Specific run ID to audit (latest if None).
        threshold: Contamination threshold for flagging papers.
        store_results: Whether to persist results to SQLite.
        output_json: Whether to output raw JSON instead of summary.
    """
    result = run_blinding_audit_for_workspace(
        workspace,
        run_id=run_id,
        contamination_threshold=threshold,
        store_results=store_results,
    )

    if output_json:
        logger.info(json.dumps(result.to_dict(), indent=2))
        return

    logger.info("Epistemic Blinding Audit — Run: %s", result.run_id)
    logger.info("Timestamp: %s", result.timestamp)
    logger.info("Papers audited: %d", len(result.paper_scores))
    logger.info("Aggregate contamination score: %.4f", result.aggregate_score)
    logger.info("High-contamination papers: %d", len(result.high_contamination_papers))
    logger.info("Recommendation: %s", result.recommendation)

    if result.paper_scores:
        logger.info("")
        logger.info("Per-paper scores:")
        for score in sorted(
            result.paper_scores, key=lambda s: s.overall_score, reverse=True
        ):
            flag = "⚠️" if score.overall_score >= threshold else "✓"
            logger.info(
                "  %s %s: score=%.4f, refs=%d, claims=%d/%d contaminated",
                flag,
                score.paper_id[:12],
                score.overall_score,
                score.identity_references,
                score.contaminated_claims,
                score.total_claims,
            )
