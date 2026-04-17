"""CLI handler for the cbr command (Case-Based Reasoning)."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from research_pipeline.memory.cbr import (
    cbr_lookup,
    cbr_retain,
)

logger = logging.getLogger(__name__)


def handle_cbr_lookup(
    workspace: Path,
    *,
    topic: str,
    max_results: int = 5,
    min_quality: float = 0.0,
    output_json: bool = False,
) -> None:
    """Look up past cases and recommend a strategy.

    Args:
        workspace: Path to the workspace directory.
        topic: Research topic to look up.
        max_results: Maximum cases to retrieve.
        min_quality: Minimum synthesis quality filter.
        output_json: Whether to output raw JSON.
    """
    rec = cbr_lookup(
        topic,
        workspace,
        max_results=max_results,
        min_quality=min_quality,
    )

    if output_json:
        logger.info(json.dumps(rec.to_dict(), indent=2))
        return

    logger.info("CBR Strategy Recommendation for: %s", topic)
    logger.info("Confidence: %.2f", rec.confidence)
    logger.info("Recommended sources: %s", ", ".join(rec.recommended_sources))
    logger.info("Recommended profile: %s", rec.recommended_profile)

    if rec.recommended_query_terms:
        logger.info("Suggested terms: %s", ", ".join(rec.recommended_query_terms[:5]))

    if rec.basis_cases:
        logger.info(
            "Based on %d past case(s): %s",
            len(rec.basis_cases),
            ", ".join(rec.basis_cases),
        )

    logger.info("Reasoning: %s", rec.reasoning)


def handle_cbr_retain(
    workspace: Path,
    *,
    run_id: str,
    topic: str,
    outcome: str = "unknown",
    strategy_notes: str = "",
    output_json: bool = False,
) -> None:
    """Store a completed run as a CBR case.

    Args:
        workspace: Path to the workspace directory.
        run_id: The pipeline run identifier.
        topic: Research topic for this run.
        outcome: Quality outcome label.
        strategy_notes: Free-text notes about the strategy.
        output_json: Whether to output raw JSON.
    """
    case = cbr_retain(
        run_id,
        topic,
        workspace,
        outcome=outcome,
        strategy_notes=strategy_notes,
    )

    if output_json:
        logger.info(json.dumps(case.to_dict(), indent=2))
        return

    logger.info("Stored CBR case: %s", case.case_id)
    logger.info("Topic: %s", case.topic)
    logger.info("Sources used: %s", ", ".join(case.sources_used) or "none detected")
    logger.info("Papers: %d, Shortlisted: %d", case.paper_count, case.shortlist_count)
    logger.info("Synthesis quality: %.3f", case.synthesis_quality)
    logger.info("Outcome: %s", case.outcome)
