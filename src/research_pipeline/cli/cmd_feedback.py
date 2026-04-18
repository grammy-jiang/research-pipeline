"""CLI command for user feedback on screened papers.

Records accept/reject decisions and optionally recomputes adjusted
BM25 weights using ELO-style learning.
"""

import json
import logging
from pathlib import Path

import typer

from research_pipeline.config.loader import load_config
from research_pipeline.feedback.models import FeedbackDecision, FeedbackRecord
from research_pipeline.feedback.store import FeedbackStore
from research_pipeline.infra.logging import setup_logging
from research_pipeline.storage.workspace import get_stage_dir

logger = logging.getLogger(__name__)


def feedback_cmd(
    run_id: str,
    accept: list[str] | None = None,
    reject: list[str] | None = None,
    reason: str = "",
    show: bool = False,
    adjust: bool = False,
    workspace: Path | None = None,
) -> None:
    """Record feedback and/or show stats/adjusted weights.

    Args:
        run_id: The run ID whose screened papers to give feedback on.
        accept: Paper IDs to mark as accepted.
        reject: Paper IDs to mark as rejected.
        reason: Optional reason for the decisions.
        show: If True, show current feedback stats.
        adjust: If True, recompute adjusted weights.
        workspace: Optional workspace path.
    """
    setup_logging(level=logging.INFO)
    ws = Path(workspace) if workspace else Path(load_config().workspace)
    run_root = ws / run_id

    store = FeedbackStore()

    # Load cheap scores for accepted/rejected papers
    score_map: dict[str, float] = {}
    screen_dir = get_stage_dir(run_root, "screen")
    shortlist_path = screen_dir / "shortlist.json"
    if shortlist_path.exists():
        try:
            data = json.loads(shortlist_path.read_text(encoding="utf-8"))
            for entry in data:
                paper = entry.get("paper", {})
                pid = paper.get("arxiv_id") or paper.get("doi") or ""
                score = entry.get(
                    "final_score", entry.get("cheap", {}).get("cheap_score", 0.0)
                )
                if pid:
                    score_map[pid] = score
        except (json.JSONDecodeError, KeyError) as exc:
            logger.warning("Could not load shortlist scores: %s", exc)

    # Record accepted papers
    recorded = 0
    for paper_id in accept or []:
        record = FeedbackRecord(
            paper_id=paper_id,
            run_id=run_id,
            decision=FeedbackDecision.ACCEPT,
            reason=reason,
            cheap_score=score_map.get(paper_id, 0.0),
        )
        store.record(record)
        recorded += 1
        logger.info("Accepted: %s", paper_id)

    # Record rejected papers
    for paper_id in reject or []:
        record = FeedbackRecord(
            paper_id=paper_id,
            run_id=run_id,
            decision=FeedbackDecision.REJECT,
            reason=reason,
            cheap_score=score_map.get(paper_id, 0.0),
        )
        store.record(record)
        recorded += 1
        logger.info("Rejected: %s", paper_id)

    if recorded > 0:
        logger.info("Recorded %d feedback entries for run %s", recorded, run_id)

    # Show feedback stats
    if show:
        counts = store.count(run_id=run_id)
        total_counts = store.count()
        typer.echo(f"\nFeedback for run {run_id}:")
        typer.echo(
            f"  Accept: {counts['accept']}, "
            f"Reject: {counts['reject']}, "
            f"Total: {counts['total']}"
        )
        typer.echo("\nAll-time feedback:")
        typer.echo(
            f"  Accept: {total_counts['accept']}, Reject: {total_counts['reject']}, "
            f"Total: {total_counts['total']}"
        )

        latest = store.get_latest_weights()
        if latest is not None:
            typer.echo(
                f"\nLatest adjusted weights (from {latest.feedback_count} records):"
            )
            for key, val in latest.to_weight_dict().items():
                typer.echo(f"  {key}: {val:.4f}")

    # Recompute weights
    if adjust:
        adjusted = store.compute_adjusted_weights()
        typer.echo(f"\nAdjusted weights (from {adjusted.feedback_count} records):")
        for key, val in adjusted.to_weight_dict().items():
            typer.echo(f"  {key}: {val:.4f}")

    store.close()
