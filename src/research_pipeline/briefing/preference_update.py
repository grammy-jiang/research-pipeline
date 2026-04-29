"""Reviewable preference update and rollback helpers.

Phase D rule: every durable preference change must record

* ``trigger`` — the event/feedback that caused it
* ``procedure`` — the deterministic rule applied
* ``observable_effect`` — what a human can verify after the change
* ``before_weight`` / ``after_weight``
* ``rollback`` — the inverse operation that restores the prior state
* ``review_record`` — the human/audit record that reviewed the change

Conflicting feedback (positive + negative on the same target) and
insufficient feedback (fewer than ``min_feedback`` events overall) must
NOT silently change durable preferences.  This engine enforces both.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from research_pipeline.briefing.feedback import (
    NEGATIVE_SIGNALS,
    POSITIVE_SIGNALS,
    BriefingFeedbackStore,
)

#: Fixed promotion-rule procedure name for D05 explicit-feedback updates.
PREFERENCE_PROCEDURE = "explicit_feedback_promotion_v1"


def rollback_preference_adjustment(
    db_path: Path, adjustment_id: str
) -> dict[str, object]:
    """Rollback a stored briefing preference adjustment by deleting it."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute(
            "SELECT * FROM preference_adjustments WHERE adjustment_id = ?",
            (adjustment_id,),
        ).fetchone()
        if row is None:
            raise ValueError(f"preference adjustment not found: {adjustment_id}")
        rollback = json.loads(row["rollback_json"])
        conn.execute(
            "DELETE FROM preference_adjustments WHERE adjustment_id = ?",
            (adjustment_id,),
        )
        conn.commit()
        return {
            "adjustment_id": adjustment_id,
            "rolled_back": True,
            "rollback": rollback,
        }
    finally:
        conn.close()


def _observable_effect(target_type: str, target_id: str, after: float) -> str:
    direction = "boosts" if after > 0 else "penalises" if after < 0 else "leaves"
    return (
        f"ranking {direction} {target_type}:{target_id} by {after:+.3f} "
        f"on subsequent runs"
    )


def apply_preference_updates(
    store: BriefingFeedbackStore,
    *,
    min_feedback: int = 3,
    review_record: str = "",
) -> list[dict[str, object]]:
    """Apply reversible preference updates from explicit feedback.

    * No-op when fewer than ``min_feedback`` total events are recorded.
    * Skips any target whose feedback contains both positive and
      negative signals (conflict); the corresponding row created by
      :meth:`BriefingFeedbackStore.create_adjustments` is rolled back so
      no durable change is left behind.
    * Augments the audit row with ``procedure``, ``observable_effect``,
      and ``review_record`` so that downstream auditing has every Phase
      D-required field.
    """
    if not isinstance(min_feedback, int) or min_feedback < 1:
        raise ValueError("min_feedback must be a positive integer")

    feedback = store.list_feedback()
    if len(feedback) < min_feedback:
        return []

    raw_adjustments = store.create_adjustments(min_feedback=min_feedback)
    if not raw_adjustments:
        return []

    db_path = Path(store.db_path)
    surviving: list[dict[str, object]] = []
    for row in raw_adjustments:
        conflict = row.get("conflict") or {}
        if conflict:
            # Defensive: pull the durable change back so conflicts do not
            # silently shift preferences.
            rollback_preference_adjustment(db_path, str(row["adjustment_id"]))
            continue
        target_type = str(row["target_type"])
        target_id = str(row["target_id"])
        after_weight = float(row["after_weight"])  # type: ignore[arg-type]
        augmented: dict[str, object] = {
            **row,
            "procedure": PREFERENCE_PROCEDURE,
            "observable_effect": _observable_effect(
                target_type, target_id, after_weight
            ),
            "review_record": review_record,
        }
        surviving.append(augmented)
    return surviving


__all__ = [
    "NEGATIVE_SIGNALS",
    "POSITIVE_SIGNALS",
    "PREFERENCE_PROCEDURE",
    "apply_preference_updates",
    "rollback_preference_adjustment",
]
