"""D07 — feedback rollback and audit helpers.

Phase D requires every durable preference change to declare a complete
audit envelope (trigger, procedure, observable_effect, before/after,
rollback, review_record).  This module provides:

* :func:`audit_promotion_record` — verifies a promotion record carries
  the full envelope.
* :func:`audit_feedback_sufficiency` — verifies enough explicit feedback
  is present to justify a durable change.
* :func:`safe_rollback` — rolls back an adjustment and re-runs the
  envelope audit so the rollback itself is auditable.

All functions are read-only on the caller's data; ``safe_rollback``
delegates to :func:`rollback_preference_adjustment` for the only
write.  No behavioural signals are accepted anywhere.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path

from research_pipeline.briefing.feedback import (
    NEGATIVE_SIGNALS,
    POSITIVE_SIGNALS,
    feedback_target_key,
)
from research_pipeline.briefing.models import FeedbackEvent
from research_pipeline.briefing.preference_update import (
    rollback_preference_adjustment,
)

#: Required keys for a Phase D promotion record.
REQUIRED_PROMOTION_KEYS: frozenset[str] = frozenset(
    {
        "trigger",
        "procedure",
        "observable_effect",
        "before_weight",
        "after_weight",
        "rollback",
        "review_record",
    }
)


def audit_promotion_record(
    record: Mapping[str, object],
) -> tuple[bool, list[str]]:
    """Audit a single promotion record for Phase D completeness.

    Returns ``(ok, issues)`` where ``ok`` is ``True`` only when every
    required key is present, non-empty, and ``before_weight`` differs
    from ``after_weight``.
    """
    issues: list[str] = []
    for key in sorted(REQUIRED_PROMOTION_KEYS):
        if key not in record:
            issues.append(f"missing field: {key}")
            continue
        value = record[key]
        empty = (
            value is None
            or (isinstance(value, str) and not value.strip())
            or (isinstance(value, Mapping) and not value)
        )
        if empty:
            issues.append(f"empty field: {key}")
    if not issues:
        before = record["before_weight"]
        after = record["after_weight"]
        if not isinstance(before, (int, float)) or not isinstance(after, (int, float)):
            issues.append("before/after weights must be numeric")
        elif float(before) == float(after):
            issues.append("before_weight equals after_weight (no observable change)")
    return (not issues, issues)


def audit_feedback_sufficiency(
    events: Iterable[FeedbackEvent],
    *,
    min_count: int = 3,
) -> tuple[bool, str]:
    """Audit a collection of explicit feedback events for sufficiency.

    Returns ``(ok, reason)``.  ``ok`` is ``False`` when fewer than
    ``min_count`` events exist or when the events both positively and
    negatively touch the same target (conflict).
    """
    if min_count < 1:
        raise ValueError("min_count must be >= 1")
    materialised = list(events)
    if len(materialised) < min_count:
        return (
            False,
            f"insufficient feedback: {len(materialised)} < {min_count}",
        )
    pos_targets: set[str] = set()
    neg_targets: set[str] = set()
    for event in materialised:
        key = feedback_target_key(event.target_type, event.target_id)
        if event.signal_type in POSITIVE_SIGNALS:
            pos_targets.add(key)
        elif event.signal_type in NEGATIVE_SIGNALS:
            neg_targets.add(key)
    if pos_targets & neg_targets:
        return (False, "conflicting positive and negative feedback on same target")
    return (True, "sufficient")


def safe_rollback(
    db_path: Path,
    adjustment_id: str,
    *,
    review_record: str = "",
) -> dict[str, object]:
    """Rollback a preference adjustment and emit a rollback envelope.

    The returned dict carries both the rollback receipt from
    :func:`rollback_preference_adjustment` and a ``rollback_envelope``
    describing the rollback itself with the Phase D audit fields.
    """
    receipt = rollback_preference_adjustment(db_path, adjustment_id)
    envelope: dict[str, object] = {
        "trigger": f"rollback requested for {adjustment_id}",
        "procedure": "rollback_preference_adjustment_v1",
        "observable_effect": (
            f"preference adjustment {adjustment_id} no longer alters ranking"
        ),
        "rollback": receipt.get("rollback", {}),
        "review_record": review_record or "automated-rollback",
    }
    return {
        **receipt,
        "rollback_envelope": envelope,
    }


__all__ = [
    "REQUIRED_PROMOTION_KEYS",
    "audit_feedback_sufficiency",
    "audit_promotion_record",
    "safe_rollback",
]
