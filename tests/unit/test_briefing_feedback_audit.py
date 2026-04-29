"""D07 — feedback audit and rollback tests."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from research_pipeline.briefing.feedback_audit import (
    REQUIRED_PROMOTION_KEYS,
    audit_feedback_sufficiency,
    audit_promotion_record,
    safe_rollback,
)
from research_pipeline.briefing.feedback_store import BriefingFeedbackStore
from research_pipeline.briefing.models import FeedbackEvent, FeedbackSignal
from research_pipeline.briefing.preference_update import apply_preference_updates


def _full_record() -> dict[str, object]:
    return {
        "trigger": "3 explicit feedback events",
        "procedure": "explicit_feedback_promotion_v1",
        "observable_effect": "ranking boosts topic:alpha by +0.75",
        "before_weight": 0.0,
        "after_weight": 0.75,
        "rollback": {"target": "topic:alpha", "restore_weight": 0.0},
        "review_record": "reviewer:user",
    }


def _event(target_id: str, signal: FeedbackSignal) -> FeedbackEvent:
    return FeedbackEvent(
        feedback_id=f"feedback_{target_id}_{signal.value}",
        timestamp="2026-04-27T00:00:00Z",
        target_type="topic",
        target_id=target_id,
        signal_type=signal,
        reason="r",
    )


def test_audit_promotion_record_full_passes() -> None:
    ok, issues = audit_promotion_record(_full_record())
    assert ok is True
    assert issues == []


def test_audit_promotion_record_detects_missing_field() -> None:
    record = _full_record()
    record.pop("review_record")
    ok, issues = audit_promotion_record(record)
    assert ok is False
    assert any("review_record" in issue for issue in issues)


def test_audit_promotion_record_detects_empty_review_record() -> None:
    record = _full_record()
    record["review_record"] = "   "
    ok, issues = audit_promotion_record(record)
    assert ok is False
    assert any("review_record" in issue for issue in issues)


def test_audit_promotion_record_detects_no_change() -> None:
    record = _full_record()
    record["before_weight"] = 0.5
    record["after_weight"] = 0.5
    ok, issues = audit_promotion_record(record)
    assert ok is False
    assert any("no observable change" in issue for issue in issues)


def test_required_keys_constant_is_complete() -> None:
    assert (
        frozenset(
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
        == REQUIRED_PROMOTION_KEYS
    )


def test_audit_sufficiency_insufficient() -> None:
    events = [_event("alpha", FeedbackSignal.MORE_LIKE_THIS)]
    ok, reason = audit_feedback_sufficiency(events, min_count=3)
    assert ok is False
    assert "insufficient" in reason


def test_audit_sufficiency_conflicting() -> None:
    events = [
        _event("alpha", FeedbackSignal.MORE_LIKE_THIS),
        _event("alpha", FeedbackSignal.MORE_LIKE_THIS),
        _event("alpha", FeedbackSignal.LESS_LIKE_THIS),
    ]
    ok, reason = audit_feedback_sufficiency(events, min_count=3)
    assert ok is False
    assert "conflicting" in reason


def test_audit_sufficiency_passes() -> None:
    events = [
        _event("alpha", FeedbackSignal.MORE_LIKE_THIS),
        _event("beta", FeedbackSignal.LESS_LIKE_THIS),
        _event("gamma", FeedbackSignal.MORE_LIKE_THIS),
    ]
    ok, reason = audit_feedback_sufficiency(events, min_count=3)
    assert ok is True
    assert reason == "sufficient"


def test_audit_sufficiency_min_count_must_be_positive() -> None:
    with pytest.raises(ValueError):
        audit_feedback_sufficiency([], min_count=0)


def test_safe_rollback_returns_envelope_and_clears_row(tmp_path: Path) -> None:
    db = tmp_path / "f.db"
    store = BriefingFeedbackStore(db)
    try:
        for i in range(3):
            store.record(
                target_type="topic",
                target_id="alpha",
                signal=FeedbackSignal.MORE_LIKE_THIS,
                reason=f"r{i}",
            )
        adjustments = apply_preference_updates(store, min_feedback=3)
    finally:
        store.close()
    assert len(adjustments) == 1
    adjustment_id = str(adjustments[0]["adjustment_id"])

    receipt = safe_rollback(db, adjustment_id, review_record="audit-2026-04-27")
    assert receipt["rolled_back"] is True
    envelope = receipt["rollback_envelope"]
    assert isinstance(envelope, dict)
    assert envelope["procedure"] == "rollback_preference_adjustment_v1"
    assert envelope["review_record"] == "audit-2026-04-27"

    conn = sqlite3.connect(str(db))
    try:
        rows = conn.execute("SELECT * FROM preference_adjustments").fetchall()
    finally:
        conn.close()
    assert rows == []


def test_safe_rollback_raises_for_unknown_id(tmp_path: Path) -> None:
    db = tmp_path / "f.db"
    store = BriefingFeedbackStore(db)
    store.close()
    with pytest.raises(ValueError):
        safe_rollback(db, "adjustment_unknown")
