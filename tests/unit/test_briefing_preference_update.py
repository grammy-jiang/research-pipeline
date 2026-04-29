"""D05 — preference update engine tests."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from research_pipeline.briefing.feedback_store import BriefingFeedbackStore
from research_pipeline.briefing.models import FeedbackSignal
from research_pipeline.briefing.preference_update import (
    PREFERENCE_PROCEDURE,
    apply_preference_updates,
    rollback_preference_adjustment,
)


def _seed(
    store: BriefingFeedbackStore, target_id: str, signal: FeedbackSignal, *, n: int
) -> None:
    for i in range(n):
        store.record(
            target_type="topic",
            target_id=target_id,
            signal=signal,
            reason=f"r{i}",
        )


def test_insufficient_feedback_is_noop(tmp_path: Path) -> None:
    store = BriefingFeedbackStore(tmp_path / "f.db")
    try:
        store.record(
            target_type="topic",
            target_id="alpha",
            signal=FeedbackSignal.MORE_LIKE_THIS,
        )
        result = apply_preference_updates(store, min_feedback=3)
    finally:
        store.close()
    assert result == []


def test_positive_feedback_creates_audited_record(tmp_path: Path) -> None:
    db = tmp_path / "f.db"
    store = BriefingFeedbackStore(db)
    try:
        _seed(store, "alpha", FeedbackSignal.MORE_LIKE_THIS, n=3)
        result = apply_preference_updates(
            store, min_feedback=3, review_record="reviewer:human"
        )
    finally:
        store.close()
    assert len(result) == 1
    row = result[0]
    assert row["target_type"] == "topic"
    assert row["target_id"] == "alpha"
    assert float(row["after_weight"]) > 0  # type: ignore[arg-type]
    assert float(row["before_weight"]) == 0.0  # type: ignore[arg-type]
    assert row["procedure"] == PREFERENCE_PROCEDURE
    assert row["review_record"] == "reviewer:human"
    assert "boosts" in str(row["observable_effect"])
    assert "trigger" in row
    rollback = row["rollback"]
    assert isinstance(rollback, dict)
    assert rollback["restore_weight"] == 0.0


def test_negative_feedback_records_penalty(tmp_path: Path) -> None:
    store = BriefingFeedbackStore(tmp_path / "f.db")
    try:
        _seed(store, "beta", FeedbackSignal.LESS_LIKE_THIS, n=3)
        result = apply_preference_updates(store, min_feedback=3)
    finally:
        store.close()
    assert len(result) == 1
    assert float(result[0]["after_weight"]) < 0  # type: ignore[arg-type]
    assert "penalises" in str(result[0]["observable_effect"])


def test_conflicting_feedback_is_skipped_and_rolled_back(tmp_path: Path) -> None:
    db = tmp_path / "f.db"
    store = BriefingFeedbackStore(db)
    try:
        _seed(store, "gamma", FeedbackSignal.MORE_LIKE_THIS, n=2)
        _seed(store, "gamma", FeedbackSignal.LESS_LIKE_THIS, n=2)
        result = apply_preference_updates(store, min_feedback=3)
    finally:
        store.close()
    assert result == []
    # No durable adjustment row should be left behind.
    conn = sqlite3.connect(str(db))
    try:
        rows = conn.execute(
            "SELECT * FROM preference_adjustments WHERE target_id = ?",
            ("gamma",),
        ).fetchall()
    finally:
        conn.close()
    assert rows == []


def test_rollback_after_apply_restores_state(tmp_path: Path) -> None:
    db = tmp_path / "f.db"
    store = BriefingFeedbackStore(db)
    try:
        _seed(store, "alpha", FeedbackSignal.MORE_LIKE_THIS, n=3)
        result = apply_preference_updates(store, min_feedback=3)
    finally:
        store.close()
    assert len(result) == 1
    rb = rollback_preference_adjustment(db, str(result[0]["adjustment_id"]))
    assert rb["rolled_back"] is True
    conn = sqlite3.connect(str(db))
    try:
        rows = conn.execute("SELECT * FROM preference_adjustments").fetchall()
    finally:
        conn.close()
    assert rows == []


def test_min_feedback_must_be_positive(tmp_path: Path) -> None:
    store = BriefingFeedbackStore(tmp_path / "f.db")
    try:
        with pytest.raises(ValueError):
            apply_preference_updates(store, min_feedback=0)
    finally:
        store.close()
