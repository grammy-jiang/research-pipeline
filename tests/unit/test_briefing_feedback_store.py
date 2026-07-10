"""D02 — explicit feedback event store tests."""

from __future__ import annotations

import shutil
import sqlite3
from pathlib import Path

import pytest

from research_pipeline.briefing.feedback_store import (
    BriefingFeedbackStore,
    FeedbackStore,
    feedback_target_key,
)
from research_pipeline.briefing.models import FeedbackSignal

FIXTURES = Path(__file__).parents[1] / "fixtures" / "briefing" / "feedback"
EMPTY_FIXTURE = FIXTURES / "feedback_empty.sqlite"


def test_feedback_store_alias_points_to_briefing_store() -> None:
    assert FeedbackStore is BriefingFeedbackStore


def test_empty_fixture_has_expected_schema() -> None:
    assert EMPTY_FIXTURE.exists()
    conn = sqlite3.connect(str(EMPTY_FIXTURE))
    try:
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
    finally:
        conn.close()
    assert {"feedback_events", "preference_adjustments"} <= tables


def test_record_appends_event_only(tmp_path: Path) -> None:
    db = tmp_path / "feedback.db"
    shutil.copyfile(EMPTY_FIXTURE, db)
    store = BriefingFeedbackStore(db)
    try:
        event = store.record(
            target_type="cluster",
            target_id="cluster_abc",
            signal=FeedbackSignal.KEEP,
            reason="actionable",
        )
        events = store.list_feedback()
    finally:
        store.close()

    assert event.feedback_id.startswith("feedback_")
    assert event.target_type == "cluster"
    assert event.signal_type is FeedbackSignal.KEEP
    assert len(events) == 1


def test_record_rejects_malformed_target_id(tmp_path: Path) -> None:
    store = BriefingFeedbackStore(tmp_path / "feedback.db")
    try:
        with pytest.raises(ValueError, match="malformed|non-empty"):
            store.record(
                target_type="cluster",
                target_id="bad id with spaces",
                signal=FeedbackSignal.KEEP,
            )
        with pytest.raises(ValueError, match="target_type"):
            store.record(
                target_type="behavior",
                target_id="abc",
                signal=FeedbackSignal.KEEP,
            )
    finally:
        store.close()


def test_persistence_round_trip(tmp_path: Path) -> None:
    db = tmp_path / "feedback.db"
    first = BriefingFeedbackStore(db)
    try:
        first.record(
            target_type="topic",
            target_id="topic_alpha",
            signal=FeedbackSignal.MORE_LIKE_THIS,
        )
    finally:
        first.close()

    second = BriefingFeedbackStore(db)
    try:
        events = second.list_feedback()
    finally:
        second.close()

    assert len(events) == 1
    assert events[0].target_id == "topic_alpha"


def test_weights_by_target_separates_positive_and_negative(tmp_path: Path) -> None:
    store = BriefingFeedbackStore(tmp_path / "feedback.db")
    try:
        store.record(
            target_type="topic",
            target_id="topic_a",
            signal=FeedbackSignal.MORE_LIKE_THIS,
        )
        store.record(
            target_type="topic",
            target_id="topic_a",
            signal=FeedbackSignal.MORE_LIKE_THIS,
        )
        store.record(
            target_type="source",
            target_id="source_b",
            signal=FeedbackSignal.TOO_NOISY,
        )
        weights = store.weights_by_target()
    finally:
        store.close()

    assert weights[feedback_target_key("topic", "topic_a")] > 0
    assert weights[feedback_target_key("source", "source_b")] < 0


def test_conflict_summary_is_only_targets_with_both(tmp_path: Path) -> None:
    store = BriefingFeedbackStore(tmp_path / "feedback.db")
    try:
        store.record(
            target_type="cluster",
            target_id="c1",
            signal=FeedbackSignal.KEEP,
        )
        store.record(
            target_type="cluster",
            target_id="c1",
            signal=FeedbackSignal.HIDE,
        )
        store.record(
            target_type="cluster",
            target_id="c2",
            signal=FeedbackSignal.KEEP,
        )
        conflicts = store.conflict_summary()
    finally:
        store.close()

    assert "cluster:c1" in conflicts
    assert "cluster:c2" not in conflicts


def test_create_adjustments_respects_min_feedback(tmp_path: Path) -> None:
    store = BriefingFeedbackStore(tmp_path / "feedback.db")
    try:
        for _ in range(2):
            store.record(
                target_type="topic",
                target_id="topic_a",
                signal=FeedbackSignal.KEEP,
            )
        below = store.create_adjustments(min_feedback=3)
        store.record(
            target_type="topic",
            target_id="topic_a",
            signal=FeedbackSignal.KEEP,
        )
        at_threshold = store.create_adjustments(min_feedback=3)
    finally:
        store.close()

    assert below == []
    assert at_threshold and at_threshold[0]["target_type"] == "topic"
    assert at_threshold[0]["after_weight"] > 0
    assert at_threshold[0]["rollback"] == {
        "target": "topic:topic_a",
        "restore_weight": 0.0,
    }


def test_no_behavioral_signals_accepted(tmp_path: Path) -> None:
    """Phase D rule: behavioral signals are not part of FeedbackSignal."""
    store = BriefingFeedbackStore(tmp_path / "feedback.db")
    try:
        for forbidden in ("click", "dwell", "read_time", "scroll"):
            with pytest.raises(ValueError, match="unsupported feedback signal"):
                store.record(
                    target_type="cluster",
                    target_id="c1",
                    signal=forbidden,  # type: ignore[arg-type]
                )
    finally:
        store.close()


def test_create_adjustments_appends_audit_history(tmp_path: Path) -> None:
    """Re-running with the same weights appends rows; it never overwrites (#119)."""
    store = BriefingFeedbackStore(tmp_path / "feedback.db")
    try:
        for _ in range(3):
            store.record(
                target_type="topic",
                target_id="topic_a",
                signal=FeedbackSignal.KEEP,
            )
        first = store.create_adjustments(min_feedback=3)
        assert first
        second = store.create_adjustments(min_feedback=3)
        assert second
        count = store._conn.execute(
            "SELECT COUNT(*) FROM preference_adjustments"
        ).fetchone()[0]
        assert count == len(first) + len(second)
        ids = {
            r[0]
            for r in store._conn.execute(
                "SELECT adjustment_id FROM preference_adjustments"
            )
        }
        assert len(ids) == count  # unique per event, no PK collisions
    finally:
        store.close()
