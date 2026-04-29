"""D04 — manual review → explicit feedback tests."""

from __future__ import annotations

from pathlib import Path

from research_pipeline.briefing.feedback_store import BriefingFeedbackStore
from research_pipeline.briefing.manual_review import (
    import_reviewed_suggestions,
    record_review_as_feedback,
)
from research_pipeline.briefing.models import FeedbackSignal, TopicAliasSuggestion


def _suggestion(
    *,
    suggestion_id: str = "sugg_1",
    topic_id: str = "topic_alpha",
    status: str = "approved",
    review_record: str = "approved-by-user",
) -> TopicAliasSuggestion:
    return TopicAliasSuggestion(
        suggestion_id=suggestion_id,
        created_at="2026-04-27T00:00:00Z",
        topic_id=topic_id,
        suggested_alias="alpha-beta",
        reason="merge candidate",
        status=status,  # type: ignore[arg-type]
        review_record=review_record,
    )


def test_approved_suggestion_records_more_like_this(tmp_path: Path) -> None:
    store = BriefingFeedbackStore(tmp_path / "feedback.db")
    try:
        event = record_review_as_feedback(store, _suggestion(status="approved"))
        events = store.list_feedback()
    finally:
        store.close()
    assert event is not None
    assert event.signal_type is FeedbackSignal.MORE_LIKE_THIS
    assert event.target_type == "topic"
    assert event.target_id == "topic_alpha"
    assert event.reason == "approved-by-user"
    assert events == [event]


def test_rejected_suggestion_records_less_like_this(tmp_path: Path) -> None:
    store = BriefingFeedbackStore(tmp_path / "feedback.db")
    try:
        event = record_review_as_feedback(
            store,
            _suggestion(status="rejected", review_record="duplicate"),
        )
    finally:
        store.close()
    assert event is not None
    assert event.signal_type is FeedbackSignal.LESS_LIKE_THIS


def test_pending_suggestion_is_skipped(tmp_path: Path) -> None:
    store = BriefingFeedbackStore(tmp_path / "feedback.db")
    try:
        # Pending suggestion: review_record may be empty.
        suggestion = TopicAliasSuggestion(
            suggestion_id="sugg_pending",
            created_at="2026-04-27T00:00:00Z",
            topic_id="topic_alpha",
            suggested_alias="alpha-beta",
            reason="merge candidate",
        )
        result = record_review_as_feedback(store, suggestion)
        assert result is None
        assert store.list_feedback() == []
    finally:
        store.close()


def test_import_reviewed_only_appends_reviewed(tmp_path: Path) -> None:
    pending = TopicAliasSuggestion(
        suggestion_id="sugg_pending",
        created_at="2026-04-27T00:00:00Z",
        topic_id="topic_alpha",
        suggested_alias="alpha-beta",
        reason="merge candidate",
    )
    approved = _suggestion(suggestion_id="sugg_a", status="approved")
    rejected = _suggestion(
        suggestion_id="sugg_r",
        topic_id="topic_beta",
        status="rejected",
        review_record="not-a-merge",
    )

    store = BriefingFeedbackStore(tmp_path / "feedback.db")
    try:
        events = import_reviewed_suggestions(store, [pending, approved, rejected])
    finally:
        store.close()

    assert {event.target_id for event in events} == {"topic_alpha", "topic_beta"}
    assert {event.signal_type for event in events} == {
        FeedbackSignal.MORE_LIKE_THIS,
        FeedbackSignal.LESS_LIKE_THIS,
    }
