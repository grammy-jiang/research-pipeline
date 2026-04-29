"""D04 — convert manual review labels into explicit feedback events.

Phase B introduced reviewable :class:`TopicAliasSuggestion` records.
Phase D treats an *approved* suggestion as a positive
``MORE_LIKE_THIS`` signal on the suggestion's topic and a *rejected*
suggestion as a negative ``LESS_LIKE_THIS`` signal — both carry the
explicit ``review_record`` as the feedback ``reason``.

This module is read-only on suggestions: it never mutates them.  It
appends feedback events through
:class:`research_pipeline.briefing.feedback_store.BriefingFeedbackStore`.
Pending suggestions and unmodified review records are skipped.
"""

from __future__ import annotations

from collections.abc import Iterable

from research_pipeline.briefing.feedback_store import BriefingFeedbackStore
from research_pipeline.briefing.models import (
    FeedbackEvent,
    FeedbackSignal,
    TopicAliasSuggestion,
)

# Mapping from a reviewed suggestion status to an explicit feedback signal.
_STATUS_TO_SIGNAL: dict[str, FeedbackSignal] = {
    "approved": FeedbackSignal.MORE_LIKE_THIS,
    "rejected": FeedbackSignal.LESS_LIKE_THIS,
}


def record_review_as_feedback(
    store: BriefingFeedbackStore,
    suggestion: TopicAliasSuggestion,
) -> FeedbackEvent | None:
    """Append a feedback event for a reviewed suggestion.

    Returns the appended :class:`FeedbackEvent`, or ``None`` if the
    suggestion is still pending (no review record exists).  Raises
    :class:`ValueError` if the suggestion's status is not one of
    ``approved``/``rejected``/``pending`` (defensive — the model already
    enforces this).
    """
    if suggestion.status == "pending":
        return None
    try:
        signal = _STATUS_TO_SIGNAL[suggestion.status]
    except KeyError as exc:
        raise ValueError(
            f"unsupported suggestion status: {suggestion.status!r}"
        ) from exc
    return store.record(
        target_type="topic",
        target_id=suggestion.topic_id,
        signal=signal,
        reason=suggestion.review_record,
        context={
            "suggestion_id": suggestion.suggestion_id,
            "suggested_alias": suggestion.suggested_alias,
        },
    )


def import_reviewed_suggestions(
    store: BriefingFeedbackStore,
    suggestions: Iterable[TopicAliasSuggestion],
) -> list[FeedbackEvent]:
    """Append feedback for every reviewed (non-pending) suggestion."""
    events: list[FeedbackEvent] = []
    for suggestion in suggestions:
        event = record_review_as_feedback(store, suggestion)
        if event is not None:
            events.append(event)
    return events
