"""Thin review-queue helpers for topic alias suggestions."""

from __future__ import annotations

from research_pipeline.briefing.models import TopicAliasSuggestion
from research_pipeline.briefing.topic_memory_store import TopicMemoryStore


def queue_alias_suggestions(
    store: TopicMemoryStore,
    suggestions: list[TopicAliasSuggestion],
) -> list[TopicAliasSuggestion]:
    """Persist reviewable alias suggestions into the durable queue."""
    queued: list[TopicAliasSuggestion] = []
    for suggestion in suggestions:
        persisted = store.suggest_alias(
            suggestion.topic_id,
            suggestion.suggested_alias,
            reason=suggestion.reason,
        )
        queued.append(persisted or _existing_suggestion(store, suggestion))
    return queued


def list_topic_review_queue(
    store: TopicMemoryStore,
    status: str | None = "pending",
) -> list[TopicAliasSuggestion]:
    """List queued topic-review items in deterministic store order."""
    return store.list_alias_suggestions(status=status)


def review_topic_alias(
    store: TopicMemoryStore,
    suggestion_id: str,
    *,
    approve: bool,
    review_record: str,
) -> TopicAliasSuggestion:
    """Approve or reject a queued alias suggestion with explicit review metadata."""
    return store.review_alias_suggestion(
        suggestion_id,
        approve=approve,
        review_record=review_record,
    )


def _existing_suggestion(
    store: TopicMemoryStore,
    suggestion: TopicAliasSuggestion,
) -> TopicAliasSuggestion:
    for queued in store.list_alias_suggestions(status=None):
        if queued.suggestion_id == suggestion.suggestion_id:
            return queued
    raise ValueError(
        f"alias suggestion not found after queueing: {suggestion.suggestion_id}"
    )
