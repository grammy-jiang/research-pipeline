from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from research_pipeline.briefing.models import (
    AccessMethod,
    BriefingCluster,
    IntelligenceEvent,
    SourceClass,
    TopicAliasSuggestion,
)
from research_pipeline.briefing.topic_memory_store import TopicMemoryStore
from research_pipeline.briefing.topic_review import (
    list_topic_review_queue,
    queue_alias_suggestions,
    review_topic_alias,
)

FIXTURE_EMPTY_DB = (
    Path(__file__).parents[1]
    / "fixtures"
    / "briefing"
    / "memory"
    / "topic_memory_empty.sqlite"
)


def _event(event_id: str, source_id: str) -> IntelligenceEvent:
    return IntelligenceEvent(
        event_id=event_id,
        source_name=source_id,
        source_id=source_id,
        source_type=SourceClass.PRIMARY_ARTIFACT,
        item_type="release",
        canonical_url=f"https://example.com/{event_id}",
        title=f"Event {event_id}",
        retrieved_at="2026-04-29T10:00:00Z",
        collection_method=AccessMethod.API,
        content_hash=f"hash_{event_id}",
        dedup_key=f"dedup_{event_id}",
        published_at="2026-04-29T09:00:00Z",
    )


def _cluster(cluster_id: str, topic_id: str, title: str) -> BriefingCluster:
    event = _event(f"{cluster_id}_e1", f"src_{cluster_id}")
    return BriefingCluster(
        cluster_id=cluster_id,
        title=title,
        primary_event_id=event.event_id,
        event_ids=(event.event_id,),
        topic_ids=(topic_id,),
        canonical_urls=(event.canonical_url,),
        first_seen_at="2026-04-29T09:00:00Z",
        last_seen_at="2026-04-29T09:00:00Z",
        source_classes=(SourceClass.PRIMARY_ARTIFACT,),
        primary_artifact_present=True,
        events=(event,),
    )


def _store_from_fixture(tmp_path: Path) -> TopicMemoryStore:
    db_path = tmp_path / "topic_memory.sqlite"
    shutil.copyfile(FIXTURE_EMPTY_DB, db_path)
    return TopicMemoryStore(db_path)


def _seed_topic(store: TopicMemoryStore) -> None:
    store.upsert_from_clusters(
        [_cluster("cluster_a", "topic_llm_memory", "LLM Memory")],
        "2026-04-01",
    )


def test_empty_store_returns_empty_review_queue(tmp_path: Path) -> None:
    store = _store_from_fixture(tmp_path)
    try:
        assert list_topic_review_queue(store) == []
    finally:
        store.close()


def test_queueing_alias_suggestions_persists_pending_items(tmp_path: Path) -> None:
    store = _store_from_fixture(tmp_path)
    try:
        _seed_topic(store)
        suggestions = [
            TopicAliasSuggestion(
                suggestion_id="alias_suggestion_manual",
                created_at="2026-04-29T10:00:00Z",
                topic_id="topic_llm_memory",
                suggested_alias="agent memory",
                reason="Title differs from canonical topic name.",
            )
        ]

        queued = queue_alias_suggestions(store, suggestions)
        pending = list_topic_review_queue(store)

        assert len(queued) == 1
        assert len(pending) == 1
        assert pending[0].status == "pending"
        assert pending[0].suggested_alias == "agent memory"
    finally:
        store.close()


def test_repeated_alias_submissions_are_deduplicated(tmp_path: Path) -> None:
    store = _store_from_fixture(tmp_path)
    try:
        _seed_topic(store)
        suggestion = TopicAliasSuggestion(
            suggestion_id="alias_suggestion_manual",
            created_at="2026-04-29T10:00:00Z",
            topic_id="topic_llm_memory",
            suggested_alias="agent memory",
            reason="Title differs from canonical topic name.",
        )

        first = queue_alias_suggestions(store, [suggestion])
        second = queue_alias_suggestions(store, [suggestion])
        pending = list_topic_review_queue(store)

        assert len(first) == 1
        assert len(second) == 1
        assert len(pending) == 1
        assert (
            pending[0].suggestion_id
            == first[0].suggestion_id
            == second[0].suggestion_id
        )
    finally:
        store.close()


def test_review_without_review_record_is_rejected(tmp_path: Path) -> None:
    store = _store_from_fixture(tmp_path)
    try:
        _seed_topic(store)
        suggestion = queue_alias_suggestions(
            store,
            [
                TopicAliasSuggestion(
                    suggestion_id="alias_suggestion_manual",
                    created_at="2026-04-29T10:00:00Z",
                    topic_id="topic_llm_memory",
                    suggested_alias="agent memory",
                    reason="Title differs from canonical topic name.",
                )
            ],
        )[0]

        with pytest.raises(ValueError, match="review_record is required"):
            review_topic_alias(
                store,
                suggestion.suggestion_id,
                approve=True,
                review_record="",
            )
    finally:
        store.close()


def test_approving_alias_with_review_record_updates_state(tmp_path: Path) -> None:
    store = _store_from_fixture(tmp_path)
    try:
        _seed_topic(store)
        suggestion = queue_alias_suggestions(
            store,
            [
                TopicAliasSuggestion(
                    suggestion_id="alias_suggestion_manual",
                    created_at="2026-04-29T10:00:00Z",
                    topic_id="topic_llm_memory",
                    suggested_alias="agent memory",
                    reason="Title differs from canonical topic name.",
                )
            ],
        )[0]

        reviewed = review_topic_alias(
            store,
            suggestion.suggestion_id,
            approve=True,
            review_record="B07 approved alias",
        )
        approved = list_topic_review_queue(store, status="approved")
        memory = store.get("topic_llm_memory")

        assert reviewed.status == "approved"
        assert len(approved) == 1
        assert memory is not None
        assert "agent memory" in memory.aliases
    finally:
        store.close()
