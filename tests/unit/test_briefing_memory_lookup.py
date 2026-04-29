from __future__ import annotations

import shutil
from pathlib import Path

from research_pipeline.briefing.memory_lookup import (
    lookup_recent_topic_context,
    suggest_aliases,
)
from research_pipeline.briefing.models import (
    AccessMethod,
    BriefingCluster,
    IntelligenceEvent,
    SourceClass,
)
from research_pipeline.briefing.topic_memory_store import TopicMemoryStore

FIXTURE_EMPTY_DB = (
    Path(__file__).parents[1]
    / "fixtures"
    / "briefing"
    / "memory"
    / "topic_memory_empty.sqlite"
)


def _event(event_id: str, source_id: str, canonical_url: str) -> IntelligenceEvent:
    return IntelligenceEvent(
        event_id=event_id,
        source_name=source_id,
        source_id=source_id,
        source_type=SourceClass.PRIMARY_ARTIFACT,
        item_type="release",
        canonical_url=canonical_url,
        title=f"Event {event_id}",
        retrieved_at="2026-04-29T10:00:00Z",
        collection_method=AccessMethod.API,
        content_hash=f"hash_{event_id}",
        dedup_key=f"dedup_{event_id}",
        published_at="2026-04-29T09:00:00Z",
    )


def _cluster(
    cluster_id: str,
    title: str,
    *,
    topic_ids: tuple[str, ...] = (),
    canonical_url: str | None = None,
) -> BriefingCluster:
    url = canonical_url or f"https://example.com/{cluster_id}"
    event = _event(f"{cluster_id}_e1", f"src_{cluster_id}", url)
    return BriefingCluster(
        cluster_id=cluster_id,
        title=title,
        primary_event_id=event.event_id,
        event_ids=(event.event_id,),
        topic_ids=topic_ids,
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


def test_lookup_returns_empty_when_topic_missing(tmp_path: Path) -> None:
    store = _store_from_fixture(tmp_path)
    try:
        cluster = _cluster("cluster_a", "LLM Memory", topic_ids=("topic_missing",))
        assert lookup_recent_topic_context(store, cluster) == []
    finally:
        store.close()


def test_lookup_returns_empty_for_empty_store(tmp_path: Path) -> None:
    store = _store_from_fixture(tmp_path)
    try:
        cluster = _cluster("cluster_a", "LLM Memory")
        assert lookup_recent_topic_context(store, cluster) == []
    finally:
        store.close()


def test_lookup_matches_by_topic_id(tmp_path: Path) -> None:
    store = _store_from_fixture(tmp_path)
    try:
        seed = _cluster("seed_a", "LLM Memory", topic_ids=("topic_llm_memory",))
        store.upsert_from_clusters([seed], "2026-04-01")

        cluster = _cluster(
            "cluster_a",
            "Different Current Title",
            topic_ids=("topic_llm_memory",),
        )
        memories = lookup_recent_topic_context(store, cluster)

        assert len(memories) == 1
        assert memories[0].topic_id == "topic_llm_memory"
    finally:
        store.close()


def test_lookup_matches_by_normalized_title_or_alias(tmp_path: Path) -> None:
    store = _store_from_fixture(tmp_path)
    try:
        seed = _cluster("seed_a", "LLM Memory", topic_ids=("topic_llm_memory",))
        store.upsert_from_clusters([seed], "2026-04-01")
        suggestion = store.suggest_alias("topic_llm_memory", "Agent Memory")
        assert suggestion is not None
        store.review_alias_suggestion(
            suggestion.suggestion_id,
            approve=True,
            review_record="B03 approved alias",
        )

        cluster = _cluster("cluster_a", "  agent   memory  ")
        memories = lookup_recent_topic_context(store, cluster)

        assert len(memories) == 1
        assert memories[0].topic_id == "topic_llm_memory"
    finally:
        store.close()


def test_lookup_matches_by_canonical_cluster_linkage(tmp_path: Path) -> None:
    store = _store_from_fixture(tmp_path)
    try:
        seed = _cluster("seed_a", "LLM Memory", topic_ids=("topic_llm_memory",))
        store.upsert_from_clusters([seed], "2026-04-01")

        # No topic ID or title match; cluster ID linkage should still resolve.
        cluster = _cluster("seed_a", "Unrelated New Title")
        memories = lookup_recent_topic_context(store, cluster)

        assert len(memories) == 1
        assert memories[0].topic_id == "topic_llm_memory"
    finally:
        store.close()


def test_explicit_topic_id_beats_conflicting_title_match(tmp_path: Path) -> None:
    store = _store_from_fixture(tmp_path)
    try:
        store.upsert_from_clusters(
            [_cluster("seed_a", "Topic Alpha", topic_ids=("topic_alpha",))],
            "2026-04-01",
        )
        store.upsert_from_clusters(
            [_cluster("seed_b", "Topic Beta", topic_ids=("topic_beta",))],
            "2026-04-01",
        )

        cluster = _cluster(
            "cluster_a",
            "Topic Beta",
            topic_ids=("topic_alpha",),
        )
        memories = lookup_recent_topic_context(store, cluster)

        assert [memory.topic_id for memory in memories] == ["topic_alpha"]
    finally:
        store.close()


def test_suggest_aliases_is_reviewable_only_and_does_not_write(tmp_path: Path) -> None:
    store = _store_from_fixture(tmp_path)
    try:
        seed = _cluster("seed_a", "LLM Memory", topic_ids=("topic_llm_memory",))
        store.upsert_from_clusters([seed], "2026-04-01")
        memory = store.get("topic_llm_memory")
        assert memory is not None

        cluster = _cluster("cluster_a", "Agent Memory", topic_ids=("topic_llm_memory",))
        suggestions = suggest_aliases(cluster, [memory])

        assert len(suggestions) == 1
        assert suggestions[0].status == "pending"

        reloaded = store.get("topic_llm_memory")
        assert reloaded is not None
        assert reloaded.aliases == memory.aliases
    finally:
        store.close()
