from __future__ import annotations

import shutil
from pathlib import Path

from research_pipeline.briefing.lifecycle import classify_cluster_lifecycle
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
    primary_artifact_present: bool = True,
) -> BriefingCluster:
    event = _event(
        f"{cluster_id}_e1",
        f"src_{cluster_id}",
        f"https://example.com/{cluster_id}",
    )
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
        primary_artifact_present=primary_artifact_present,
        events=(event,),
    )


def _store_from_fixture(tmp_path: Path) -> TopicMemoryStore:
    db_path = tmp_path / "topic_memory.sqlite"
    shutil.copyfile(FIXTURE_EMPTY_DB, db_path)
    return TopicMemoryStore(db_path)


def test_missing_memory_classifies_new(tmp_path: Path) -> None:
    store = _store_from_fixture(tmp_path)
    try:
        cluster = _cluster("cluster_a", "LLM Memory", topic_ids=("topic_missing",))
        assert classify_cluster_lifecycle(store, cluster, "2026-04-29") == "new"
    finally:
        store.close()


def test_empty_store_classifies_new(tmp_path: Path) -> None:
    store = _store_from_fixture(tmp_path)
    try:
        cluster = _cluster("cluster_a", "LLM Memory")
        assert classify_cluster_lifecycle(store, cluster, "2026-04-29") == "new"
    finally:
        store.close()


def test_repeated_low_novelty_classifies_cooling(tmp_path: Path) -> None:
    store = _store_from_fixture(tmp_path)
    try:
        seed = _cluster("seed_a", "LLM Memory", topic_ids=("topic_llm_memory",))
        store.upsert_from_clusters([seed], "2026-04-01")
        store.upsert_from_clusters([seed], "2026-04-02")
        store.upsert_from_clusters([seed], "2026-04-03")

        current = _cluster("cluster_a", "LLM Memory", topic_ids=("topic_llm_memory",))
        assert classify_cluster_lifecycle(store, current, "2026-04-03") == "cooling"
    finally:
        store.close()


def test_resurfaced_topic_classifies_resurfaced(tmp_path: Path) -> None:
    store = _store_from_fixture(tmp_path)
    try:
        seed = _cluster("seed_a", "LLM Memory", topic_ids=("topic_llm_memory",))
        store.upsert_from_clusters([seed], "2026-04-01")

        current = _cluster("cluster_a", "LLM Memory", topic_ids=("topic_llm_memory",))
        assert classify_cluster_lifecycle(store, current, "2026-04-20") == "resurfaced"
    finally:
        store.close()


def test_old_topic_without_strong_evidence_is_dormant(tmp_path: Path) -> None:
    store = _store_from_fixture(tmp_path)
    try:
        seed = _cluster("seed_a", "LLM Memory", topic_ids=("topic_llm_memory",))
        store.upsert_from_clusters([seed], "2026-04-01")

        current = _cluster(
            "cluster_a",
            "LLM Memory",
            topic_ids=("topic_llm_memory",),
            primary_artifact_present=False,
        )
        assert classify_cluster_lifecycle(store, current, "2026-05-15") == "dormant"
    finally:
        store.close()


def test_current_explicit_evidence_overrides_stale_title_only_memory(
    tmp_path: Path,
) -> None:
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

        current = _cluster("cluster_a", "Topic Beta", topic_ids=("topic_alpha",))
        status = classify_cluster_lifecycle(store, current, "2026-04-02")

        assert status == "active"
    finally:
        store.close()


def test_lifecycle_classification_is_read_only(tmp_path: Path) -> None:
    store = _store_from_fixture(tmp_path)
    try:
        seed = _cluster("seed_a", "LLM Memory", topic_ids=("topic_llm_memory",))
        store.upsert_from_clusters([seed], "2026-04-01")

        before_alias_count = len(store.list_alias_suggestions())
        before_memory = store.get("topic_llm_memory")
        assert before_memory is not None

        current = _cluster("cluster_a", "Agent Memory", topic_ids=("topic_llm_memory",))
        _ = classify_cluster_lifecycle(store, current, "2026-04-02")

        after_alias_count = len(store.list_alias_suggestions())
        after_memory = store.get("topic_llm_memory")
        assert after_memory is not None

        assert after_alias_count == before_alias_count
        assert after_memory == before_memory
    finally:
        store.close()
