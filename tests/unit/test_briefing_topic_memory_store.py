from __future__ import annotations

import shutil
from pathlib import Path

import pytest

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


def test_missing_topic_returns_none(tmp_path: Path) -> None:
    store = _store_from_fixture(tmp_path)
    try:
        assert store.get("topic_missing") is None
    finally:
        store.close()


def test_empty_store_has_no_pending_alias_suggestions(tmp_path: Path) -> None:
    store = _store_from_fixture(tmp_path)
    try:
        assert store.list_alias_suggestions() == []
    finally:
        store.close()


def test_repeated_low_novelty_reporting_increases_fatigue_and_is_active(
    tmp_path: Path,
) -> None:
    store = _store_from_fixture(tmp_path)
    try:
        cluster = _cluster("cluster_a", "topic_llm_memory", "LLM Memory")
        first = store.upsert_from_clusters([cluster], "2026-04-01")[0]
        second = store.upsert_from_clusters([cluster], "2026-04-02")[0]

        assert first.status == "new"
        assert second.status == "active"
        assert second.fatigue_score > first.fatigue_score
    finally:
        store.close()


def test_resurfaced_topic_after_dormancy_is_marked_resurfaced(
    tmp_path: Path,
) -> None:
    store = _store_from_fixture(tmp_path)
    try:
        cluster = _cluster("cluster_a", "topic_llm_memory", "LLM Memory")
        store.upsert_from_clusters([cluster], "2026-04-01")
        resurfaced = store.upsert_from_clusters([cluster], "2026-04-20")[0]

        assert resurfaced.status == "resurfaced"
    finally:
        store.close()


def test_ambiguous_duplicate_alias_suggestions_are_not_duplicated(
    tmp_path: Path,
) -> None:
    store = _store_from_fixture(tmp_path)
    try:
        cluster = _cluster("cluster_a", "topic_llm_memory", "LLM Memory")
        store.upsert_from_clusters([cluster], "2026-04-01")

        # Normalized to same canonical name -> ignored.
        assert store.suggest_alias("topic_llm_memory", "  llm   memory ") is None

        first = store.suggest_alias("topic_llm_memory", "Agent Memory")
        second = store.suggest_alias("topic_llm_memory", "Agent Memory")

        assert first is not None
        assert second is not None
        assert len(store.list_alias_suggestions("pending")) == 1
    finally:
        store.close()


def test_alias_approval_without_review_record_is_rejected(tmp_path: Path) -> None:
    store = _store_from_fixture(tmp_path)
    try:
        cluster = _cluster("cluster_a", "topic_llm_memory", "LLM Memory")
        store.upsert_from_clusters([cluster], "2026-04-01")
        suggestion = store.suggest_alias("topic_llm_memory", "Agent Memory")
        assert suggestion is not None

        with pytest.raises(ValueError, match="review_record is required"):
            store.review_alias_suggestion(
                suggestion.suggestion_id,
                approve=True,
                review_record="",
            )

        reloaded = store.list_alias_suggestions("pending")
        assert len(reloaded) == 1
        assert reloaded[0].suggestion_id == suggestion.suggestion_id

        memory = store.get("topic_llm_memory")
        assert memory is not None
        assert "agent memory" not in {alias.lower() for alias in memory.aliases}
    finally:
        store.close()


def test_current_evidence_updates_last_seen_and_canonical_clusters(
    tmp_path: Path,
) -> None:
    store = _store_from_fixture(tmp_path)
    try:
        first_cluster = _cluster("cluster_a", "topic_llm_memory", "LLM Memory")
        second_cluster = _cluster("cluster_b", "topic_llm_memory", "LLM Memory")

        store.upsert_from_clusters([first_cluster], "2026-04-01")
        store.upsert_from_clusters([second_cluster], "2026-04-10")

        memory = store.get("topic_llm_memory")
        assert memory is not None
        assert memory.last_seen_at == "2026-04-10"
        assert set(memory.canonical_clusters) == {"cluster_a", "cluster_b"}
    finally:
        store.close()


# --- issue #119: comma-safe multivalue storage ---


def test_topic_memory_foreign_key_enforced(tmp_path: Path) -> None:
    """An audit row for a non-existent topic is rejected by the FK (#119)."""
    import sqlite3

    from research_pipeline.briefing.topic_memory import TopicMemoryStore

    store = TopicMemoryStore(tmp_path / "topics.db")
    try:
        with pytest.raises(sqlite3.IntegrityError):
            store._conn.execute(
                "INSERT INTO topic_memory_audit (timestamp, topic_id, trigger, "
                "effect, rollback) VALUES (?, ?, ?, ?, ?)",
                ("2026-01-01", "no-such-topic", "t", "e", "r"),
            )
    finally:
        store.close()


def test_topic_memory_schema_version_tracked(tmp_path: Path) -> None:
    from research_pipeline.briefing.topic_memory import (
        _SCHEMA_VERSION,
        TopicMemoryStore,
    )

    store = TopicMemoryStore(tmp_path / "topics.db")
    try:
        version = store._conn.execute("PRAGMA user_version").fetchone()[0]
        assert version == _SCHEMA_VERSION
    finally:
        store.close()


def test_encode_decode_preserves_comma_bearing_values() -> None:
    from research_pipeline.briefing.topic_memory import _decode_list, _encode_list

    values = ("agent, memory", "MCP", "retrieval-augmented, generation")
    assert _decode_list(_encode_list(values)) == values


def test_decode_empty_is_empty_tuple() -> None:
    from research_pipeline.briefing.topic_memory import _decode_list

    assert _decode_list("") == ()


def test_decode_falls_back_to_legacy_comma_format() -> None:
    # Rows written before JSON encoding used comma-join; still readable.
    from research_pipeline.briefing.topic_memory import _decode_list

    assert _decode_list("agent memory,MCP") == ("agent memory", "MCP")
