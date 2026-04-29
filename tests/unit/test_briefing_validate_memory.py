from __future__ import annotations

import shutil
from pathlib import Path

from research_pipeline.briefing.models import (
    AccessMethod,
    BriefingCluster,
    IntelligenceEvent,
    SourceClass,
)
from research_pipeline.briefing.normalize import topic_id_for_title
from research_pipeline.briefing.topic_memory_store import TopicMemoryStore
from research_pipeline.briefing.validate_memory import validate_topic_memory

FIXTURE_EMPTY_DB = (
    Path(__file__).parents[1]
    / "fixtures"
    / "briefing"
    / "memory"
    / "topic_memory_empty.sqlite"
)


def _event(
    event_id: str,
    title: str,
    *,
    topic_id: str | None = None,
) -> IntelligenceEvent:
    return IntelligenceEvent(
        event_id=event_id,
        source_name="fixture-source",
        source_id="fixture-source",
        source_type=SourceClass.PRIMARY_ARTIFACT,
        item_type="release",
        canonical_url=f"https://example.com/{event_id}",
        title=title,
        retrieved_at="2026-04-29T10:00:00Z",
        collection_method=AccessMethod.API,
        content_hash=f"hash_{event_id}",
        dedup_key=f"dedup_{event_id}",
        published_at="2026-04-29T09:00:00Z",
        topics=((topic_id,) if topic_id else ()),
    )


def _cluster(
    cluster_id: str,
    title: str,
    *,
    topic_id: str | None = None,
) -> BriefingCluster:
    event = _event(f"{cluster_id}_event", title, topic_id=topic_id)
    return BriefingCluster(
        cluster_id=cluster_id,
        title=title,
        primary_event_id=event.event_id,
        event_ids=(event.event_id,),
        topic_ids=((topic_id,) if topic_id else ()),
        canonical_urls=(event.canonical_url,),
        first_seen_at="2026-04-29T09:00:00Z",
        last_seen_at="2026-04-29T09:00:00Z",
        source_classes=(SourceClass.PRIMARY_ARTIFACT,),
        primary_artifact_present=True,
        novelty_type="new",
        events=(event,),
    )


def _store_from_fixture(tmp_path: Path) -> TopicMemoryStore:
    db_path = tmp_path / "topic_memory.sqlite"
    shutil.copyfile(FIXTURE_EMPTY_DB, db_path)
    return TopicMemoryStore(db_path)


def _seed_memory(store: TopicMemoryStore, title: str) -> str:
    topic_id = topic_id_for_title(title)
    store.upsert_from_clusters(
        [_cluster("seed", title, topic_id=topic_id)],
        "2026-04-01",
    )
    return topic_id


def test_validate_memory_allows_new_clusters_with_empty_store(tmp_path: Path) -> None:
    store = _store_from_fixture(tmp_path)
    try:
        cluster = _cluster("fresh", "Fresh Topic")

        result = validate_topic_memory([cluster], topic_memory=store)

        assert result.passed is True
        assert result.errors == ()
    finally:
        store.close()


def test_validate_memory_requires_matching_record_for_cooling_cluster(
    tmp_path: Path,
) -> None:
    store = _store_from_fixture(tmp_path)
    try:
        cluster = _cluster("cooling", "Cooling Topic")
        cluster = cluster.model_copy(
            update={"novelty_type": "cooling", "fatigue_penalty": 0.8}
        )

        result = validate_topic_memory([cluster], topic_memory=store)

        assert result.passed is False
        assert any("no matching topic memory" in error for error in result.errors)
    finally:
        store.close()


def test_validate_memory_requires_resurfaced_prior_state(tmp_path: Path) -> None:
    store = _store_from_fixture(tmp_path)
    try:
        topic_id = _seed_memory(store, "Dormant Topic")
        store._conn.execute(
            "UPDATE topic_memory SET last_reported_at = ? WHERE topic_id = ?",
            ("2026-04-25", topic_id),
        )
        store._conn.commit()
        cluster = _cluster("resurfaced", "Dormant Topic", topic_id=topic_id)
        cluster = cluster.model_copy(update={"novelty_type": "resurfaced"})

        result = validate_topic_memory([cluster], topic_memory=store)

        assert result.passed is False
        assert any(
            "marked resurfaced without dormant prior memory" in error
            for error in result.errors
        )
    finally:
        store.close()


def test_validate_memory_detects_ambiguous_fallback_match(tmp_path: Path) -> None:
    store = _store_from_fixture(tmp_path)
    try:
        first_topic_id = _seed_memory(store, "LLM Memory")
        second_topic_id = "topic_other_memory"
        store.upsert_from_clusters(
            [_cluster("seed_other", "Other Memory", topic_id=second_topic_id)],
            "2026-04-01",
        )
        suggestion = store.suggest_alias(second_topic_id, "LLM Memory")
        assert suggestion is not None
        store.review_alias_suggestion(
            suggestion.suggestion_id,
            approve=True,
            review_record="approved alias for ambiguity setup",
        )
        cluster = _cluster("ambiguous", "LLM Memory")

        result = validate_topic_memory([cluster], topic_memory=store)

        assert result.passed is False
        assert any(
            "matches multiple topic memories via fallback" in error
            for error in result.errors
        )
        assert first_topic_id != second_topic_id
    finally:
        store.close()


def test_validate_memory_detects_invalid_review_queue_rows(tmp_path: Path) -> None:
    store = _store_from_fixture(tmp_path)
    try:
        topic_id = _seed_memory(store, "LLM Memory")
        store._conn.execute(
            """
            INSERT INTO topic_alias_suggestions (
                suggestion_id, created_at, topic_id, suggested_alias,
                reason, status, review_record
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "alias_suggestion_bad",
                "2026-04-29T10:00:00Z",
                topic_id,
                "agent memory",
                "invalid approved row",
                "approved",
                "",
            ),
        )
        store._conn.commit()

        result = validate_topic_memory([], topic_memory=store)

        assert result.passed is False
        assert any("invalid alias review queue" in error for error in result.errors)
    finally:
        store.close()
