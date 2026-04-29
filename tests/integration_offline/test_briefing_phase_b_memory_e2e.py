from __future__ import annotations

import shutil
from pathlib import Path

from research_pipeline.briefing.dedup import cluster_events
from research_pipeline.briefing.models import (
    AccessMethod,
    IntelligenceEvent,
    SourceClass,
)
from research_pipeline.briefing.normalize import topic_id_for_title
from research_pipeline.briefing.rank import RankingOptions, rank_clusters
from research_pipeline.briefing.report import render_daily_brief
from research_pipeline.briefing.topic_memory_store import TopicMemoryStore
from research_pipeline.briefing.validate import validate_daily_report
from research_pipeline.briefing.validate_memory import validate_topic_memory

FIXTURE_BASE = Path(__file__).parent.parent / "fixtures" / "briefing" / "e2e"
FIXTURE_EMPTY_DB = (
    Path(__file__).parent.parent
    / "fixtures"
    / "briefing"
    / "memory"
    / "topic_memory_empty.sqlite"
)


def _event(
    event_id: str,
    title: str,
    *,
    source_id: str = "fixture-source",
) -> IntelligenceEvent:
    return IntelligenceEvent(
        event_id=event_id,
        source_name=source_id,
        source_id=source_id,
        source_type=SourceClass.PRIMARY_ARTIFACT,
        item_type="release",
        canonical_url=f"https://example.com/{event_id}",
        title=title,
        retrieved_at="2026-05-01T10:00:00Z",
        collection_method=AccessMethod.API,
        content_hash=f"hash_{event_id}",
        dedup_key=f"dedup_{event_id}",
        published_at="2026-05-01T09:00:00Z",
        topics=(topic_id_for_title(title),),
        summary_hint=(
            f"Detailed engineering update for {title} with concrete changes and "
            "implementation notes."
        ),
        excerpt=(
            f"Detailed engineering update for {title} with concrete changes and "
            "implementation notes."
        ),
    )


def _store_from_fixture(tmp_path: Path) -> TopicMemoryStore:
    db_path = tmp_path / "topic_memory.sqlite"
    shutil.copyfile(FIXTURE_EMPTY_DB, db_path)
    return TopicMemoryStore(db_path)


def _seed_topic(store: TopicMemoryStore, title: str) -> str:
    topic_id = topic_id_for_title(title)
    seed_cluster = cluster_events([_event("seed", title)])[0]
    store.upsert_from_clusters([seed_cluster], "2026-04-01")
    return topic_id


def test_repeated_topic_scenario_passes_offline(tmp_path: Path) -> None:
    scenario_dir = FIXTURE_BASE / "memory_repeated_topic"
    assert scenario_dir.exists()
    store = _store_from_fixture(tmp_path)
    try:
        _seed_topic(store, "LLM Memory")
        store._conn.execute(
            """
            UPDATE topic_memory
            SET fatigue_score = 0.9,
                report_count_7d = 4,
                report_count_30d = 4,
                last_reported_at = ?
            WHERE topic_id = ?
            """,
            ("2026-04-29", topic_id_for_title("LLM Memory")),
        )
        store._conn.commit()

        clusters = cluster_events([_event("repeat", "LLM Memory")])
        ranked = rank_clusters(
            clusters,
            source_weights={"fixture-source": (1.0, 0.1)},
            options=RankingOptions(topic_memory=store),
        )
        markdown = render_daily_brief(
            ranked,
            run_date="2026-05-01",
            topic_memory=store,
        )

        report_result = validate_daily_report(markdown, ranked)
        memory_result = validate_topic_memory(ranked, topic_memory=store)

        assert report_result.passed is True, report_result.errors
        assert memory_result.passed is True, memory_result.errors
        assert ranked[0].novelty_type == "cooling"
    finally:
        store.close()


def test_resurfaced_topic_scenario_passes_offline(tmp_path: Path) -> None:
    scenario_dir = FIXTURE_BASE / "memory_resurfaced_topic"
    assert scenario_dir.exists()
    store = _store_from_fixture(tmp_path)
    try:
        _seed_topic(store, "Dormant Topic")
        store._conn.execute(
            """
            UPDATE topic_memory
            SET status = 'resurfaced',
                fatigue_score = 0.0,
                report_count_7d = 0,
                report_count_30d = 1,
                last_reported_at = ?
            WHERE topic_id = ?
            """,
            ("2026-04-01", topic_id_for_title("Dormant Topic")),
        )
        store._conn.commit()

        clusters = cluster_events([_event("resurfaced", "Dormant Topic")])
        ranked = rank_clusters(
            clusters,
            source_weights={"fixture-source": (1.0, 0.1)},
            options=RankingOptions(topic_memory=store),
        )
        markdown = render_daily_brief(
            ranked,
            run_date="2026-05-01",
            topic_memory=store,
        )

        report_result = validate_daily_report(markdown, ranked)
        memory_result = validate_topic_memory(ranked, topic_memory=store)

        assert report_result.passed is True, report_result.errors
        assert memory_result.passed is True, memory_result.errors
        assert ranked[0].novelty_type == "resurfaced"
    finally:
        store.close()


def test_false_merge_scenario_fails_memory_validation_offline(tmp_path: Path) -> None:
    scenario_dir = FIXTURE_BASE / "memory_false_merge"
    assert scenario_dir.exists()
    store = _store_from_fixture(tmp_path)
    try:
        first_topic_id = _seed_topic(store, "LLM Memory")
        second_topic_id = "topic_agentic_memory"
        seed_cluster = cluster_events([_event("seed-other", "Agentic Systems")])[0]
        seed_cluster = seed_cluster.model_copy(update={"topic_ids": (second_topic_id,)})
        store.upsert_from_clusters([seed_cluster], "2026-04-01")
        suggestion = store.suggest_alias(second_topic_id, "LLM Memory")
        assert suggestion is not None
        store.review_alias_suggestion(
            suggestion.suggestion_id,
            approve=True,
            review_record="approved alias for ambiguity setup",
        )

        clusters = cluster_events([_event("ambiguous", "LLM Memory")])
        clusters = [clusters[0].model_copy(update={"topic_ids": ()})]
        ranked = rank_clusters(
            clusters,
            source_weights={"fixture-source": (1.0, 0.1)},
            options=RankingOptions(topic_memory=store),
        )
        memory_result = validate_topic_memory(ranked, topic_memory=store)

        assert memory_result.passed is False
        assert any(
            "matches multiple topic memories via fallback" in error
            for error in memory_result.errors
        )
        assert first_topic_id != second_topic_id
    finally:
        store.close()
