from __future__ import annotations

import shutil
from pathlib import Path

from research_pipeline.briefing.models import (
    AccessMethod,
    BriefingCluster,
    IntelligenceEvent,
    SourceClass,
)
from research_pipeline.briefing.rank import RankingOptions, rank_clusters
from research_pipeline.briefing.topic_memory_store import TopicMemoryStore

FIXTURE_EMPTY_DB = (
    Path(__file__).parents[1]
    / "fixtures"
    / "briefing"
    / "memory"
    / "topic_memory_empty.sqlite"
)


def _event(
    event_id: str,
    source_id: str,
    *,
    source_class: SourceClass,
    summary_hint: str,
) -> IntelligenceEvent:
    return IntelligenceEvent(
        event_id=event_id,
        source_name=source_id,
        source_id=source_id,
        source_type=source_class,
        item_type="release",
        canonical_url=f"https://example.com/{event_id}",
        title=f"Event {event_id}",
        retrieved_at="2026-04-29T10:00:00Z",
        collection_method=AccessMethod.API,
        content_hash=f"hash_{event_id}",
        dedup_key=f"dedup_{event_id}",
        published_at="2026-04-29T09:00:00Z",
        summary_hint=summary_hint,
    )


def _cluster(
    cluster_id: str,
    title: str,
    *,
    topic_ids: tuple[str, ...] = (),
    source_class: SourceClass = SourceClass.PRIMARY_ARTIFACT,
    primary_artifact_present: bool = True,
    summary_hint: str = "Detailed benchmark with implementation notes",
) -> BriefingCluster:
    event = _event(
        f"{cluster_id}_e1",
        f"src_{cluster_id}",
        source_class=source_class,
        summary_hint=summary_hint,
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
        source_classes=(source_class,),
        primary_artifact_present=primary_artifact_present,
        events=(event,),
    )


def _store_from_fixture(tmp_path: Path) -> TopicMemoryStore:
    db_path = tmp_path / "topic_memory.sqlite"
    shutil.copyfile(FIXTURE_EMPTY_DB, db_path)
    return TopicMemoryStore(db_path)


def test_clusters_without_memory_keep_zero_adjustments() -> None:
    cluster = _cluster("cluster_a", "Fresh Topic")

    ranked = rank_clusters([cluster], options=RankingOptions(min_rank_score=0.0))

    assert len(ranked) == 1
    assert ranked[0].fatigue_penalty == 0.0
    assert ranked[0].resurfaced_boost == 0.0
    assert ranked[0].novelty_type == "new"


def test_repeated_low_novelty_topic_ranks_below_equivalent_fresh_topic(
    tmp_path: Path,
) -> None:
    store = _store_from_fixture(tmp_path)
    try:
        repeated_seed = _cluster(
            "seed_a",
            "Repeated Topic",
            topic_ids=("topic_repeated",),
        )
        store.upsert_from_clusters([repeated_seed], "2026-04-01")
        store.upsert_from_clusters([repeated_seed], "2026-04-02")
        store.upsert_from_clusters([repeated_seed], "2026-04-03")

        repeated = _cluster(
            "cluster_a",
            "Repeated Topic",
            topic_ids=("topic_repeated",),
        )
        fresh = _cluster("cluster_b", "Fresh Topic", topic_ids=("topic_fresh",))

        ranked = rank_clusters(
            [repeated, fresh],
            options=RankingOptions(min_rank_score=0.0, topic_memory=store),
        )

        assert [cluster.cluster_id for cluster in ranked] == ["cluster_b", "cluster_a"]
        assert ranked[0].fatigue_penalty == 0.0
        assert ranked[1].fatigue_penalty > 0.0
        assert ranked[1].novelty_type == "cooling"
    finally:
        store.close()


def test_resurfaced_topic_gets_positive_boost_and_resurfaced_novelty(
    tmp_path: Path,
) -> None:
    store = _store_from_fixture(tmp_path)
    try:
        seed = _cluster("seed_a", "Resurfaced Topic", topic_ids=("topic_resurfaced",))
        store.upsert_from_clusters([seed], "2026-04-01")
        store.upsert_from_clusters([seed], "2026-04-20")

        cluster = _cluster(
            "cluster_a",
            "Resurfaced Topic",
            topic_ids=("topic_resurfaced",),
        )
        ranked = rank_clusters(
            [cluster],
            options=RankingOptions(min_rank_score=0.0, topic_memory=store),
        )

        assert len(ranked) == 1
        assert ranked[0].resurfaced_boost > 0.0
        assert ranked[0].novelty_type == "resurfaced"
    finally:
        store.close()


def test_strong_resurfaced_primary_evidence_is_not_filtered_by_fatigue(
    tmp_path: Path,
) -> None:
    store = _store_from_fixture(tmp_path)
    try:
        seed = _cluster(
            "seed_a",
            "High Signal Topic",
            topic_ids=("topic_high_signal",),
        )
        store.upsert_from_clusters([seed], "2026-04-01")
        store.upsert_from_clusters([seed], "2026-04-02")
        store.upsert_from_clusters([seed], "2026-04-03")
        store.upsert_from_clusters([seed], "2026-04-04")
        store.upsert_from_clusters([seed], "2026-04-20")

        cluster = _cluster(
            "cluster_a",
            "High Signal Topic",
            topic_ids=("topic_high_signal",),
            source_class=SourceClass.PRIMARY_ARTIFACT,
            primary_artifact_present=True,
            summary_hint="Detailed benchmark, migration guide, and release notes",
        )
        ranked = rank_clusters(
            [cluster],
            options=RankingOptions(topic_memory=store),
        )

        assert len(ranked) == 1
        assert ranked[0].novelty_type == "resurfaced"
        assert ranked[0].rank_score >= 4.0
    finally:
        store.close()
