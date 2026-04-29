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
from research_pipeline.briefing.report import render_daily_brief
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
    source_class: SourceClass = SourceClass.PRIMARY_ARTIFACT,
    summary_hint: str = "Detailed benchmark with implementation notes",
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
) -> BriefingCluster:
    event = _event(
        f"{cluster_id}_e1",
        f"src_{cluster_id}",
        source_class=source_class,
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
        primary_artifact_present=True,
        suggested_action="read",
        ranking_explanation="class=3.00; trust=1.00",
        events=(event,),
    )


def _store_from_fixture(tmp_path: Path) -> TopicMemoryStore:
    db_path = tmp_path / "topic_memory.sqlite"
    shutil.copyfile(FIXTURE_EMPTY_DB, db_path)
    return TopicMemoryStore(db_path)


def test_report_without_matching_memory_omits_prior_context() -> None:
    cluster = _cluster("cluster_a", "Fresh Topic")

    markdown = render_daily_brief([cluster], run_date="2026-04-29")

    assert "**Prior context:**" not in markdown
    assert "## Prior Context" not in markdown


def test_resurfaced_topic_renders_concise_prior_context(tmp_path: Path) -> None:
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

        markdown = render_daily_brief(
            ranked,
            run_date="2026-04-29",
            topic_memory=store,
        )

        assert "**Prior context:** resurfaced topic" in markdown
        assert "2026-04-20" in markdown
        assert "## Prior Context" not in markdown
    finally:
        store.close()


def test_cooling_topic_renders_concise_prior_context_with_fatigue(
    tmp_path: Path,
) -> None:
    store = _store_from_fixture(tmp_path)
    try:
        seed = _cluster("seed_a", "Repeated Topic", topic_ids=("topic_repeated",))
        store.upsert_from_clusters([seed], "2026-04-01")
        store.upsert_from_clusters([seed], "2026-04-02")
        store.upsert_from_clusters([seed], "2026-04-03")

        cluster = _cluster(
            "cluster_a",
            "Repeated Topic",
            topic_ids=("topic_repeated",),
        )
        ranked = rank_clusters(
            [cluster],
            options=RankingOptions(min_rank_score=0.0, topic_memory=store),
        )

        markdown = render_daily_brief(
            ranked,
            run_date="2026-04-29",
            topic_memory=store,
        )

        assert "**Prior context:** repeated topic" in markdown
        assert "fatigue=" in markdown
        assert "## Prior Context" not in markdown
    finally:
        store.close()
