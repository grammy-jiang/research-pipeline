from __future__ import annotations

from research_pipeline.briefing.models import (
    AccessMethod,
    BriefingCluster,
    IntelligenceEvent,
    SourceClass,
    SourcePolicy,
)
from research_pipeline.briefing.rank import RankingOptions, rank_clusters


def _event(
    event_id: str,
    source_id: str,
    summary_hint: str,
    *,
    published_at: str,
    source_class: SourceClass,
) -> IntelligenceEvent:
    return IntelligenceEvent(
        event_id=event_id,
        source_name=f"Source {source_id}",
        source_id=source_id,
        source_type=source_class,
        source_policy=SourcePolicy.PUBLIC_OFFICIAL,
        item_type="release",
        canonical_url=f"https://example.com/{event_id}",
        title=f"Event {event_id}",
        retrieved_at="2026-06-12T00:00:00Z",
        collection_method=AccessMethod.API,
        content_hash=f"content-{event_id}",
        dedup_key=f"dedup-{event_id}",
        published_at=published_at,
        summary_hint=summary_hint,
    )


def _cluster(
    cluster_id: str,
    title: str,
    source_class: SourceClass,
    summary_hint: str,
    *,
    published_at: str,
    primary_artifact_present: bool,
) -> BriefingCluster:
    event = _event(
        event_id=f"{cluster_id}:e1",
        source_id=f"src-{cluster_id}",
        summary_hint=summary_hint,
        published_at=published_at,
        source_class=source_class,
    )
    return BriefingCluster(
        cluster_id=cluster_id,
        title=title,
        primary_event_id=event.event_id,
        event_ids=(event.event_id,),
        topic_ids=(f"topic-{cluster_id}",),
        canonical_urls=(event.canonical_url,),
        first_seen_at="2026-06-12T00:00:00Z",
        last_seen_at="2026-06-12T00:00:00Z",
        source_classes=(source_class,),
        primary_artifact_present=primary_artifact_present,
        events=(event,),
    )


def test_rank_clusters_empty_input_returns_empty_list() -> None:
    ranked = rank_clusters([])

    assert ranked == []


def test_rank_clusters_higher_score_ranks_first() -> None:
    high_signal = _cluster(
        cluster_id="high",
        title="Alpha",
        source_class=SourceClass.PRIMARY_ARTIFACT,
        summary_hint="Detailed benchmark with reproduction package",
        published_at="2026-06-12T10:00:00Z",
        primary_artifact_present=True,
    )
    lower_signal = _cluster(
        cluster_id="low",
        title="Beta",
        source_class=SourceClass.NEWSLETTER,
        summary_hint="General roundup update",
        published_at="2026-06-12T10:00:00Z",
        primary_artifact_present=True,
    )

    ranked = rank_clusters(
        [lower_signal, high_signal],
        options=RankingOptions(min_rank_score=0.0),
    )

    assert [cluster.cluster_id for cluster in ranked] == ["high", "low"]
    assert ranked[0].rank_score > ranked[1].rank_score


def test_rank_clusters_filters_low_information_clusters() -> None:
    low_information = _cluster(
        cluster_id="noise",
        title="Noise",
        source_class=SourceClass.PRIMARY_ARTIFACT,
        summary_hint="previous release bug fixes",
        published_at="2026-06-12T10:00:00Z",
        primary_artifact_present=True,
    )
    informative = _cluster(
        cluster_id="signal",
        title="Signal",
        source_class=SourceClass.PRIMARY_ARTIFACT,
        summary_hint="Detailed release notes and migration guide available",
        published_at="2026-06-12T11:00:00Z",
        primary_artifact_present=True,
    )

    ranked = rank_clusters(
        [low_information, informative],
        options=RankingOptions(min_rank_score=0.0),
    )

    assert [cluster.cluster_id for cluster in ranked] == ["signal"]


def test_rank_clusters_uses_deterministic_tie_breakers() -> None:
    newer = _cluster(
        cluster_id="newer",
        title="Alpha",
        source_class=SourceClass.PRIMARY_ARTIFACT,
        summary_hint="Detailed reproducibility package",
        published_at="2026-06-12T11:00:00Z",
        primary_artifact_present=True,
    )
    older = _cluster(
        cluster_id="older",
        title="Beta",
        source_class=SourceClass.PRIMARY_ARTIFACT,
        summary_hint="Detailed reproducibility package",
        published_at="2026-06-12T10:00:00Z",
        primary_artifact_present=True,
    )

    ranked_first = rank_clusters(
        [older, newer],
        options=RankingOptions(min_rank_score=0.0),
    )
    ranked_second = rank_clusters(
        [newer, older],
        options=RankingOptions(min_rank_score=0.0),
    )

    assert [cluster.cluster_id for cluster in ranked_first] == ["newer", "older"]
    assert [cluster.cluster_id for cluster in ranked_second] == ["newer", "older"]


def test_rank_clusters_respects_min_rank_score_and_max_items() -> None:
    high = _cluster(
        cluster_id="high",
        title="A",
        source_class=SourceClass.PRIMARY_ARTIFACT,
        summary_hint="Detailed release benchmark",
        published_at="2026-06-12T11:00:00Z",
        primary_artifact_present=True,
    )
    medium = _cluster(
        cluster_id="medium",
        title="B",
        source_class=SourceClass.IMPLEMENTATION_SOURCE,
        summary_hint="Implementation details and examples",
        published_at="2026-06-12T11:00:00Z",
        primary_artifact_present=True,
    )
    low = _cluster(
        cluster_id="low",
        title="C",
        source_class=SourceClass.SOCIAL_SIGNAL,
        summary_hint="interesting launch",
        published_at="2026-06-12T11:00:00Z",
        primary_artifact_present=False,
    )

    ranked = rank_clusters(
        [low, medium, high],
        options=RankingOptions(min_rank_score=4.0, max_items=1),
    )

    assert [cluster.cluster_id for cluster in ranked] == ["high"]
