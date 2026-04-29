"""D06 — explicit feedback adjustments to ranking."""

from __future__ import annotations

from research_pipeline.briefing.models import (
    AccessMethod,
    BriefingCluster,
    IntelligenceEvent,
    SourceClass,
    SourcePolicy,
)
from research_pipeline.briefing.rank import (
    RankingOptions,
    explicit_feedback_components,
    rank_clusters,
)


def _event(event_id: str, source_id: str) -> IntelligenceEvent:
    return IntelligenceEvent(
        event_id=event_id,
        source_name=f"Source {source_id}",
        source_id=source_id,
        source_type=SourceClass.PRIMARY_ARTIFACT,
        source_policy=SourcePolicy.PUBLIC_OFFICIAL,
        item_type="release",
        canonical_url=f"https://example.com/{event_id}",
        title=f"Event {event_id}",
        retrieved_at="2026-06-12T00:00:00Z",
        collection_method=AccessMethod.API,
        content_hash=f"content-{event_id}",
        dedup_key=f"dedup-{event_id}",
        published_at="2026-06-12T10:00:00Z",
        summary_hint="Detailed benchmark with reproduction package available",
    )


def _cluster(cluster_id: str, *, topic_id: str, source_id: str) -> BriefingCluster:
    event = _event(f"{cluster_id}:e1", source_id)
    return BriefingCluster(
        cluster_id=cluster_id,
        title=f"Cluster {cluster_id}",
        primary_event_id=event.event_id,
        event_ids=(event.event_id,),
        topic_ids=(topic_id,),
        canonical_urls=(event.canonical_url,),
        first_seen_at="2026-06-12T00:00:00Z",
        last_seen_at="2026-06-12T00:00:00Z",
        source_classes=(SourceClass.PRIMARY_ARTIFACT,),
        primary_artifact_present=True,
        events=(event,),
    )


def test_no_feedback_baseline_components_are_zero() -> None:
    cluster = _cluster("c1", topic_id="topic_a", source_id="src_a")
    topic, source, neg = explicit_feedback_components(cluster, {})
    assert (topic, source, neg) == (0.0, 0.0, 0.0)


def test_positive_topic_feedback_boosts_rank() -> None:
    cluster = _cluster("c1", topic_id="topic_a", source_id="src_a")
    baseline = rank_clusters([cluster], options=RankingOptions(min_rank_score=0.0))[0]
    boosted = rank_clusters(
        [cluster],
        options=RankingOptions(
            min_rank_score=0.0, feedback_weights={"topic:topic_a": 0.5}
        ),
    )[0]
    assert boosted.rank_score > baseline.rank_score
    topic, _src, neg = explicit_feedback_components(cluster, {"topic:topic_a": 0.5})
    assert topic == 0.5
    assert neg == 0.0


def test_positive_source_feedback_boosts_rank() -> None:
    cluster = _cluster("c1", topic_id="topic_a", source_id="src_a")
    boosted = rank_clusters(
        [cluster],
        options=RankingOptions(
            min_rank_score=0.0, feedback_weights={"source:src_a": 0.4}
        ),
    )[0]
    _t, source, neg = explicit_feedback_components(cluster, {"source:src_a": 0.4})
    assert source == 0.4
    assert neg == 0.0
    assert boosted.rank_score > 0.0


def test_negative_feedback_penalises_rank() -> None:
    cluster = _cluster("c1", topic_id="topic_a", source_id="src_a")
    baseline = rank_clusters([cluster], options=RankingOptions(min_rank_score=0.0))[0]
    penalised = rank_clusters(
        [cluster],
        options=RankingOptions(
            min_rank_score=0.0, feedback_weights={"topic:topic_a": -0.7}
        ),
    )[0]
    assert penalised.rank_score < baseline.rank_score
    topic, _src, neg = explicit_feedback_components(cluster, {"topic:topic_a": -0.7})
    assert topic == 0.0
    assert neg == 0.7


def test_conflicting_aggregate_cancels_to_baseline() -> None:
    # Topic-level positive cancelled by equally large negative through the
    # ranking score (sum of bonuses = 0).
    cluster = _cluster("c1", topic_id="topic_a", source_id="src_a")
    baseline = rank_clusters([cluster], options=RankingOptions(min_rank_score=0.0))[0]
    cancelled = rank_clusters(
        [cluster],
        options=RankingOptions(
            min_rank_score=0.0,
            feedback_weights={"topic:topic_a": 0.5, "source:src_a": -0.5},
        ),
    )[0]
    assert cancelled.rank_score == baseline.rank_score


def test_explicit_components_match_phase_d_formula() -> None:
    cluster = _cluster("c1", topic_id="topic_a", source_id="src_a")
    weights = {"topic:topic_a": 0.6, "source:src_a": -0.2}
    topic, source, neg = explicit_feedback_components(cluster, weights)
    # rank_score(with) - rank_score(without) == topic + source - neg
    baseline = rank_clusters([cluster], options=RankingOptions(min_rank_score=0.0))[
        0
    ].rank_score
    adjusted = rank_clusters(
        [cluster],
        options=RankingOptions(min_rank_score=0.0, feedback_weights=weights),
    )[0].rank_score
    assert round(adjusted - baseline, 6) == round(topic + source - neg, 6)


def test_deterministic_ordering_under_feedback() -> None:
    a = _cluster("a", topic_id="topic_x", source_id="src_a")
    b = _cluster("b", topic_id="topic_y", source_id="src_b")
    weights = {"topic:topic_x": 0.5}
    out1 = rank_clusters(
        [a, b], options=RankingOptions(min_rank_score=0.0, feedback_weights=weights)
    )
    out2 = rank_clusters(
        [b, a], options=RankingOptions(min_rank_score=0.0, feedback_weights=weights)
    )
    assert [c.cluster_id for c in out1] == [c.cluster_id for c in out2]
    assert out1[0].cluster_id == "a"
