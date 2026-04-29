"""Shared dossier test fixtures (Phase E)."""

from __future__ import annotations

from research_pipeline.briefing.models import (
    AccessMethod,
    BriefingCluster,
    IntelligenceEvent,
    SourceClass,
    SourcePolicy,
)


def make_event(
    event_id: str = "evt1",
    source_class: SourceClass = SourceClass.PRIMARY_ARTIFACT,
    *,
    canonical_url: str = "https://example.com/release-notes",
    title: str = "Release notes for v1.0",
    summary_hint: str = "First release with streaming output.",
    published_at: str | None = "2026-04-29",
    retrieved_at: str = "2026-04-29T00:00:00Z",
) -> IntelligenceEvent:
    return IntelligenceEvent(
        event_id=event_id,
        source_name="Acme Lab",
        source_id="acme",
        source_type=source_class,
        source_policy=SourcePolicy.PUBLIC_OFFICIAL,
        item_type="release",
        canonical_url=canonical_url,
        title=title,
        retrieved_at=retrieved_at,
        collection_method=AccessMethod.API,
        content_hash=f"hash-{event_id}",
        dedup_key=f"dk-{event_id}",
        published_at=published_at,
        summary_hint=summary_hint,
        confidence="high",
    )


def make_cluster(
    *,
    cluster_id: str = "cluster_a",
    title: str = "Acme v1.0 release",
    topic_ids: tuple[str, ...] = ("topic_acme",),
    primary_artifact_present: bool = True,
    canonical_urls: tuple[str, ...] | None = None,
    events: tuple[IntelligenceEvent, ...] | None = None,
    rank_score: float = 0.9,
    evidence_type: str = "supported_fact",
) -> BriefingCluster:
    if events is None:
        events = (make_event(),)
    if canonical_urls is None:
        canonical_urls = tuple(str(e.canonical_url) for e in events)
    return BriefingCluster(
        cluster_id=cluster_id,
        title=title,
        primary_event_id=events[0].event_id,
        event_ids=tuple(e.event_id for e in events),
        topic_ids=topic_ids,
        canonical_urls=canonical_urls,
        first_seen_at="2026-04-29T00:00:00Z",
        last_seen_at="2026-04-29T00:00:00Z",
        source_classes=tuple({e.source_type for e in events}),
        primary_artifact_present=primary_artifact_present,
        evidence_type=evidence_type,
        ranking_explanation="class=3.00; trust=1.00",
        rank_score=rank_score,
        events=events,
    )
