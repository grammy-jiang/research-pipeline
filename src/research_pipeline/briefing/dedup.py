"""Exact deduplication and clustering for briefing events."""

from __future__ import annotations

from collections import defaultdict

from research_pipeline.briefing.models import (
    BriefingCluster,
    IntelligenceEvent,
    SourceClass,
)
from research_pipeline.briefing.normalize import (
    cluster_id_for,
    normalize_title,
    topic_id_for_title,
)

SOURCE_CLASS_PREFERENCE: dict[SourceClass, float] = {
    SourceClass.IMPLEMENTATION_SOURCE: 3.0,
    SourceClass.PRIMARY_ARTIFACT: 2.8,
    SourceClass.ACADEMIC_SOURCE: 2.6,
    SourceClass.NEWSLETTER: 1.7,
    SourceClass.TECHNICAL_DISCUSSION: 1.2,
    SourceClass.MEDIA_NEWS: 0.7,
    SourceClass.VIDEO_AUDIO: 0.5,
    SourceClass.SOCIAL_SIGNAL: 0.1,
}

CONFIDENCE_RANK = {"high": 3, "medium": 2, "low": 1}
EVIDENCE_RANK = {
    "supported_fact": 3,
    "inference": 2,
    "speculation_or_watch_item": 1,
}


def cluster_events(events: list[IntelligenceEvent]) -> list[BriefingCluster]:
    """Deduplicate events by exact keys and build stable clusters."""
    grouped: dict[str, list[IntelligenceEvent]] = defaultdict(list)
    key_index: dict[str, str] = {}
    for event in events:
        keys = [
            event.dedup_key or "",
            f"url:{event.canonical_url}",
            f"title:{normalize_title(event.title)}",
        ]
        key = next((key_index[item] for item in keys if item in key_index), "")
        if not key:
            key = event.dedup_key or f"title:{normalize_title(event.title)}"
        for item in keys:
            if item:
                key_index[item] = key
        grouped[key].append(event)

    clusters: list[BriefingCluster] = []
    for dedup_key, group in grouped.items():
        ordered = sorted(
            group,
            key=_event_quality_key,
            reverse=True,
        )
        primary = ordered[0]
        urls = tuple(dict.fromkeys(str(event.canonical_url) for event in ordered))
        topics = tuple(
            dict.fromkeys(
                topic
                for event in ordered
                for topic in (event.topics or (topic_id_for_title(event.title),))
            )
        )
        source_classes = tuple(dict.fromkeys(event.source_type for event in ordered))
        first_seen = min(event.retrieved_at for event in ordered)
        last_seen = max(event.retrieved_at for event in ordered)
        primary_present = any(
            event.source_type
            in {
                SourceClass.PRIMARY_ARTIFACT,
                SourceClass.IMPLEMENTATION_SOURCE,
                SourceClass.ACADEMIC_SOURCE,
            }
            for event in ordered
        )
        clusters.append(
            BriefingCluster(
                cluster_id=cluster_id_for(dedup_key),
                title=primary.title,
                primary_event_id=primary.event_id,
                event_ids=tuple(event.event_id for event in ordered),
                topic_ids=topics,
                canonical_urls=urls,
                first_seen_at=first_seen,
                last_seen_at=last_seen,
                source_classes=source_classes,
                primary_artifact_present=primary_present,
                confidence=primary.confidence,
                evidence_type=primary.evidence_type,
                duplicate_penalty=max(0.0, float(len(ordered) - 1) * 0.1),
                events=tuple(ordered),
            )
        )
    return sorted(
        clusters, key=lambda cluster: (cluster.title.lower(), cluster.cluster_id)
    )


def _event_quality_key(
    event: IntelligenceEvent,
) -> tuple[float, int, int, int, str, str]:
    summary = event.summary_hint or event.excerpt
    duplicate_penalty = 1 if "duplicate" in summary.lower() else 0
    return (
        SOURCE_CLASS_PREFERENCE.get(event.source_type, 0.0),
        CONFIDENCE_RANK.get(event.confidence, 0),
        EVIDENCE_RANK.get(event.evidence_type, 0),
        len(summary) - (1000 * duplicate_penalty),
        event.published_at or "",
        event.event_id,
    )
