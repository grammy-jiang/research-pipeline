from __future__ import annotations

from research_pipeline.briefing.dedup import cluster_events
from research_pipeline.briefing.models import (
    AccessMethod,
    IntelligenceEvent,
    SourceClass,
)


def _event(
    *,
    event_id: str,
    dedup_key: str,
    title: str,
    canonical_url: str,
    source_type: SourceClass = SourceClass.TECHNICAL_DISCUSSION,
    confidence: str = "medium",
    evidence_type: str = "supported_fact",
    summary_hint: str = "",
    published_at: str | None = None,
    retrieved_at: str = "2026-04-29T00:00:00Z",
) -> IntelligenceEvent:
    return IntelligenceEvent(
        event_id=event_id,
        source_name="Test Source",
        source_id="test.source",
        source_type=source_type,
        item_type="test_item",
        canonical_url=canonical_url,
        title=title,
        retrieved_at=retrieved_at,
        collection_method=AccessMethod.RSS_ATOM,
        content_hash=f"hash-{event_id}",
        dedup_key=dedup_key,
        source_native_id=event_id,
        summary_hint=summary_hint,
        confidence=confidence,  # type: ignore[arg-type]
        evidence_type=evidence_type,  # type: ignore[arg-type]
        published_at=published_at,
    )


def test_cluster_events_empty_input_returns_empty_list() -> None:
    assert cluster_events([]) == []


def test_cluster_events_merges_exact_dedup_key() -> None:
    events = [
        _event(
            event_id="e1",
            dedup_key="native:test:a",
            title="A Title",
            canonical_url="https://example.com/a",
        ),
        _event(
            event_id="e2",
            dedup_key="native:test:a",
            title="A Title Duplicate",
            canonical_url="https://example.com/a-dup",
            summary_hint="duplicate mention",
        ),
    ]

    clusters = cluster_events(events)

    assert len(clusters) == 1
    cluster = clusters[0]
    assert cluster.cluster_id.startswith("cluster_")
    assert set(cluster.event_ids) == {"e1", "e2"}
    assert cluster.duplicate_penalty == 0.1


def test_cluster_events_merges_by_url_fallback() -> None:
    events = [
        _event(
            event_id="u1",
            dedup_key="native:test:1",
            title="URL Match A",
            canonical_url="https://example.com/same",
        ),
        _event(
            event_id="u2",
            dedup_key="native:test:2",
            title="URL Match B",
            canonical_url="https://example.com/same",
        ),
    ]

    clusters = cluster_events(events)

    assert len(clusters) == 1
    assert set(clusters[0].event_ids) == {"u1", "u2"}


def test_cluster_events_merges_by_normalized_title_fallback() -> None:
    events = [
        _event(
            event_id="t1",
            dedup_key="native:test:t1",
            title="  Same   Title  ",
            canonical_url="https://example.com/one",
        ),
        _event(
            event_id="t2",
            dedup_key="native:test:t2",
            title="same title",
            canonical_url="https://example.com/two",
        ),
    ]

    clusters = cluster_events(events)

    assert len(clusters) == 1
    assert set(clusters[0].event_ids) == {"t1", "t2"}


def test_cluster_primary_selection_prefers_higher_quality_source() -> None:
    events = [
        _event(
            event_id="low",
            dedup_key="native:test:quality",
            title="Quality Topic",
            canonical_url="https://example.com/low",
            source_type=SourceClass.SOCIAL_SIGNAL,
            confidence="low",
            evidence_type="speculation_or_watch_item",
            summary_hint="duplicate noisy mention",
        ),
        _event(
            event_id="high",
            dedup_key="native:test:quality",
            title="Quality Topic Official",
            canonical_url="https://example.com/high",
            source_type=SourceClass.IMPLEMENTATION_SOURCE,
            confidence="high",
            evidence_type="supported_fact",
            summary_hint="official release notes",
            published_at="2026-04-29T10:00:00Z",
        ),
    ]

    clusters = cluster_events(events)

    assert len(clusters) == 1
    assert clusters[0].primary_event_id == "high"
    assert clusters[0].title == "Quality Topic Official"


def test_cluster_output_order_is_deterministic() -> None:
    events = [
        _event(
            event_id="b1",
            dedup_key="native:test:b",
            title="Beta",
            canonical_url="https://example.com/b",
        ),
        _event(
            event_id="a1",
            dedup_key="native:test:a",
            title="Alpha",
            canonical_url="https://example.com/a",
        ),
    ]

    clusters = cluster_events(events)

    assert [cluster.title for cluster in clusters] == ["Alpha", "Beta"]
