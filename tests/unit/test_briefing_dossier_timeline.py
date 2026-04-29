"""Phase E E05 — evidence timeline construction."""

from __future__ import annotations

from research_pipeline.briefing.dossier_timeline import build_evidence_timeline
from research_pipeline.briefing.models import SourceClass, TopicMemory
from tests.unit._dossier_fixtures import make_cluster, make_event


def test_timeline_from_cluster_events_only() -> None:
    e1 = make_event(
        event_id="e1",
        canonical_url="https://example.com/a",
        published_at="2026-04-29",
    )
    e2 = make_event(
        event_id="e2",
        canonical_url="https://example.com/b",
        published_at="2026-04-28",
    )
    cluster = make_cluster(events=(e1, e2))
    entries = build_evidence_timeline(cluster)
    assert len(entries) == 2
    # Sorted by date ascending
    assert entries[0].date == "2026-04-28"
    assert entries[1].date == "2026-04-29"
    assert all(e.origin == "cluster_event" for e in entries)


def test_timeline_dedup_by_url() -> None:
    e1 = make_event(event_id="e1", canonical_url="https://example.com/a")
    e2 = make_event(event_id="e2", canonical_url="https://example.com/a")
    cluster = make_cluster(events=(e1, e2))
    entries = build_evidence_timeline(cluster)
    assert len(entries) == 1


def test_timeline_with_topic_memory() -> None:
    cluster = make_cluster()
    memory = TopicMemory(
        topic_id="topic_acme",
        name="Acme",
        first_seen_at="2026-03-01T00:00:00Z",
        last_seen_at="2026-04-01T00:00:00Z",
        last_reported_at="2026-04-01T00:00:00Z",
        obsidian_note="https://vault.local/topic_acme",
    )
    entries = build_evidence_timeline(cluster, topic_memories=[memory])
    assert len(entries) == 2
    origins = sorted(e.origin for e in entries)
    assert origins == ["cluster_event", "topic_memory"]


def test_timeline_empty_topic_memory() -> None:
    cluster = make_cluster()
    entries = build_evidence_timeline(cluster, topic_memories=[])
    assert len(entries) == 1
    assert entries[0].origin == "cluster_event"


def test_timeline_topic_memory_dedup() -> None:
    cluster = make_cluster(
        events=(make_event(canonical_url="https://example.com/release"),)
    )
    memory = TopicMemory(
        topic_id="topic_acme",
        name="Acme",
        first_seen_at="2026-03-01T00:00:00Z",
        last_seen_at="2026-04-01T00:00:00Z",
        last_reported_at="2026-04-01T00:00:00Z",
        obsidian_note="https://example.com/release",  # same URL as event
    )
    entries = build_evidence_timeline(cluster, topic_memories=[memory])
    assert len(entries) == 1


def test_timeline_topic_memory_obsidian_uri_fallback() -> None:
    cluster = make_cluster()
    memory = TopicMemory(
        topic_id="topic_acme",
        name="Acme",
        first_seen_at="2026-03-01T00:00:00Z",
        last_seen_at="2026-04-01T00:00:00Z",
        last_reported_at="2026-04-01T00:00:00Z",
        obsidian_note="topic_acme.md",  # not a URL — falls back to obsidian://
    )
    entries = build_evidence_timeline(cluster, topic_memories=[memory])
    memory_entries = [e for e in entries if e.origin == "topic_memory"]
    assert len(memory_entries) == 1
    assert memory_entries[0].evidence_url.startswith("obsidian://")


def test_timeline_deterministic_sort() -> None:
    e1 = make_event(
        event_id="e1",
        canonical_url="https://example.com/b",
        published_at="2026-04-29",
    )
    e2 = make_event(
        event_id="e2",
        canonical_url="https://example.com/a",
        published_at="2026-04-29",  # same date
    )
    cluster = make_cluster(events=(e1, e2))
    entries1 = build_evidence_timeline(cluster)
    entries2 = build_evidence_timeline(cluster)
    assert entries1 == entries2
    # Tie-break on URL: 'a' before 'b'
    assert entries1[0].evidence_url.endswith("/a")


def test_timeline_event_source_class_preserved() -> None:
    e = make_event(source_class=SourceClass.PRIMARY_ARTIFACT)
    cluster = make_cluster(events=(e,))
    entries = build_evidence_timeline(cluster)
    assert entries[0].source_class == "primary_artifact"
