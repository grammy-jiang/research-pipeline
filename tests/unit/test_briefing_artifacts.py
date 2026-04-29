"""Tests for briefing artifact layout management."""

from __future__ import annotations

import json
from pathlib import Path

from research_pipeline.briefing.artifacts import (
    ArtifactLayout,
    load_artifacts,
    save_artifacts,
)
from research_pipeline.briefing.models import (
    AccessMethod,
    BriefingCluster,
    IntelligenceEvent,
    SourceClass,
)


def _event(
    event_id: str, source_id: str = "test_source", title: str = "Test Event"
) -> IntelligenceEvent:
    """Helper to create IntelligenceEvent for testing."""
    return IntelligenceEvent(
        event_id=event_id,
        source_name="Test Source",
        source_id=source_id,
        source_type=SourceClass.PRIMARY_ARTIFACT,
        item_type="release",
        canonical_url="https://example.com/event",
        title=title,
        retrieved_at="2025-01-20T12:00:00Z",
        collection_method=AccessMethod.API,
        content_hash="abc123",
        dedup_key=f"dedup_{event_id}",
        published_at="2025-01-20T00:00:00Z",
    )


def _cluster(
    cluster_id: str, source_class: SourceClass = SourceClass.PRIMARY_ARTIFACT
) -> BriefingCluster:
    """Helper to create BriefingCluster for testing."""
    return BriefingCluster(
        cluster_id=cluster_id,
        title=f"Cluster {cluster_id}",
        primary_event_id="e0",
        event_ids=("e0",),
        canonical_urls=("https://example.com/c1:e1",),
        first_seen_at="2025-01-20T00:00:00Z",
        last_seen_at="2025-01-20T12:00:00Z",
        source_classes=(source_class,),
        primary_artifact_present=True,
        rank_score=0.95,
        authority_score=0.8,
    )


def test_artifact_layout_creates_directory_structure(tmp_path: Path) -> None:
    """Artifact layout creates fixed directory structure on demand."""
    layout = ArtifactLayout(tmp_path / "briefings" / "2025-01-20")

    layout.create()

    assert layout.raw_dir.exists()
    assert layout.normalized_dir.exists()
    assert layout.clusters_dir.exists()
    assert layout.ranked_dir.exists()
    assert layout.reports_dir.exists()
    assert layout.validation_dir.exists()


def test_artifact_layout_properties_resolve_correctly(tmp_path: Path) -> None:
    """Artifact layout properties resolve to correct paths."""
    root = tmp_path / "briefings" / "2025-01-20"
    layout = ArtifactLayout(root)

    assert layout.events_path == root / "normalized" / "events.jsonl"
    assert layout.clusters_path == root / "clusters" / "clusters.jsonl"
    assert layout.ranked_clusters_path == root / "ranked" / "ranked_clusters.jsonl"
    assert layout.daily_report_path == root / "reports" / "daily.md"
    assert layout.validation_path == root / "validation" / "validation.json"
    assert layout.source_snapshot_path == root / "source_registry_snapshot.json"
    assert layout.telemetry_path == root / "telemetry.jsonl"


def test_save_and_load_events(tmp_path: Path) -> None:
    """Save and load intelligence events JSONL."""
    layout = ArtifactLayout(tmp_path / "briefings" / "2025-01-20")
    layout.create()
    events = [_event(f"e{i}") for i in range(3)]

    save_artifacts(layout, events=events)
    loaded = load_artifacts(layout, load_events=True)

    assert loaded.events == events
    assert len(loaded.events) == 3


def test_save_and_load_clusters(tmp_path: Path) -> None:
    """Save and load briefing clusters JSONL."""
    layout = ArtifactLayout(tmp_path / "briefings" / "2025-01-20")
    layout.create()
    clusters = [_cluster(f"c{i}") for i in range(2)]

    save_artifacts(layout, clusters=clusters)
    loaded = load_artifacts(layout, load_clusters=True)

    assert loaded.clusters == clusters
    assert len(loaded.clusters) == 2


def test_save_and_load_ranked_clusters(tmp_path: Path) -> None:
    """Save and load ranked briefing clusters."""
    layout = ArtifactLayout(tmp_path / "briefings" / "2025-01-20")
    layout.create()
    clusters = [_cluster(f"c{i}") for i in range(2)]

    save_artifacts(layout, ranked_clusters=clusters)
    loaded = load_artifacts(layout, load_ranked_clusters=True)

    assert loaded.ranked_clusters == clusters


def test_save_and_load_daily_report(tmp_path: Path) -> None:
    """Save and load daily Markdown report."""
    layout = ArtifactLayout(tmp_path / "briefings" / "2025-01-20")
    layout.create()
    report_content = "# Daily Brief\n\n## Executive Signal\nTest."

    save_artifacts(layout, daily_report=report_content)
    loaded = load_artifacts(layout, load_daily_report=True)

    assert loaded.daily_report == report_content


def test_save_and_load_validation_result(tmp_path: Path) -> None:
    """Save and load validation result JSON."""
    layout = ArtifactLayout(tmp_path / "briefings" / "2025-01-20")
    layout.create()
    validation = {"passed": True, "errors": [], "warnings": []}

    save_artifacts(layout, validation=validation)
    loaded = load_artifacts(layout, load_validation=True)

    assert loaded.validation == validation
    assert loaded.validation["passed"] is True


def test_save_raw_source_data(tmp_path: Path) -> None:
    """Save raw source JSONL data."""
    layout = ArtifactLayout(tmp_path / "briefings" / "2025-01-20")
    layout.create()
    events = [_event(f"e{i}", source_id="github_releases") for i in range(2)]

    save_artifacts(layout, raw_sources={"github_releases": events})

    raw_path = layout.raw_dir / "github_releases.jsonl"
    assert raw_path.exists()
    content = raw_path.read_text()
    lines = content.strip().split("\n")
    assert len(lines) == 2


def test_load_artifacts_selective(tmp_path: Path) -> None:
    """Load only requested artifacts."""
    layout = ArtifactLayout(tmp_path / "briefings" / "2025-01-20")
    layout.create()
    events = [_event("e1")]
    clusters = [_cluster("c1")]

    save_artifacts(layout, events=events, clusters=clusters)
    loaded = load_artifacts(layout, load_events=True, load_clusters=False)

    assert loaded.events == events
    assert loaded.clusters is None


def test_artifact_layout_handles_missing_files(tmp_path: Path) -> None:
    """Load artifacts gracefully handles missing files."""
    layout = ArtifactLayout(tmp_path / "briefings" / "2025-01-20")
    layout.create()

    loaded = load_artifacts(layout, load_events=True)

    assert loaded.events is None or loaded.events == []


def test_save_source_snapshot(tmp_path: Path) -> None:
    """Save source registry snapshot JSON."""
    layout = ArtifactLayout(tmp_path / "briefings" / "2025-01-20")
    layout.create()
    snapshot = {
        "sources": [
            {
                "source_id": "github_releases",
                "enabled": True,
                "poll_interval_hours": 24,
            }
        ]
    }

    save_artifacts(layout, source_snapshot=snapshot)

    assert layout.source_snapshot_path.exists()
    content = json.loads(layout.source_snapshot_path.read_text())
    assert content["sources"][0]["source_id"] == "github_releases"


def test_save_telemetry_events(tmp_path: Path) -> None:
    """Append telemetry events to JSONL."""
    layout = ArtifactLayout(tmp_path / "briefings" / "2025-01-20")
    layout.create()
    telemetry = [
        {"event": "poll_started", "timestamp": "2025-01-20T12:00:00Z"},
        {"event": "poll_completed", "timestamp": "2025-01-20T12:05:00Z"},
    ]

    save_artifacts(layout, telemetry_events=telemetry)

    assert layout.telemetry_path.exists()
    content = layout.telemetry_path.read_text()
    lines = content.strip().split("\n")
    assert len(lines) == 2


def test_artifact_layout_supports_concurrent_reads(tmp_path: Path) -> None:
    """Multiple readers can load artifacts concurrently."""
    layout = ArtifactLayout(tmp_path / "briefings" / "2025-01-20")
    layout.create()
    events = [_event(f"e{i}") for i in range(3)]

    save_artifacts(layout, events=events)

    # Simulate concurrent reads
    loaded1 = load_artifacts(layout, load_events=True)
    loaded2 = load_artifacts(layout, load_events=True)

    assert loaded1.events == loaded2.events
    assert loaded1.events is not None
    assert len(loaded1.events) == 3


def test_artifact_dates_are_normalized(tmp_path: Path) -> None:
    """Artifact paths are normalized to YYYY-MM-DD format."""
    root = tmp_path / "briefings" / "2025-01-20"
    layout = ArtifactLayout(root)

    assert "2025-01-20" in str(layout.root)
    assert not str(layout.root).endswith(".")
