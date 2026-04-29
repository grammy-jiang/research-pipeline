"""Artifact layout and loading/saving for briefing runs."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from research_pipeline.briefing.io import read_json, read_jsonl, write_json, write_jsonl
from research_pipeline.briefing.models import BriefingCluster, IntelligenceEvent


@dataclass(frozen=True)
class ArtifactLayout:
    """Fixed artifact layout for a single briefing run."""

    root: Path
    raw_dir: Path = field(init=False)
    normalized_dir: Path = field(init=False)
    clusters_dir: Path = field(init=False)
    ranked_dir: Path = field(init=False)
    reports_dir: Path = field(init=False)
    validation_dir: Path = field(init=False)
    telemetry_path: Path = field(init=False)
    source_snapshot_path: Path = field(init=False)

    def __post_init__(self) -> None:
        """Initialize derived paths after root is set."""
        object.__setattr__(self, "raw_dir", self.root / "raw")
        object.__setattr__(self, "normalized_dir", self.root / "normalized")
        object.__setattr__(self, "clusters_dir", self.root / "clusters")
        object.__setattr__(self, "ranked_dir", self.root / "ranked")
        object.__setattr__(self, "reports_dir", self.root / "reports")
        object.__setattr__(self, "validation_dir", self.root / "validation")
        object.__setattr__(self, "telemetry_path", self.root / "telemetry.jsonl")
        object.__setattr__(
            self, "source_snapshot_path", self.root / "source_registry_snapshot.json"
        )

    @property
    def events_path(self) -> Path:
        """Path to normalized events JSONL."""
        return self.normalized_dir / "events.jsonl"

    @property
    def clusters_path(self) -> Path:
        """Path to clusters JSONL."""
        return self.clusters_dir / "clusters.jsonl"

    @property
    def ranked_clusters_path(self) -> Path:
        """Path to ranked clusters JSONL."""
        return self.ranked_dir / "ranked_clusters.jsonl"

    @property
    def daily_report_path(self) -> Path:
        """Path to the daily Markdown report."""
        return self.reports_dir / "daily.md"

    @property
    def validation_path(self) -> Path:
        """Path to validation JSON."""
        return self.validation_dir / "validation.json"

    def create(self) -> None:
        """Create all required artifact directories."""
        for path in (
            self.raw_dir,
            self.normalized_dir,
            self.clusters_dir,
            self.ranked_dir,
            self.reports_dir,
            self.validation_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


@dataclass
class LoadedArtifacts:
    """Container for loaded briefing artifacts."""

    events: list[IntelligenceEvent] | None = None
    clusters: list[BriefingCluster] | None = None
    ranked_clusters: list[BriefingCluster] | None = None
    daily_report: str | None = None
    validation: dict[str, Any] | None = None


def save_artifacts(
    layout: ArtifactLayout,
    *,
    events: list[IntelligenceEvent] | None = None,
    clusters: list[BriefingCluster] | None = None,
    ranked_clusters: list[BriefingCluster] | None = None,
    daily_report: str | None = None,
    validation: dict[str, Any] | None = None,
    raw_sources: dict[str, list[IntelligenceEvent]] | None = None,
    source_snapshot: dict[str, Any] | None = None,
    telemetry_events: list[dict[str, Any]] | None = None,
) -> None:
    """Save briefing artifacts to the fixed layout."""
    if events is not None:
        write_jsonl(layout.events_path, events)

    if clusters is not None:
        write_jsonl(layout.clusters_path, clusters)

    if ranked_clusters is not None:
        write_jsonl(layout.ranked_clusters_path, ranked_clusters)

    if daily_report is not None:
        layout.daily_report_path.parent.mkdir(parents=True, exist_ok=True)
        layout.daily_report_path.write_text(daily_report)

    if validation is not None:
        write_json(layout.validation_path, validation)

    if raw_sources is not None:
        for source_id, source_events in raw_sources.items():
            raw_path = layout.raw_dir / f"{source_id}.jsonl"
            write_jsonl(raw_path, source_events)

    if source_snapshot is not None:
        write_json(layout.source_snapshot_path, source_snapshot)

    if telemetry_events is not None:
        layout.telemetry_path.parent.mkdir(parents=True, exist_ok=True)
        with open(layout.telemetry_path, "a") as f:
            for event in telemetry_events:
                f.write(json.dumps(event) + "\n")


def load_artifacts(
    layout: ArtifactLayout,
    *,
    load_events: bool = False,
    load_clusters: bool = False,
    load_ranked_clusters: bool = False,
    load_daily_report: bool = False,
    load_validation: bool = False,
) -> LoadedArtifacts:
    """Load briefing artifacts from the fixed layout."""
    result = LoadedArtifacts()

    if load_events and layout.events_path.exists():
        result.events = read_jsonl(layout.events_path, IntelligenceEvent)

    if load_clusters and layout.clusters_path.exists():
        result.clusters = read_jsonl(layout.clusters_path, BriefingCluster)

    if load_ranked_clusters and layout.ranked_clusters_path.exists():
        result.ranked_clusters = read_jsonl(
            layout.ranked_clusters_path, BriefingCluster
        )

    if load_daily_report and layout.daily_report_path.exists():
        result.daily_report = layout.daily_report_path.read_text()

    if load_validation and layout.validation_path.exists():
        result.validation = read_json(layout.validation_path)

    return result
