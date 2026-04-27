"""Fixed artifact layout for briefing runs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from research_pipeline.briefing.normalize import today_utc


@dataclass(frozen=True)
class BriefingPaths:
    """Resolved paths for one briefing run."""

    root: Path
    raw_dir: Path
    normalized_dir: Path
    clusters_dir: Path
    ranked_dir: Path
    reports_dir: Path
    validation_dir: Path
    telemetry_path: Path
    source_snapshot_path: Path

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
        """Create the fixed artifact directories."""
        for path in (
            self.raw_dir,
            self.normalized_dir,
            self.clusters_dir,
            self.ranked_dir,
            self.reports_dir,
            self.validation_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


def resolve_briefing_paths(
    workspace: Path | None = None, run_date: str | None = None
) -> BriefingPaths:
    """Resolve the Phase A briefing artifact layout."""
    root_workspace = workspace or Path("./workspace")
    date_value = run_date or today_utc()
    root = root_workspace / "briefings" / date_value
    return BriefingPaths(
        root=root,
        raw_dir=root / "raw",
        normalized_dir=root / "normalized",
        clusters_dir=root / "clusters",
        ranked_dir=root / "ranked",
        reports_dir=root / "reports",
        validation_dir=root / "validation",
        telemetry_path=root / "telemetry.jsonl",
        source_snapshot_path=root / "source_registry_snapshot.json",
    )
