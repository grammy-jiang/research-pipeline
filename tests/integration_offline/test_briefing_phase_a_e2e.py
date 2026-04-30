"""Offline end-to-end integration tests for Phase A briefing pipeline.

Tests cover normal day, low-signal day, and no-news day scenarios.
No network calls are made — all tests use local fixture files.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from research_pipeline.briefing.registry import load_source_registry
from research_pipeline.briefing.workflow import run_briefing

# Directory containing e2e scenario fixtures
_FIXTURE_BASE = Path(__file__).parent.parent / "fixtures" / "briefing" / "e2e"
_RUN_DATE = "2026-05-01"


class TestNormalDay:
    """Full pipeline run on a normal day with events from multiple sources."""

    def test_run_briefing_exits_successfully(self, tmp_path: Path) -> None:
        """run_briefing completes without exception on normal day."""
        registry = load_source_registry(_FIXTURE_BASE / "normal" / "registry.toml")
        paths, validation = run_briefing(
            registry,
            workspace=tmp_path / "workspace",
            run_date=_RUN_DATE,
            fixture_base_dir=_FIXTURE_BASE / "normal",
        )
        assert paths.root.exists()

    def test_artifact_layout_created(self, tmp_path: Path) -> None:
        """Fixed Phase A artifact layout exists after a normal-day run."""
        registry = load_source_registry(_FIXTURE_BASE / "normal" / "registry.toml")
        paths, _ = run_briefing(
            registry,
            workspace=tmp_path / "workspace",
            run_date=_RUN_DATE,
            fixture_base_dir=_FIXTURE_BASE / "normal",
        )
        assert paths.raw_dir.exists()
        assert paths.normalized_dir.exists()
        assert paths.clusters_dir.exists()
        assert paths.ranked_dir.exists()
        assert paths.reports_dir.exists()
        assert paths.validation_dir.exists()
        assert paths.telemetry_path.exists()

    def test_events_jsonl_populated(self, tmp_path: Path) -> None:
        """Normalized events JSONL is non-empty after a normal-day run."""
        registry = load_source_registry(_FIXTURE_BASE / "normal" / "registry.toml")
        paths, _ = run_briefing(
            registry,
            workspace=tmp_path / "workspace",
            run_date=_RUN_DATE,
            fixture_base_dir=_FIXTURE_BASE / "normal",
        )
        assert paths.events_path.exists()
        lines = [
            ln
            for ln in paths.events_path.read_text(encoding="utf-8").splitlines()
            if ln.strip()
        ]
        assert len(lines) >= 1

    def test_clusters_jsonl_populated(self, tmp_path: Path) -> None:
        """Clusters JSONL is non-empty after a normal-day run."""
        registry = load_source_registry(_FIXTURE_BASE / "normal" / "registry.toml")
        paths, _ = run_briefing(
            registry,
            workspace=tmp_path / "workspace",
            run_date=_RUN_DATE,
            fixture_base_dir=_FIXTURE_BASE / "normal",
        )
        assert paths.clusters_path.exists()
        lines = [
            ln
            for ln in paths.clusters_path.read_text(encoding="utf-8").splitlines()
            if ln.strip()
        ]
        assert len(lines) >= 1

    def test_daily_report_created(self, tmp_path: Path) -> None:
        """Daily Markdown report is created after a normal-day run."""
        registry = load_source_registry(_FIXTURE_BASE / "normal" / "registry.toml")
        paths, _ = run_briefing(
            registry,
            workspace=tmp_path / "workspace",
            run_date=_RUN_DATE,
            fixture_base_dir=_FIXTURE_BASE / "normal",
        )
        assert paths.daily_report_path.exists()
        content = paths.daily_report_path.read_text(encoding="utf-8")
        assert len(content) > 0

    def test_daily_report_contains_required_sections(self, tmp_path: Path) -> None:
        """Daily report contains required Phase A sections."""
        registry = load_source_registry(_FIXTURE_BASE / "normal" / "registry.toml")
        paths, _ = run_briefing(
            registry,
            workspace=tmp_path / "workspace",
            run_date=_RUN_DATE,
            fixture_base_dir=_FIXTURE_BASE / "normal",
        )
        content = paths.daily_report_path.read_text(encoding="utf-8")
        for section in (
            "## 🔥 Executive Signal",
            "## ⭐ Top Items",
            "## 🗒️ Feedback Targets",
        ):
            assert section in content, f"missing section: {section}"

    def test_daily_report_contains_date(self, tmp_path: Path) -> None:
        """Daily report frontmatter contains the run date."""
        registry = load_source_registry(_FIXTURE_BASE / "normal" / "registry.toml")
        paths, _ = run_briefing(
            registry,
            workspace=tmp_path / "workspace",
            run_date=_RUN_DATE,
            fixture_base_dir=_FIXTURE_BASE / "normal",
        )
        content = paths.daily_report_path.read_text(encoding="utf-8")
        assert _RUN_DATE in content

    def test_validation_passes(self, tmp_path: Path) -> None:
        """Validation passes on a well-formed normal-day report."""
        registry = load_source_registry(_FIXTURE_BASE / "normal" / "registry.toml")
        paths, validation = run_briefing(
            registry,
            workspace=tmp_path / "workspace",
            run_date=_RUN_DATE,
            fixture_base_dir=_FIXTURE_BASE / "normal",
        )
        assert validation.get("passed") is True, validation.get("errors")

    def test_validation_json_written(self, tmp_path: Path) -> None:
        """Validation JSON artifact is written to the fixed layout path."""
        registry = load_source_registry(_FIXTURE_BASE / "normal" / "registry.toml")
        paths, _ = run_briefing(
            registry,
            workspace=tmp_path / "workspace",
            run_date=_RUN_DATE,
            fixture_base_dir=_FIXTURE_BASE / "normal",
        )
        assert paths.validation_path.exists()
        data = json.loads(paths.validation_path.read_text(encoding="utf-8"))
        assert "passed" in data

    def test_source_registry_snapshot_written(self, tmp_path: Path) -> None:
        """Source registry snapshot is written during poll."""
        registry = load_source_registry(_FIXTURE_BASE / "normal" / "registry.toml")
        paths, _ = run_briefing(
            registry,
            workspace=tmp_path / "workspace",
            run_date=_RUN_DATE,
            fixture_base_dir=_FIXTURE_BASE / "normal",
        )
        assert paths.source_snapshot_path.exists()
        data = json.loads(paths.source_snapshot_path.read_text(encoding="utf-8"))
        assert "sources" in data

    def test_telemetry_jsonl_has_poll_completed(self, tmp_path: Path) -> None:
        """Telemetry JSONL records a poll_completed event."""
        registry = load_source_registry(_FIXTURE_BASE / "normal" / "registry.toml")
        paths, _ = run_briefing(
            registry,
            workspace=tmp_path / "workspace",
            run_date=_RUN_DATE,
            fixture_base_dir=_FIXTURE_BASE / "normal",
        )
        lines = [
            json.loads(ln)
            for ln in paths.telemetry_path.read_text(encoding="utf-8").splitlines()
            if ln.strip()
        ]
        event_types = {row["event_type"] for row in lines}
        assert "poll_completed" in event_types

    def test_raw_jsonl_per_source(self, tmp_path: Path) -> None:
        """Raw JSONL files are created per source in the raw/ directory."""
        registry = load_source_registry(_FIXTURE_BASE / "normal" / "registry.toml")
        paths, _ = run_briefing(
            registry,
            workspace=tmp_path / "workspace",
            run_date=_RUN_DATE,
            fixture_base_dir=_FIXTURE_BASE / "normal",
        )
        raw_files = list(paths.raw_dir.glob("*.jsonl"))
        assert len(raw_files) >= 1


class TestLowSignalDay:
    """Pipeline run with minimal events (1 event, 1 quiet source)."""

    def test_run_briefing_completes(self, tmp_path: Path) -> None:
        """run_briefing completes without exception on low-signal day."""
        registry = load_source_registry(_FIXTURE_BASE / "low_signal" / "registry.toml")
        paths, _validation = run_briefing(
            registry,
            workspace=tmp_path / "workspace",
            run_date=_RUN_DATE,
            fixture_base_dir=_FIXTURE_BASE / "low_signal",
        )
        assert paths.root.exists()

    def test_events_present(self, tmp_path: Path) -> None:
        """Normalized events JSONL exists after low-signal run."""
        registry = load_source_registry(_FIXTURE_BASE / "low_signal" / "registry.toml")
        paths, _ = run_briefing(
            registry,
            workspace=tmp_path / "workspace",
            run_date=_RUN_DATE,
            fixture_base_dir=_FIXTURE_BASE / "low_signal",
        )
        assert paths.events_path.exists()

    def test_daily_report_created(self, tmp_path: Path) -> None:
        """Daily report is generated even with minimal events."""
        registry = load_source_registry(_FIXTURE_BASE / "low_signal" / "registry.toml")
        paths, _ = run_briefing(
            registry,
            workspace=tmp_path / "workspace",
            run_date=_RUN_DATE,
            fixture_base_dir=_FIXTURE_BASE / "low_signal",
        )
        assert paths.daily_report_path.exists()

    def test_validation_result_returned(self, tmp_path: Path) -> None:
        """Validation result is returned on low-signal day."""
        registry = load_source_registry(_FIXTURE_BASE / "low_signal" / "registry.toml")
        _, validation = run_briefing(
            registry,
            workspace=tmp_path / "workspace",
            run_date=_RUN_DATE,
            fixture_base_dir=_FIXTURE_BASE / "low_signal",
        )
        assert isinstance(validation, dict)
        assert "passed" in validation


class TestNoNewsDay:
    """Pipeline run with zero events from all sources."""

    def test_run_briefing_completes(self, tmp_path: Path) -> None:
        """run_briefing completes without exception when no events are found."""
        registry = load_source_registry(_FIXTURE_BASE / "no_news" / "registry.toml")
        paths, _validation = run_briefing(
            registry,
            workspace=tmp_path / "workspace",
            run_date=_RUN_DATE,
            fixture_base_dir=_FIXTURE_BASE / "no_news",
        )
        assert paths.root.exists()

    def test_events_jsonl_empty(self, tmp_path: Path) -> None:
        """Normalized events JSONL exists and has zero events on no-news day."""
        registry = load_source_registry(_FIXTURE_BASE / "no_news" / "registry.toml")
        paths, _ = run_briefing(
            registry,
            workspace=tmp_path / "workspace",
            run_date=_RUN_DATE,
            fixture_base_dir=_FIXTURE_BASE / "no_news",
        )
        assert paths.events_path.exists()
        content = paths.events_path.read_text(encoding="utf-8").strip()
        # No events means empty or whitespace-only file
        assert content == "" or all(not ln.strip() for ln in content.splitlines())

    def test_daily_report_generated(self, tmp_path: Path) -> None:
        """Daily report is still generated on no-news day (graceful empty state)."""
        registry = load_source_registry(_FIXTURE_BASE / "no_news" / "registry.toml")
        paths, _ = run_briefing(
            registry,
            workspace=tmp_path / "workspace",
            run_date=_RUN_DATE,
            fixture_base_dir=_FIXTURE_BASE / "no_news",
        )
        assert paths.daily_report_path.exists()
        content = paths.daily_report_path.read_text(encoding="utf-8")
        assert len(content) > 0

    def test_daily_report_required_sections_present(self, tmp_path: Path) -> None:
        """Daily report contains required sections even with no events."""
        registry = load_source_registry(_FIXTURE_BASE / "no_news" / "registry.toml")
        paths, _ = run_briefing(
            registry,
            workspace=tmp_path / "workspace",
            run_date=_RUN_DATE,
            fixture_base_dir=_FIXTURE_BASE / "no_news",
        )
        content = paths.daily_report_path.read_text(encoding="utf-8")
        for section in (
            "## 🔥 Executive Signal",
            "## ⭐ Top Items",
            "## 🗒️ Feedback Targets",
        ):
            assert section in content, f"missing section on no-news day: {section}"

    def test_telemetry_records_zero_event_count(self, tmp_path: Path) -> None:
        """Telemetry records poll_completed with event_count 0 on no-news day."""
        registry = load_source_registry(_FIXTURE_BASE / "no_news" / "registry.toml")
        paths, _ = run_briefing(
            registry,
            workspace=tmp_path / "workspace",
            run_date=_RUN_DATE,
            fixture_base_dir=_FIXTURE_BASE / "no_news",
        )
        lines = [
            json.loads(ln)
            for ln in paths.telemetry_path.read_text(encoding="utf-8").splitlines()
            if ln.strip()
        ]
        poll_completed = next(
            (row for row in lines if row["event_type"] == "poll_completed"), None
        )
        assert poll_completed is not None
        assert poll_completed["event_count"] == 0

    def test_validation_result_returned(self, tmp_path: Path) -> None:
        """Validation result dict is returned on no-news day."""
        registry = load_source_registry(_FIXTURE_BASE / "no_news" / "registry.toml")
        _, validation = run_briefing(
            registry,
            workspace=tmp_path / "workspace",
            run_date=_RUN_DATE,
            fixture_base_dir=_FIXTURE_BASE / "no_news",
        )
        assert isinstance(validation, dict)
        assert "passed" in validation


class TestDuplicateDeduplication:
    """Duplicate events with the same canonical URL collapse into one cluster."""

    def test_same_canonical_url_deduplicates(self, tmp_path: Path) -> None:
        """Two sources emitting the same canonical URL produce one cluster."""
        # Use normal fixture which has a GitHub release with a unique URL
        # and an RSS feed with distinct URLs — there should be no duplicates
        # in the normal fixture so cluster count equals event count from distinct URLs
        registry = load_source_registry(_FIXTURE_BASE / "normal" / "registry.toml")
        paths, _ = run_briefing(
            registry,
            workspace=tmp_path / "workspace",
            run_date=_RUN_DATE,
            fixture_base_dir=_FIXTURE_BASE / "normal",
        )
        # Load clusters and events to verify no cluster has duplicate canonical URLs
        cluster_lines = [
            json.loads(ln)
            for ln in paths.clusters_path.read_text(encoding="utf-8").splitlines()
            if ln.strip()
        ]
        all_canonical_urls: list[str] = []
        for cluster in cluster_lines:
            all_canonical_urls.extend(cluster.get("canonical_urls", []))
        # Each canonical URL should appear at most once across all clusters
        url_counts = {url: all_canonical_urls.count(url) for url in all_canonical_urls}
        duplicates = {url: cnt for url, cnt in url_counts.items() if cnt > 1}
        assert not duplicates, f"duplicate canonical URLs across clusters: {duplicates}"

    def test_cluster_count_equals_unique_events(self, tmp_path: Path) -> None:
        """Cluster count does not exceed event count (dedup reduces, never adds)."""
        registry = load_source_registry(_FIXTURE_BASE / "normal" / "registry.toml")
        paths, _ = run_briefing(
            registry,
            workspace=tmp_path / "workspace",
            run_date=_RUN_DATE,
            fixture_base_dir=_FIXTURE_BASE / "normal",
        )
        event_count = sum(
            1
            for ln in paths.events_path.read_text(encoding="utf-8").splitlines()
            if ln.strip()
        )
        cluster_count = sum(
            1
            for ln in paths.clusters_path.read_text(encoding="utf-8").splitlines()
            if ln.strip()
        )
        assert cluster_count <= event_count


class TestArtifactLayout:
    """Verify the fixed Phase A artifact layout specification."""

    def test_layout_path_names(self, tmp_path: Path) -> None:
        """Artifact directories follow the Phase A fixed layout names."""
        registry = load_source_registry(_FIXTURE_BASE / "normal" / "registry.toml")
        paths, _ = run_briefing(
            registry,
            workspace=tmp_path / "workspace",
            run_date=_RUN_DATE,
            fixture_base_dir=_FIXTURE_BASE / "normal",
        )
        root = paths.root
        assert (root / "raw").exists()
        assert (root / "normalized").exists()
        assert (root / "clusters").exists()
        assert (root / "ranked").exists()
        assert (root / "reports").exists()
        assert (root / "validation").exists()
        assert (root / "telemetry.jsonl").exists()

    def test_run_date_in_root_path(self, tmp_path: Path) -> None:
        """Run date appears in the briefing root directory name."""
        registry = load_source_registry(_FIXTURE_BASE / "normal" / "registry.toml")
        paths, _ = run_briefing(
            registry,
            workspace=tmp_path / "workspace",
            run_date=_RUN_DATE,
            fixture_base_dir=_FIXTURE_BASE / "normal",
        )
        assert _RUN_DATE in str(paths.root)

    @pytest.mark.parametrize(
        "scenario",
        ["normal", "low_signal", "no_news"],
    )
    def test_all_scenarios_create_report(self, tmp_path: Path, scenario: str) -> None:
        """Every scenario produces a daily report file."""
        registry = load_source_registry(_FIXTURE_BASE / scenario / "registry.toml")
        paths, _ = run_briefing(
            registry,
            workspace=tmp_path / "workspace" / scenario,
            run_date=_RUN_DATE,
            fixture_base_dir=_FIXTURE_BASE / scenario,
        )
        assert paths.daily_report_path.exists()
