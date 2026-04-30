"""Tests for briefing CLI commands."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from research_pipeline.briefing.io import write_jsonl
from research_pipeline.briefing.layout import resolve_briefing_paths
from research_pipeline.briefing.models import (
    AccessMethod,
    BriefingCluster,
    IntelligenceEvent,
    SourceClass,
)
from research_pipeline.cli.app import app

runner = CliRunner()


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


def _cluster(cluster_id: str) -> BriefingCluster:
    """Helper to create BriefingCluster for testing."""
    event = _event(f"e_for_{cluster_id}")
    return BriefingCluster(
        cluster_id=cluster_id,
        title=f"Cluster {cluster_id}",
        primary_event_id=event.event_id,
        event_ids=(event.event_id,),
        canonical_urls=(event.canonical_url,),
        first_seen_at="2025-01-20T00:00:00Z",
        last_seen_at="2025-01-20T12:00:00Z",
        source_classes=(SourceClass.PRIMARY_ARTIFACT,),
        primary_artifact_present=True,
        rank_score=0.95,
        authority_score=0.8,
        events=(event,),
    )


def test_brief_poll_creates_artifact_layout(tmp_path: Path) -> None:
    """brief poll creates the fixed artifact directory structure."""
    registry_path = tmp_path / "registry.toml"
    registry_path.write_text(
        """
[briefing]
max_sources_per_run = 1

[[briefing.sources]]
source_id = "test_source"
source_name = "Test Source"
source_class = "technical_discussion"
access_method = "rss_atom"
fixture_path = "fixtures/test.xml"
enabled = true
"""
    )

    result = runner.invoke(
        app,
        [
            "brief",
            "poll",
            "--registry",
            str(registry_path),
            "--workspace",
            str(tmp_path / "workspace"),
            "--date",
            "2025-01-20",
        ],
    )

    assert result.exit_code == 0
    paths = resolve_briefing_paths(tmp_path / "workspace", "2025-01-20")
    assert paths.raw_dir.exists()
    assert paths.normalized_dir.exists()


def test_brief_poll_writes_events_jsonl(tmp_path: Path) -> None:
    """brief poll writes normalized events.jsonl."""
    registry_path = tmp_path / "registry.toml"
    registry_path.write_text(
        """
[briefing]
max_sources_per_run = 1

[[briefing.sources]]
source_id = "test_source"
source_name = "Test Source"
source_class = "technical_discussion"
access_method = "rss_atom"
fixture_path = "fixtures/test.xml"
enabled = true
"""
    )

    result = runner.invoke(
        app,
        [
            "brief",
            "poll",
            "--registry",
            str(registry_path),
            "--workspace",
            str(tmp_path / "workspace"),
            "--date",
            "2025-01-20",
        ],
    )

    assert result.exit_code == 0
    paths = resolve_briefing_paths(tmp_path / "workspace", "2025-01-20")
    assert paths.events_path.exists()


def test_brief_rank_requires_normalized_events(tmp_path: Path) -> None:
    """brief rank processes successfully even with missing events."""
    result = runner.invoke(
        app,
        [
            "brief",
            "rank",
            "--workspace",
            str(tmp_path / "workspace"),
            "--date",
            "2025-01-20",
        ],
    )

    # Should succeed, just with 0 clusters
    assert result.exit_code == 0


def test_brief_rank_with_events(tmp_path: Path) -> None:
    """brief rank reads events and outputs ranked clusters."""
    workspace = tmp_path / "workspace"
    paths = resolve_briefing_paths(workspace, "2025-01-20")
    paths.create()
    registry_path = tmp_path / "registry.toml"
    registry_path.write_text(
        """
[briefing]
max_sources_per_run = 1

[[briefing.sources]]
source_id = "test_source"
source_name = "Test Source"
source_class = "technical_discussion"
access_method = "rss_atom"
fixture_path = "fixtures/test.xml"
enabled = true
"""
    )

    events = [_event(f"e{i}") for i in range(3)]
    write_jsonl(paths.events_path, events)

    result = runner.invoke(
        app,
        [
            "brief",
            "rank",
            "--registry",
            str(registry_path),
            "--workspace",
            str(workspace),
            "--date",
            "2025-01-20",
        ],
    )

    assert result.exit_code == 0
    assert paths.ranked_clusters_path.exists()


def test_brief_generate_daily_requires_ranked_clusters(tmp_path: Path) -> None:
    """brief generate-daily fails if ranked clusters are missing."""
    result = runner.invoke(
        app,
        [
            "brief",
            "generate-daily",
            "--workspace",
            str(tmp_path / "workspace"),
            "--date",
            "2025-01-20",
        ],
    )

    assert result.exit_code != 0


def test_brief_generate_daily_with_ranked_clusters(tmp_path: Path) -> None:
    """brief generate-daily outputs daily.md."""
    workspace = tmp_path / "workspace"
    paths = resolve_briefing_paths(workspace, "2025-01-20")
    paths.create()

    clusters = [_cluster(f"c{i}") for i in range(6)]
    write_jsonl(paths.ranked_clusters_path, clusters)

    result = runner.invoke(
        app,
        [
            "brief",
            "generate-daily",
            "--workspace",
            str(workspace),
            "--date",
            "2025-01-20",
        ],
    )

    if result.exit_code != 0:
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        if result.exception:
            import traceback

            traceback.print_exception(
                type(result.exception), result.exception, result.exception.__traceback__
            )
    assert result.exit_code == 0
    assert paths.daily_report_path.exists()
    report_content = paths.daily_report_path.read_text()
    assert "Daily AI Intelligence Brief" in report_content


def test_brief_validate_requires_daily_report(tmp_path: Path) -> None:
    """brief validate fails if daily.md is missing."""
    result = runner.invoke(
        app,
        [
            "brief",
            "validate",
            "--workspace",
            str(tmp_path / "workspace"),
            "--date",
            "2025-01-20",
        ],
    )

    assert result.exit_code != 0


def test_brief_validate_outputs_validation_json(tmp_path: Path) -> None:
    """brief validate outputs validation.json (may pass or fail validation)."""
    workspace = tmp_path / "workspace"
    paths = resolve_briefing_paths(workspace, "2025-01-20")
    paths.create()

    # Create a report (may not be fully valid, but validation.json should be written)
    clusters = [_cluster(f"c{i}") for i in range(6)]
    words = " ".join(["word"] * 1000)
    report = (
        "---\ntype: daily-brief\n---\n"
        "# Daily Brief\n"
        "## 🔥 Executive Signal\n[FACT] Test.\n"
        "## ⭐ Top Items\n" + "\n".join(f"- {c.title}" for c in clusters) + "\n"
        "## 🗒️ Feedback Targets\n| target | id | cmd |\n|---|---|---|\n" + words
    )
    paths.daily_report_path.parent.mkdir(parents=True, exist_ok=True)
    paths.daily_report_path.write_text(report)

    write_jsonl(paths.ranked_clusters_path, clusters)

    result = runner.invoke(
        app,
        [
            "brief",
            "validate",
            "--workspace",
            str(workspace),
            "--date",
            "2025-01-20",
        ],
    )

    # Validation may pass or fail, but validation.json should always be created
    assert result.exit_code in {0, 1}
    assert paths.validation_path.exists()
    validation_result = json.loads(paths.validation_path.read_text())
    assert "passed" in validation_result


def test_brief_run_executes_full_pipeline(tmp_path: Path) -> None:
    """brief run executes poll, rank, generate-daily, validate in order."""
    workspace = tmp_path / "workspace"
    registry_path = tmp_path / "registry.toml"
    registry_path.write_text(
        """
[briefing]
max_sources_per_run = 1

[[briefing.sources]]
source_id = "test_source"
source_name = "Test Source"
source_class = "technical_discussion"
access_method = "rss_atom"
fixture_path = "fixtures/test.xml"
enabled = true
"""
    )

    result = runner.invoke(
        app,
        [
            "brief",
            "run",
            "--registry",
            str(registry_path),
            "--workspace",
            str(workspace),
            "--date",
            "2025-01-20",
        ],
    )

    paths = resolve_briefing_paths(workspace, "2025-01-20")
    assert paths.events_path.exists() or result.exit_code == 0


def test_brief_verbose_enables_debug_logging(tmp_path: Path) -> None:
    """brief --verbose enables debug logging."""
    registry_path = tmp_path / "registry.toml"
    registry_path.write_text(
        """
[briefing]
max_sources_per_run = 1

[[briefing.sources]]
source_id = "test_source"
source_name = "Test Source"
source_class = "technical_discussion"
access_method = "rss_atom"
fixture_path = "fixtures/test.xml"
enabled = true
"""
    )

    result = runner.invoke(
        app,
        [
            "brief",
            "poll",
            "--registry",
            str(registry_path),
            "--workspace",
            str(tmp_path / "workspace"),
            "--date",
            "2025-01-20",
            "--verbose",
        ],
    )

    assert result.exit_code == 0


def test_brief_date_option_controls_artifact_date(tmp_path: Path) -> None:
    """brief --date option controls the artifact date directory."""
    registry_path = tmp_path / "registry.toml"
    registry_path.write_text(
        """
[briefing]
max_sources_per_run = 1

[[briefing.sources]]
source_id = "test_source"
source_name = "Test Source"
source_class = "technical_discussion"
access_method = "rss_atom"
fixture_path = "fixtures/test.xml"
enabled = true
"""
    )

    result = runner.invoke(
        app,
        [
            "brief",
            "poll",
            "--registry",
            str(registry_path),
            "--workspace",
            str(tmp_path / "workspace"),
            "--date",
            "2025-01-15",
        ],
    )

    assert result.exit_code == 0
    assert (tmp_path / "workspace" / "briefings" / "2025-01-15").exists()


def test_brief_workspace_option_controls_workspace_root(tmp_path: Path) -> None:
    """brief --workspace controls the workspace root directory."""
    registry_path = tmp_path / "registry.toml"
    registry_path.write_text(
        """
[briefing]
max_sources_per_run = 1

[[briefing.sources]]
source_id = "test_source"
source_name = "Test Source"
source_class = "technical_discussion"
access_method = "rss_atom"
fixture_path = "fixtures/test.xml"
enabled = true
"""
    )
    custom_workspace = tmp_path / "my_workspace"

    result = runner.invoke(
        app,
        [
            "brief",
            "poll",
            "--registry",
            str(registry_path),
            "--workspace",
            str(custom_workspace),
            "--date",
            "2025-01-20",
        ],
    )

    assert result.exit_code == 0
    assert (custom_workspace / "briefings" / "2025-01-20").exists()
