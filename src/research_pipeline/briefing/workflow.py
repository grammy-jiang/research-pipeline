"""End-to-end briefing workflow functions used by CLI and MCP adapters."""

from __future__ import annotations

import json
import time
from pathlib import Path

from research_pipeline.briefing.dedup import cluster_events
from research_pipeline.briefing.feedback import BriefingFeedbackStore
from research_pipeline.briefing.io import read_json, read_jsonl, write_json, write_jsonl
from research_pipeline.briefing.layout import BriefingPaths, resolve_briefing_paths
from research_pipeline.briefing.models import (
    AccessMethod,
    BriefingCluster,
    BriefingSourceConfig,
    IntelligenceEvent,
)
from research_pipeline.briefing.rank import RankingOptions, rank_clusters
from research_pipeline.briefing.registry import (
    SourceRegistry,
    assert_phase_a_source_boundary,
)
from research_pipeline.briefing.report import render_daily_brief
from research_pipeline.briefing.sources.arxiv_events import ArxivEventsSource
from research_pipeline.briefing.sources.base import BriefingSource
from research_pipeline.briefing.sources.bluesky import BlueskySource
from research_pipeline.briefing.sources.github_releases import GitHubReleasesSource
from research_pipeline.briefing.sources.hacker_news import HackerNewsSource
from research_pipeline.briefing.sources.html_scrape import HtmlScrapeSource
from research_pipeline.briefing.sources.huggingface_papers import (
    HuggingFacePapersSource,
)
from research_pipeline.briefing.sources.manual import ManualSource
from research_pipeline.briefing.sources.papers import PaperEventsSource
from research_pipeline.briefing.sources.reddit import RedditSource
from research_pipeline.briefing.sources.rss_atom import RssAtomSource
from research_pipeline.briefing.sources.video_audio import VideoAudioSource
from research_pipeline.briefing.sources.x_api import XApiSource
from research_pipeline.briefing.telemetry import BriefingTelemetry
from research_pipeline.briefing.topic_memory import TopicMemoryStore
from research_pipeline.briefing.validate import (
    validate_daily_report,
    validation_to_json,
)
from research_pipeline.briefing.workflow_state import advance_workflow_state


def poll_sources(
    registry: SourceRegistry,
    *,
    workspace: Path | None,
    run_date: str | None,
    fixture_base_dir: Path | None = None,
) -> tuple[BriefingPaths, list[IntelligenceEvent]]:
    """Poll enabled sources and write raw plus normalized artifacts."""
    paths = resolve_briefing_paths(workspace, run_date)
    paths.create()
    telemetry = BriefingTelemetry(paths.telemetry_path)
    registry.snapshot(paths.source_snapshot_path)
    start = time.monotonic()
    events: list[IntelligenceEvent] = []
    for source in registry.enabled_sources():
        assert_phase_a_source_boundary(source)
        raw_path = paths.raw_dir / f"{source.source_id}.jsonl"
        try:
            state = _load_http_state(paths, source)
            source_events = _adapter_for(source, state, fixture_base_dir).poll()
            _write_http_state(paths, source, state)
            write_jsonl(raw_path, source_events)
            events.extend(source_events)
            telemetry.emit(
                "source_polled",
                source_id=source.source_id,
                event_count=len(source_events),
                status="ok",
            )
        except Exception as exc:
            write_jsonl(raw_path, [])
            telemetry.emit(
                "source_polled",
                source_id=source.source_id,
                event_count=0,
                status="error",
                error=str(exc),
            )
    write_jsonl(paths.events_path, events)
    telemetry.emit(
        "poll_completed",
        event_count=len(events),
        duration_seconds=round(time.monotonic() - start, 3),
    )
    advance_workflow_state(
        paths.root,
        run_date=paths.root.name,
        stage="polled",
        artifacts={"events": str(paths.events_path)},
    )
    return paths, events


def rank_events(
    paths: BriefingPaths,
    registry: SourceRegistry,
    *,
    use_memory: bool = True,
    use_feedback: bool = True,
) -> list[BriefingCluster]:
    """Deduplicate and rank normalized events."""
    events = read_jsonl(paths.events_path, IntelligenceEvent)
    clusters = cluster_events(list(events))
    write_jsonl(paths.clusters_path, clusters)
    source_weights = {
        source.source_id: (source.trust_weight, source.noise_weight)
        for source in registry.sources
    }
    memory_store = (
        TopicMemoryStore(paths.root / "memory" / "topics.db") if use_memory else None
    )
    feedback_store = (
        BriefingFeedbackStore(paths.root / "feedback" / "feedback.db")
        if use_feedback
        else None
    )
    try:
        feedback_weights = feedback_store.weights_by_target() if feedback_store else {}
        ranked = rank_clusters(
            clusters,
            source_weights=source_weights,
            options=RankingOptions(
                watchlist_terms=registry.watchlist_terms,
                feedback_weights=feedback_weights,
                topic_memory=memory_store,
            ),
        )
        if memory_store is not None:
            memory_store.upsert_from_clusters(ranked, paths.root.name)
        write_jsonl(paths.ranked_clusters_path, ranked)
        advance_workflow_state(
            paths.root,
            run_date=paths.root.name,
            stage="ranked",
            artifacts={"ranked_clusters": str(paths.ranked_clusters_path)},
        )
        BriefingTelemetry(paths.telemetry_path).emit(
            "rank_completed",
            cluster_count=len(clusters),
            ranked_count=len(ranked),
        )
        return ranked
    finally:
        if memory_store is not None:
            memory_store.close()
        if feedback_store is not None:
            feedback_store.close()


def generate_daily(paths: BriefingPaths, *, run_date: str | None = None) -> str:
    """Generate the daily Markdown report from ranked clusters."""
    clusters = read_jsonl(paths.ranked_clusters_path, BriefingCluster)
    registry = (
        load_registry_snapshot(paths) if paths.source_snapshot_path.exists() else None
    )
    quiet_sources = _quiet_sources(list(clusters), registry) if registry else []
    previous_brief_link = _previous_brief_link(paths)
    markdown = render_daily_brief(
        list(clusters),
        run_date=run_date or paths.root.name,
        quiet_sources=quiet_sources,
        previous_brief_link=previous_brief_link,
    )
    paths.daily_report_path.write_text(markdown, encoding="utf-8")
    advance_workflow_state(
        paths.root,
        run_date=paths.root.name,
        stage="generated",
        artifacts={"daily_report": str(paths.daily_report_path)},
    )
    BriefingTelemetry(paths.telemetry_path).emit(
        "daily_generated",
        path=str(paths.daily_report_path),
        item_count=len(clusters),
    )
    return markdown


def validate_daily(paths: BriefingPaths) -> dict[str, object]:
    """Validate the generated daily report and write validation JSON."""
    markdown = paths.daily_report_path.read_text(encoding="utf-8")
    clusters = read_jsonl(paths.ranked_clusters_path, BriefingCluster)
    result = validate_daily_report(markdown, list(clusters))
    payload = validation_to_json(result)
    if result.passed:
        markdown = markdown.replace("status: draft", "status: validated", 1)
        paths.daily_report_path.write_text(markdown, encoding="utf-8")
    write_json(paths.validation_path, payload)
    advance_workflow_state(
        paths.root,
        run_date=paths.root.name,
        stage="validated" if result.passed else "failed",
        artifacts={"validation": str(paths.validation_path)},
    )
    BriefingTelemetry(paths.telemetry_path).emit(
        "daily_validated",
        passed=result.passed,
        error_count=len(result.errors),
        warning_count=len(result.warnings),
    )
    return payload


def run_briefing(
    registry: SourceRegistry,
    *,
    workspace: Path | None,
    run_date: str | None,
    fixture_base_dir: Path | None = None,
) -> tuple[BriefingPaths, dict[str, object]]:
    """Run Phase A-G deterministic briefing workflow surfaces."""
    paths, _events = poll_sources(
        registry,
        workspace=workspace,
        run_date=run_date,
        fixture_base_dir=fixture_base_dir,
    )
    rank_events(paths, registry)
    generate_daily(paths, run_date=run_date)
    validation = validate_daily(paths)
    return paths, validation


def resume_briefing(
    registry: SourceRegistry,
    *,
    workspace: Path | None,
    run_date: str,
    from_stage: str,
) -> tuple[BriefingPaths, dict[str, object] | None]:
    """Resume a briefing run from an existing artifact stage."""
    paths = resolve_briefing_paths(workspace, run_date)
    if from_stage == "rank":
        rank_events(paths, registry)
        generate_daily(paths, run_date=run_date)
        return paths, validate_daily(paths)
    if from_stage == "generate-daily":
        generate_daily(paths, run_date=run_date)
        return paths, validate_daily(paths)
    if from_stage == "validate":
        return paths, validate_daily(paths)
    raise ValueError("from_stage must be rank, generate-daily, or validate")


def load_registry_snapshot(paths: BriefingPaths) -> SourceRegistry:
    """Load the run's source registry snapshot."""
    return SourceRegistry.model_validate(read_json(paths.source_snapshot_path))


def _adapter_for(
    source: BriefingSourceConfig,
    state: dict[str, str],
    fixture_base_dir: Path | None,
) -> BriefingSource:
    if source.access_method == AccessMethod.GITHUB_RELEASES:
        return GitHubReleasesSource(
            source, state=state, fixture_base_dir=fixture_base_dir
        )
    if source.access_method == AccessMethod.RSS_ATOM:
        return RssAtomSource(source, state=state, fixture_base_dir=fixture_base_dir)
    if source.access_method == AccessMethod.HTML_SCRAPE:
        return HtmlScrapeSource(source, state=state, fixture_base_dir=fixture_base_dir)
    if source.access_method == AccessMethod.MANUAL:
        return ManualSource(source)
    if source.access_method == AccessMethod.HACKER_NEWS:
        return HackerNewsSource(source, fixture_base_dir=fixture_base_dir)
    if source.access_method == AccessMethod.HUGGINGFACE_PAPERS:
        if source.fixture_path and source.fixture_path.endswith(".json"):
            return PaperEventsSource(source, fixture_base_dir=fixture_base_dir)
        return HuggingFacePapersSource(source, fixture_base_dir=fixture_base_dir)
    if source.access_method == AccessMethod.ARXIV:
        if source.fixture_path and source.fixture_path.endswith((".jsonl", ".json")):
            return PaperEventsSource(source, fixture_base_dir=fixture_base_dir)
        return ArxivEventsSource(source, fixture_base_dir=fixture_base_dir)
    if source.access_method == AccessMethod.REDDIT_API:
        return RedditSource(source, fixture_base_dir=fixture_base_dir)
    if source.access_method == AccessMethod.BLUESKY_API:
        return BlueskySource(source, fixture_base_dir=fixture_base_dir)
    if source.access_method == AccessMethod.VIDEO_AUDIO:
        return VideoAudioSource(source, fixture_base_dir=fixture_base_dir)
    if source.access_method == AccessMethod.X_API:
        return XApiSource(source, fixture_base_dir=fixture_base_dir)
    raise ValueError(
        f"unsupported briefing source access method: {source.access_method}"
    )


def _http_state_path(paths: BriefingPaths, source: BriefingSourceConfig) -> Path:
    return paths.root / "http_state" / f"{source.source_id}.json"


def _load_http_state(
    paths: BriefingPaths, source: BriefingSourceConfig
) -> dict[str, str]:
    path = _http_state_path(paths, source)
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {str(key): str(value) for key, value in data.items()}


def _write_http_state(
    paths: BriefingPaths, source: BriefingSourceConfig, state: dict[str, str]
) -> None:
    path = _http_state_path(paths, source)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _previous_brief_link(paths: BriefingPaths) -> str | None:
    """Return a relative-path link to the previous day's daily brief, if any.

    Looks at sibling directories of ``paths.root`` whose name sorts before the
    current run-date directory and points at an existing ``reports/daily.md``
    file. Returns ``None`` when none can be located.
    """
    parent = paths.root.parent
    if not parent.is_dir():
        return None
    current = paths.root.name
    candidates = sorted(
        (
            entry
            for entry in parent.iterdir()
            if entry.is_dir() and entry.name < current
        ),
        key=lambda entry: entry.name,
        reverse=True,
    )
    for prior in candidates:
        prior_report = prior / "reports" / "daily.md"
        if prior_report.is_file():
            return f"../../{prior.name}/reports/daily.md"
    return None


def _quiet_sources(
    clusters: list[BriefingCluster], registry: SourceRegistry
) -> list[str]:
    """List high-trust enabled sources with no ranked event."""
    active_source_ids = {
        event.source_id for cluster in clusters for event in cluster.events
    }
    return [
        f"{source.source_name} (`{source.source_id}`)"
        for source in registry.enabled_sources()
        if source.trust_weight >= 3.0 and source.source_id not in active_source_ids
    ]
