"""CLI command group for daily AI intelligence briefings."""

from __future__ import annotations

import logging
from pathlib import Path

import typer

from research_pipeline.briefing.dossier import (
    build_dossier,
    render_dossier,
    select_dossier_candidates,
    write_dossier,
)
from research_pipeline.briefing.feedback import BriefingFeedbackStore
from research_pipeline.briefing.io import read_jsonl
from research_pipeline.briefing.layout import resolve_briefing_paths
from research_pipeline.briefing.models import BriefingCluster, FeedbackSignal
from research_pipeline.briefing.obsidian import (
    export_daily_note,
    export_source_notes,
    export_topic_notes,
)
from research_pipeline.briefing.preference_update import rollback_preference_adjustment
from research_pipeline.briefing.registry import SourceRegistry, load_source_registry
from research_pipeline.briefing.report import render_weekly_synthesis
from research_pipeline.briefing.topic_memory import TopicMemoryStore
from research_pipeline.briefing.validate import validate_dossier_report
from research_pipeline.briefing.workflow import (
    generate_daily,
    load_registry_snapshot,
    poll_sources,
    rank_events,
    resume_briefing,
    run_briefing,
    validate_daily,
)
from research_pipeline.briefing.workflow_state import advance_workflow_state
from research_pipeline.infra.logging import setup_logging

logger = logging.getLogger(__name__)

brief_app = typer.Typer(
    name="brief",
    help="Daily AI technical intelligence briefing commands.",
    no_args_is_help=True,
)


def _setup(verbose: bool) -> None:
    setup_logging(level=logging.DEBUG if verbose else logging.INFO)


def _registry_or_snapshot(
    registry_path: Path | None, workspace: Path | None, date: str | None
) -> SourceRegistry:
    if registry_path is not None:
        return load_source_registry(registry_path)
    paths = resolve_briefing_paths(workspace, date)
    if paths.source_snapshot_path.exists():
        return load_registry_snapshot(paths)
    return load_source_registry(None)


@brief_app.command("poll")
def poll_command(
    registry: Path | None = typer.Option(
        None,
        "--registry",
        "-r",
        help="Path to briefing source registry JSON/TOML.",
    ),
    workspace: Path | None = typer.Option(
        None, "--workspace", "-w", help="Workspace root."
    ),
    date: str | None = typer.Option(None, "--date", help="Briefing date YYYY-MM-DD."),
    fixture_base_dir: Path | None = typer.Option(
        None,
        "--fixture-base-dir",
        help="Base directory for registry fixture_path values.",
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable debug logging."
    ),
) -> None:
    """Poll configured briefing sources and write raw/normalized artifacts."""
    _setup(verbose)
    source_registry = load_source_registry(registry)
    paths, events = poll_sources(
        source_registry,
        workspace=workspace,
        run_date=date,
        fixture_base_dir=fixture_base_dir,
    )
    logger.info("Polled %d events into %s", len(events), paths.root)


@brief_app.command("rank")
def rank_command(
    registry: Path | None = typer.Option(
        None, "--registry", "-r", help="Source registry."
    ),
    workspace: Path | None = typer.Option(
        None, "--workspace", "-w", help="Workspace root."
    ),
    date: str | None = typer.Option(None, "--date", help="Briefing date YYYY-MM-DD."),
    no_memory: bool = typer.Option(False, "--no-memory", help="Disable topic memory."),
    no_feedback: bool = typer.Option(
        False, "--no-feedback", help="Disable feedback weights."
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable debug logging."
    ),
) -> None:
    """Deduplicate and deterministically rank normalized events."""
    _setup(verbose)
    paths = resolve_briefing_paths(workspace, date)
    source_registry = _registry_or_snapshot(registry, workspace, date)
    ranked = rank_events(
        paths,
        source_registry,
        use_memory=not no_memory,
        use_feedback=not no_feedback,
    )
    logger.info("Ranked %d clusters into %s", len(ranked), paths.ranked_clusters_path)


@brief_app.command("generate-daily")
def generate_daily_command(
    workspace: Path | None = typer.Option(
        None, "--workspace", "-w", help="Workspace root."
    ),
    date: str | None = typer.Option(None, "--date", help="Briefing date YYYY-MM-DD."),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable debug logging."
    ),
) -> None:
    """Generate the daily Markdown brief from ranked clusters."""
    _setup(verbose)
    paths = resolve_briefing_paths(workspace, date)
    generate_daily(paths, run_date=date)
    logger.info("Generated daily brief at %s", paths.daily_report_path)


@brief_app.command("validate")
def validate_command(
    workspace: Path | None = typer.Option(
        None, "--workspace", "-w", help="Workspace root."
    ),
    date: str | None = typer.Option(None, "--date", help="Briefing date YYYY-MM-DD."),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable debug logging."
    ),
) -> None:
    """Validate the generated daily brief."""
    _setup(verbose)
    paths = resolve_briefing_paths(workspace, date)
    result = validate_daily(paths)
    logger.info("Validation result written to %s", paths.validation_path)
    if not bool(result["passed"]):
        raise typer.Exit(1)


@brief_app.command("run")
def run_command(
    registry: Path | None = typer.Option(
        None,
        "--registry",
        "-r",
        help="Path to briefing source registry JSON/TOML.",
    ),
    workspace: Path | None = typer.Option(
        None, "--workspace", "-w", help="Workspace root."
    ),
    date: str | None = typer.Option(None, "--date", help="Briefing date YYYY-MM-DD."),
    fixture_base_dir: Path | None = typer.Option(
        None,
        "--fixture-base-dir",
        help="Base directory for registry fixture_path values.",
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable debug logging."
    ),
) -> None:
    """Run poll, rank, generate-daily, and validate in order."""
    _setup(verbose)
    paths, validation = run_briefing(
        load_source_registry(registry),
        workspace=workspace,
        run_date=date,
        fixture_base_dir=fixture_base_dir,
    )
    logger.info("Briefing artifacts written to %s", paths.root)
    if not bool(validation["passed"]):
        raise typer.Exit(1)


@brief_app.command("feedback")
def feedback_command(
    signal: FeedbackSignal = typer.Option(
        ..., "--signal", help="Explicit feedback signal."
    ),
    cluster: str | None = typer.Option(None, "--cluster", help="Cluster ID target."),
    topic: str | None = typer.Option(None, "--topic", help="Topic ID target."),
    source: str | None = typer.Option(None, "--source", help="Source ID target."),
    event: str | None = typer.Option(None, "--event", help="Event ID target."),
    dossier: str | None = typer.Option(None, "--dossier", help="Dossier ID target."),
    reason: str = typer.Option("", "--reason", help="Optional feedback reason."),
    strength: float = typer.Option(1.0, "--strength", help="Signal strength 0-5."),
    show: bool = typer.Option(
        False, "--show", help="List feedback instead of recording."
    ),
    conflicts: bool = typer.Option(
        False, "--conflicts", help="List conflicting feedback."
    ),
    workspace: Path | None = typer.Option(
        None, "--workspace", "-w", help="Workspace root."
    ),
    date: str | None = typer.Option(None, "--date", help="Briefing date YYYY-MM-DD."),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable debug logging."
    ),
) -> None:
    """Record explicit feedback on a cluster, topic, source, event, or dossier."""
    _setup(verbose)
    paths = resolve_briefing_paths(workspace, date)
    store = BriefingFeedbackStore(paths.root / "feedback" / "feedback.db")
    if show or conflicts:
        try:
            if conflicts:
                for target, counts in store.conflict_summary().items():
                    logger.info("Conflict %s: %s", target, counts)
            else:
                for item in store.list_feedback():
                    logger.info(
                        "%s %s:%s %s",
                        item.timestamp,
                        item.target_type,
                        item.target_id,
                        item.signal_type.value,
                    )
        finally:
            store.close()
        return
    targets = {
        "cluster": cluster,
        "topic": topic,
        "source": source,
        "event": event,
        "dossier": dossier,
    }
    selected = [
        (target_type, target_id)
        for target_type, target_id in targets.items()
        if target_id
    ]
    if len(selected) != 1:
        raise typer.BadParameter("provide exactly one feedback target")
    try:
        feedback = store.record(
            target_type=selected[0][0],
            target_id=selected[0][1],
            signal=signal,
            strength=strength,
            reason=reason,
        )
        logger.info("Recorded feedback %s", feedback.feedback_id)
    finally:
        store.close()


@brief_app.command("topic-aliases")
def topic_aliases_command(
    approve: str | None = typer.Option(
        None, "--approve", help="Approve suggestion ID."
    ),
    reject: str | None = typer.Option(None, "--reject", help="Reject suggestion ID."),
    review: str = typer.Option("", "--review", help="Review note for approve/reject."),
    all_statuses: bool = typer.Option(False, "--all", help="Show all statuses."),
    workspace: Path | None = typer.Option(
        None, "--workspace", "-w", help="Workspace root."
    ),
    date: str | None = typer.Option(None, "--date", help="Briefing date YYYY-MM-DD."),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable debug logging."
    ),
) -> None:
    """List, approve, or reject reviewable topic alias suggestions."""
    _setup(verbose)
    paths = resolve_briefing_paths(workspace, date)
    store = TopicMemoryStore(paths.root / "memory" / "topics.db")
    try:
        if approve and reject:
            raise typer.BadParameter("use only one of --approve or --reject")
        if approve or reject:
            suggestion = store.review_alias_suggestion(
                approve or reject or "",
                approve=approve is not None,
                review_record=review or "reviewed via CLI",
            )
            logger.info(
                "%s alias suggestion %s",
                suggestion.status,
                suggestion.suggestion_id,
            )
            return
        for suggestion in store.list_alias_suggestions(
            status=None if all_statuses else "pending"
        ):
            logger.info(
                "%s %s -> %s (%s)",
                suggestion.status,
                suggestion.topic_id,
                suggestion.suggested_alias,
                suggestion.suggestion_id,
            )
    finally:
        store.close()


@brief_app.command("export-obsidian")
def export_obsidian_command(
    vault: Path = typer.Option(..., "--vault", help="Configured Obsidian vault root."),
    registry: Path | None = typer.Option(
        None, "--registry", "-r", help="Source registry."
    ),
    workspace: Path | None = typer.Option(
        None, "--workspace", "-w", help="Workspace root."
    ),
    date: str | None = typer.Option(None, "--date", help="Briefing date YYYY-MM-DD."),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable debug logging."
    ),
) -> None:
    """Export daily, topic, and source notes to an Obsidian vault."""
    _setup(verbose)
    paths = resolve_briefing_paths(workspace, date)
    markdown = paths.daily_report_path.read_text(encoding="utf-8")
    clusters = read_jsonl(paths.ranked_clusters_path, BriefingCluster)
    source_registry = _registry_or_snapshot(registry, workspace, date)
    changed = [
        export_daily_note(markdown, vault_root=vault, run_date=date or paths.root.name),
        *export_topic_notes(list(clusters), vault_root=vault),
        *export_source_notes(list(source_registry.sources), vault_root=vault),
    ]
    advance_workflow_state(
        paths.root,
        run_date=paths.root.name,
        stage="archived",
        artifacts={"obsidian_notes": ",".join(str(path) for path in changed)},
    )
    logger.info("Exported %d Obsidian notes under %s", len(changed), vault)


@brief_app.command("dossier")
def dossier_command(
    cluster: str | None = typer.Option(None, "--cluster", help="Cluster ID to expand."),
    auto: bool = typer.Option(
        False, "--auto", help="Auto-select top dossier candidates."
    ),
    max_count: int = typer.Option(1, "--max-count", help="Max automatic dossiers."),
    workspace: Path | None = typer.Option(
        None, "--workspace", "-w", help="Workspace root."
    ),
    date: str | None = typer.Option(None, "--date", help="Briefing date YYYY-MM-DD."),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable debug logging."
    ),
) -> None:
    """Generate a manual hot-topic dossier for one ranked cluster."""
    _setup(verbose)
    paths = resolve_briefing_paths(workspace, date)
    clusters = read_jsonl(paths.ranked_clusters_path, BriefingCluster)
    selected = (
        select_dossier_candidates(list(clusters), max_count=max_count) if auto else []
    )
    if not auto:
        if cluster is None:
            raise typer.BadParameter("provide --cluster or --auto")
        match = next((item for item in clusters if item.cluster_id == cluster), None)
        if match is None:
            raise typer.BadParameter(f"cluster not found: {cluster}")
        selected = [match]
    for item in selected:
        dossier = build_dossier(item, run_date=date or paths.root.name)
        markdown = render_dossier(dossier, run_date=date or paths.root.name)
        validation = validate_dossier_report(markdown)
        if not validation.passed:
            for error in validation.errors:
                logger.error("Dossier validation failed: %s", error)
            raise typer.Exit(1)
        output = paths.reports_dir / "dossiers" / f"{dossier.dossier_id}.md"
        write_dossier(output, markdown)
        logger.info("Generated dossier at %s", output)


@brief_app.command("preferences")
def preferences_command(
    rollback: str | None = typer.Option(
        None,
        "--rollback",
        help="Rollback a preference adjustment ID instead of computing updates.",
    ),
    min_feedback: int = typer.Option(
        3,
        "--min-feedback",
        help="Minimum explicit feedback events before adjustment.",
    ),
    workspace: Path | None = typer.Option(
        None, "--workspace", "-w", help="Workspace root."
    ),
    date: str | None = typer.Option(None, "--date", help="Briefing date YYYY-MM-DD."),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable debug logging."
    ),
) -> None:
    """Compute or rollback reversible preference adjustments from feedback."""
    _setup(verbose)
    paths = resolve_briefing_paths(workspace, date)
    db_path = paths.root / "feedback" / "feedback.db"
    if rollback:
        result = rollback_preference_adjustment(db_path, rollback)
        logger.info("Rolled back preference adjustment %s", result["adjustment_id"])
        return
    store = BriefingFeedbackStore(db_path)
    try:
        adjustments = store.create_adjustments(min_feedback=min_feedback)
        logger.info("Created %d preference adjustment(s)", len(adjustments))
    finally:
        store.close()


@brief_app.command("resume")
def resume_command(
    from_stage: str = typer.Option(
        ...,
        "--from-stage",
        help="Stage to resume from: rank, generate-daily, or validate.",
    ),
    registry: Path | None = typer.Option(
        None, "--registry", "-r", help="Source registry."
    ),
    workspace: Path | None = typer.Option(
        None, "--workspace", "-w", help="Workspace root."
    ),
    date: str = typer.Option(..., "--date", help="Briefing date YYYY-MM-DD."),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable debug logging."
    ),
) -> None:
    """Resume a briefing workflow from existing artifacts."""
    _setup(verbose)
    source_registry = _registry_or_snapshot(registry, workspace, date)
    paths, validation = resume_briefing(
        source_registry,
        workspace=workspace,
        run_date=date,
        from_stage=from_stage,
    )
    logger.info("Resumed briefing at %s", paths.root)
    if validation is not None and not bool(validation["passed"]):
        raise typer.Exit(1)


@brief_app.command("compare-sources")
def compare_sources_command(
    base_registry: Path = typer.Option(..., "--base-registry", help="Base registry."),
    expanded_registry: Path = typer.Option(
        ..., "--expanded-registry", help="Registry with candidate source enabled."
    ),
    workspace: Path | None = typer.Option(
        None, "--workspace", "-w", help="Workspace root."
    ),
    date: str = typer.Option(..., "--date", help="Briefing date YYYY-MM-DD."),
    fixture_base_dir: Path | None = typer.Option(
        None, "--fixture-base-dir", help="Base directory for fixtures."
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable debug logging."
    ),
) -> None:
    """Compare ranked output with and without an expanded source registry."""
    _setup(verbose)
    base_paths, _base_validation = run_briefing(
        load_source_registry(base_registry),
        workspace=(workspace or Path("./workspace")) / "compare" / "base",
        run_date=date,
        fixture_base_dir=fixture_base_dir,
    )
    expanded_paths, _expanded_validation = run_briefing(
        load_source_registry(expanded_registry),
        workspace=(workspace or Path("./workspace")) / "compare" / "expanded",
        run_date=date,
        fixture_base_dir=fixture_base_dir,
    )
    base_clusters = read_jsonl(base_paths.ranked_clusters_path, BriefingCluster)
    expanded_clusters = read_jsonl(expanded_paths.ranked_clusters_path, BriefingCluster)
    base_ids = {cluster.cluster_id for cluster in base_clusters}
    expanded_ids = {cluster.cluster_id for cluster in expanded_clusters}
    logger.info("Added clusters: %s", sorted(expanded_ids - base_ids))
    logger.info("Removed clusters: %s", sorted(base_ids - expanded_ids))


@brief_app.command("weekly-synthesis")
def weekly_synthesis_command(
    week: str = typer.Option(..., "--week", help="Week ID, e.g. 2026-W18."),
    workspace: Path | None = typer.Option(
        None, "--workspace", "-w", help="Workspace root."
    ),
    output: Path | None = typer.Option(
        None, "--output", "-o", help="Output Markdown path."
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable debug logging."
    ),
) -> None:
    """Generate a lightweight weekly trend memo from daily briefing reports."""
    _setup(verbose)
    root = (workspace or Path("./workspace")) / "briefings"
    reports = [
        path.read_text(encoding="utf-8")
        for path in sorted(root.glob("*/reports/daily.md"))
        if path.is_file()
    ]
    markdown = render_weekly_synthesis(reports, week_id=week)
    output_path = output or root / "weekly" / f"{week}.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown, encoding="utf-8")
    logger.info("Generated weekly synthesis at %s", output_path)
