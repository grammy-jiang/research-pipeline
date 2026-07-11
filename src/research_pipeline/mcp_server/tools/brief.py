from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from research_pipeline.mcp_server.schemas import (
    BriefExportObsidianInput,
    BriefGenerateDailyInput,
    BriefGenerateDossierInput,
    BriefPollSourcesInput,
    BriefRankEventsInput,
    BriefRecordFeedbackInput,
    BriefRunInput,
    BriefValidateReportInput,
    BriefWeeklySynthesisInput,
    ToolResult,
)
from research_pipeline.mcp_server.tools._common import (
    _log_info,
    _raise_tool_error,
)

if TYPE_CHECKING:
    from mcp.server.fastmcp import Context


def brief_poll_sources_tool(
    params: BriefPollSourcesInput, ctx: Context | None = None
) -> ToolResult:
    """Poll configured briefing sources."""
    try:
        from research_pipeline.briefing.registry import load_source_registry
        from research_pipeline.briefing.workflow import poll_sources

        paths, events = poll_sources(
            load_source_registry(
                Path(params.registry_path) if params.registry_path else None
            ),
            workspace=Path(params.workspace),
            run_date=params.date or None,
            fixture_base_dir=Path(params.fixture_base_dir)
            if params.fixture_base_dir
            else None,
        )
        _log_info(ctx, f"Polled {len(events)} briefing event(s)")
        return ToolResult(
            success=True,
            message=f"Polled {len(events)} briefing event(s).",
            artifacts={"root": str(paths.root), "events": str(paths.events_path)},
        )
    except Exception as exc:
        _raise_tool_error("brief_poll_sources", exc)


def brief_rank_events_tool(
    params: BriefRankEventsInput, ctx: Context | None = None
) -> ToolResult:
    """Deduplicate and rank briefing events."""
    try:
        from research_pipeline.briefing.layout import resolve_briefing_paths
        from research_pipeline.briefing.registry import load_source_registry
        from research_pipeline.briefing.workflow import (
            load_registry_snapshot,
            rank_events,
        )

        paths = resolve_briefing_paths(Path(params.workspace), params.date or None)
        registry = (
            load_source_registry(Path(params.registry_path))
            if params.registry_path
            else load_registry_snapshot(paths)
        )
        ranked = rank_events(
            paths,
            registry,
            use_memory=params.use_memory,
            use_feedback=params.use_feedback,
        )
        _log_info(ctx, f"Ranked {len(ranked)} briefing cluster(s)")
        return ToolResult(
            success=True,
            message=f"Ranked {len(ranked)} briefing cluster(s).",
            artifacts={"ranked_clusters": str(paths.ranked_clusters_path)},
        )
    except Exception as exc:
        _raise_tool_error("brief_rank_events", exc)


def brief_generate_daily_tool(
    params: BriefGenerateDailyInput, ctx: Context | None = None
) -> ToolResult:
    """Generate the daily briefing Markdown."""
    try:
        from research_pipeline.briefing.layout import resolve_briefing_paths
        from research_pipeline.briefing.workflow import generate_daily

        paths = resolve_briefing_paths(Path(params.workspace), params.date or None)
        generate_daily(paths, run_date=params.date or None)
        _log_info(ctx, f"Generated daily brief at {paths.daily_report_path}")
        return ToolResult(
            success=True,
            message="Generated daily briefing.",
            artifacts={"daily_report": str(paths.daily_report_path)},
        )
    except Exception as exc:
        _raise_tool_error("brief_generate_daily", exc)


def brief_validate_report_tool(
    params: BriefValidateReportInput, ctx: Context | None = None
) -> ToolResult:
    """Validate the daily briefing report."""
    try:
        from research_pipeline.briefing.layout import resolve_briefing_paths
        from research_pipeline.briefing.workflow import validate_daily

        paths = resolve_briefing_paths(Path(params.workspace), params.date or None)
        validation = validate_daily(paths)
        _log_info(ctx, f"Brief validation passed={validation['passed']}")
        return ToolResult(
            success=bool(validation["passed"]),
            message="Brief validation completed.",
            artifacts={"validation": validation, "path": str(paths.validation_path)},
        )
    except Exception as exc:
        _raise_tool_error("brief_validate_report", exc)


def brief_run_tool(params: BriefRunInput, ctx: Context | None = None) -> ToolResult:
    """Run the deterministic briefing workflow."""
    try:
        from research_pipeline.briefing.registry import load_source_registry
        from research_pipeline.briefing.workflow import run_briefing

        paths, validation = run_briefing(
            load_source_registry(
                Path(params.registry_path) if params.registry_path else None
            ),
            workspace=Path(params.workspace),
            run_date=params.date or None,
            fixture_base_dir=Path(params.fixture_base_dir)
            if params.fixture_base_dir
            else None,
        )
        _log_info(ctx, f"Briefing run completed at {paths.root}")
        return ToolResult(
            success=bool(validation["passed"]),
            message="Briefing run completed.",
            artifacts={
                "root": str(paths.root),
                "daily_report": str(paths.daily_report_path),
                "validation": validation,
            },
        )
    except Exception as exc:
        _raise_tool_error("brief_run", exc)


def brief_export_obsidian_tool(
    params: BriefExportObsidianInput, ctx: Context | None = None
) -> ToolResult:
    """Export briefing notes to Obsidian."""
    try:
        from research_pipeline.briefing.io import read_jsonl
        from research_pipeline.briefing.layout import resolve_briefing_paths
        from research_pipeline.briefing.models import BriefingCluster
        from research_pipeline.briefing.obsidian import (
            export_daily_note,
            export_source_notes,
            export_topic_notes,
        )
        from research_pipeline.briefing.registry import load_source_registry
        from research_pipeline.briefing.workflow import load_registry_snapshot

        paths = resolve_briefing_paths(Path(params.workspace), params.date or None)
        markdown = paths.daily_report_path.read_text(encoding="utf-8")
        clusters = read_jsonl(paths.ranked_clusters_path, BriefingCluster)
        registry = (
            load_source_registry(Path(params.registry_path))
            if params.registry_path
            else load_registry_snapshot(paths)
        )
        vault = Path(params.vault_path)
        changed = [
            export_daily_note(
                markdown, vault_root=vault, run_date=params.date or paths.root.name
            ),
            *export_topic_notes(list(clusters), vault_root=vault),  # type: ignore[arg-type]
            *export_source_notes(list(registry.sources), vault_root=vault),
        ]
        return ToolResult(
            success=True,
            message=f"Exported {len(changed)} Obsidian note(s).",
            artifacts={"notes": [str(path) for path in changed]},
        )
    except Exception as exc:
        _raise_tool_error("brief_export_obsidian", exc)


def brief_record_feedback_tool(
    params: BriefRecordFeedbackInput, ctx: Context | None = None
) -> ToolResult:
    """Record explicit briefing feedback."""
    try:
        from research_pipeline.briefing.feedback import BriefingFeedbackStore
        from research_pipeline.briefing.layout import resolve_briefing_paths
        from research_pipeline.briefing.models import FeedbackSignal

        paths = resolve_briefing_paths(Path(params.workspace), params.date or None)
        store = BriefingFeedbackStore(paths.root / "feedback" / "feedback.db")
        try:
            feedback = store.record(
                target_type=params.target_type,
                target_id=params.target_id,
                signal=FeedbackSignal(params.signal),
                strength=params.strength,
                reason=params.reason,
            )
        finally:
            store.close()
        return ToolResult(
            success=True,
            message=f"Recorded feedback {feedback.feedback_id}.",
            artifacts={"feedback": feedback.model_dump(mode="json")},
        )
    except Exception as exc:
        _raise_tool_error("brief_record_feedback", exc)


def brief_generate_dossier_tool(
    params: BriefGenerateDossierInput, ctx: Context | None = None
) -> ToolResult:
    """Generate a manual hot-topic briefing dossier."""
    try:
        from research_pipeline.briefing.dossier import (
            build_dossier,
            render_dossier,
            write_dossier,
        )
        from research_pipeline.briefing.io import read_jsonl
        from research_pipeline.briefing.layout import resolve_briefing_paths
        from research_pipeline.briefing.models import BriefingCluster

        paths = resolve_briefing_paths(Path(params.workspace), params.date or None)
        clusters = read_jsonl(paths.ranked_clusters_path, BriefingCluster)
        cluster = next(
            item
            for item in clusters
            if item.cluster_id == params.cluster_id  # type: ignore[union-attr]
        )
        dossier = build_dossier(cluster, run_date=params.date or paths.root.name)  # type: ignore[arg-type]
        markdown = render_dossier(dossier, run_date=params.date or paths.root.name)
        output = paths.reports_dir / "dossiers" / f"{dossier.dossier_id}.md"
        write_dossier(output, markdown)
        return ToolResult(
            success=True,
            message="Generated briefing dossier.",
            artifacts={"dossier": str(output), "dossier_id": dossier.dossier_id},
        )
    except Exception as exc:
        _raise_tool_error("brief_generate_dossier", exc)


def brief_weekly_synthesis_tool(
    params: BriefWeeklySynthesisInput, ctx: Context | None = None
) -> ToolResult:
    """Generate a weekly briefing synthesis."""
    try:
        from research_pipeline.briefing.report import render_weekly_synthesis

        root = Path(params.workspace) / "briefings"
        reports = [
            path.read_text(encoding="utf-8")
            for path in sorted(root.glob("*/reports/daily.md"))
            if path.is_file()
        ]
        markdown = render_weekly_synthesis(reports, week_id=params.week)
        output = (
            Path(params.output_path)
            if params.output_path
            else root / "weekly" / f"{params.week}.md"
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(markdown, encoding="utf-8")
        _log_info(ctx, f"Generated weekly briefing synthesis at {output}")
        return ToolResult(
            success=True,
            message="Generated weekly briefing synthesis.",
            artifacts={"weekly_synthesis": str(output)},
        )
    except Exception as exc:
        _raise_tool_error("brief_weekly_synthesis", exc)
