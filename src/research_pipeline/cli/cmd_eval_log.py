"""CLI command for three-channel evaluation log inspection.

Provides read access to execution traces, audit DB records, and
environment snapshots captured during pipeline runs.
"""

import json
import logging
from pathlib import Path

import typer

from research_pipeline.infra.eval_logging import EvalLogger
from research_pipeline.infra.logging import setup_logging
from research_pipeline.storage.workspace import resolve_workspace

logger = logging.getLogger(__name__)


def eval_log_cmd(
    run_id: str,
    channel: str = "all",
    stage: str = "",
    limit: int = 50,
    workspace: Path | None = None,
) -> None:
    """Inspect three-channel evaluation logs for a run.

    Channels: traces, audit, snapshots, all (default).

    Args:
        run_id: Pipeline run identifier.
        channel: Which channel to inspect.
        stage: Filter by pipeline stage.
        limit: Maximum records to display.
        workspace: Override workspace root.
    """
    setup_logging()
    ws_root = resolve_workspace(workspace)
    run_root = ws_root / run_id

    if not run_root.exists():
        typer.echo(f"Run directory not found: {run_root}")
        raise typer.Exit(code=1)

    eval_log = EvalLogger(run_root)

    if channel in ("traces", "all"):
        _show_traces(eval_log, stage=stage, limit=limit)

    if channel in ("audit", "all"):
        _show_audit(eval_log, stage=stage, limit=limit)

    if channel in ("snapshots", "all"):
        _show_snapshots(eval_log)

    if channel == "summary":
        _show_summary(eval_log)

    eval_log.close()


def _show_traces(
    eval_log: EvalLogger,
    *,
    stage: str = "",
    limit: int = 50,
) -> None:
    """Display execution traces."""
    typer.echo("\n=== Channel 1: Execution Traces ===")
    traces = eval_log.tracer.read_traces(stage=stage)
    if not traces:
        typer.echo("  No traces found.")
        return

    shown = traces[-limit:]
    typer.echo(f"  Showing {len(shown)} of {len(traces)} traces")
    for t in shown:
        ts = t.get("timestamp", "?")[:19]
        evt = t.get("event", "?")
        stg = t.get("stage", "-")
        lvl = t.get("level", "info")
        line = f"  [{ts}] {lvl:7s} {stg:12s} {evt}"
        data = t.get("data")
        if data:
            line += f"  {json.dumps(data, default=str)}"
        typer.echo(line)


def _show_audit(
    eval_log: EvalLogger,
    *,
    stage: str = "",
    limit: int = 50,
) -> None:
    """Display audit DB records."""
    typer.echo("\n=== Channel 2: Audit Database ===")
    total = eval_log.audit.count(stage=stage)
    if total == 0:
        typer.echo("  No audit records found.")
        return

    records = eval_log.audit.query(stage=stage, limit=limit)
    typer.echo(f"  Showing {len(records)} of {total} records")
    for r in records:
        ts = str(r.get("timestamp", "?"))[:19]
        stg = r.get("stage", "-")
        act = r.get("action", "-")
        model = r.get("model", "")
        tokens = r.get("tokens_used", 0)
        dur = r.get("duration_ms", 0)
        line = f"  [{ts}] {stg:12s} {act:20s}"
        if model:
            line += f"  model={model}"
        if tokens:
            line += f"  tokens={tokens}"
        if dur:
            line += f"  {dur}ms"
        typer.echo(line)


def _show_snapshots(eval_log: EvalLogger) -> None:
    """Display snapshot listing."""
    typer.echo("\n=== Channel 3: Environment Snapshots ===")
    snaps = eval_log.snapshots.list_snapshots()
    if not snaps:
        typer.echo("  No snapshots found.")
        return

    typer.echo(f"  {len(snaps)} snapshot(s):")
    for name in snaps:
        manifest = eval_log.snapshots.get_manifest(name)
        if manifest:
            fc = manifest.get("file_count", "?")
            ts_val = manifest.get("total_size", 0)
            size_kb = ts_val / 1024 if ts_val else 0
            typer.echo(f"  - {name}: {fc} files, {size_kb:.1f} KB")
        else:
            typer.echo(f"  - {name}: (no manifest)")


def _show_summary(eval_log: EvalLogger) -> None:
    """Display summary of all three channels."""
    typer.echo("\n=== Eval Logging Summary ===")
    summary = eval_log.summary()
    typer.echo(json.dumps(summary, indent=2, default=str))
