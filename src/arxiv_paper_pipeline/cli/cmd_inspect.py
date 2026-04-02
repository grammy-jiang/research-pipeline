"""CLI handler for the 'inspect' command."""

import logging
from pathlib import Path

import typer

from arxiv_paper_pipeline.storage.manifests import load_manifest

logger = logging.getLogger(__name__)


def run_inspect(
    workspace: Path | None = None,
    run_id: str | None = None,
) -> None:
    """Inspect manifests, artifacts, and cache status.

    Args:
        workspace: Workspace directory.
        run_id: Specific run to inspect.
    """
    ws = workspace or Path("runs")

    if run_id:
        run_root = ws / run_id
        if not run_root.exists():
            typer.echo(f"Run not found: {run_root}", err=True)
            raise typer.Exit(1)

        manifest = load_manifest(run_root)
        if manifest is None:
            typer.echo(f"No manifest found in {run_root}", err=True)
            raise typer.Exit(1)

        typer.echo(f"Run ID:   {manifest.run_id}")
        typer.echo(f"Created:  {manifest.created_at}")
        typer.echo(f"Version:  {manifest.package_version}")
        typer.echo(f"Topic:    {manifest.topic_input}")
        typer.echo("")
        typer.echo("Stages:")
        for name, record in manifest.stages.items():
            status_icon = "✓" if record.status == "completed" else "✗"
            duration = f"{record.duration_ms}ms" if record.duration_ms else "N/A"
            typer.echo(f"  {status_icon} {name}: {record.status} ({duration})")
            if record.errors:
                for err in record.errors:
                    typer.echo(f"    ERROR: {err}")

        typer.echo(f"\nArtifacts: {len(manifest.artifacts)}")
        typer.echo(f"LLM calls: {len(manifest.llm_calls)}")

    else:
        # List all runs
        if not ws.exists():
            typer.echo(f"Workspace not found: {ws}")
            raise typer.Exit(1)

        runs = sorted(ws.iterdir()) if ws.is_dir() else []
        run_dirs = [r for r in runs if r.is_dir() and not r.name.startswith(".")]

        if not run_dirs:
            typer.echo("No runs found.")
            return

        typer.echo(f"Found {len(run_dirs)} runs in {ws}:\n")
        for rd in run_dirs:
            manifest = load_manifest(rd)
            if manifest:
                stages = len(
                    [s for s in manifest.stages.values() if s.status == "completed"]
                )
                typer.echo(
                    f"  {rd.name}  topic={manifest.topic_input[:50]}  stages={stages}/7"
                )
            else:
                typer.echo(f"  {rd.name}  (no manifest)")
