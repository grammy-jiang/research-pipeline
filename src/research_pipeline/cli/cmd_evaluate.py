"""CLI command for schema-grounded evaluation."""

import logging
from pathlib import Path

import typer

logger = logging.getLogger(__name__)


def evaluate_cmd(
    run_id: str = typer.Option(..., "--run-id", help="Run ID to evaluate."),
    stage: str = typer.Option(
        "", "--stage", "-s", help="Specific stage (default: all)."
    ),
    workspace: str = typer.Option(
        "runs", "--workspace", "-w", help="Workspace directory."
    ),
) -> None:
    """Evaluate pipeline outputs against their schemas."""
    from research_pipeline.evaluation.schema_eval import (
        evaluate_run,
        evaluate_stage,
    )

    ws = Path(workspace)
    run_root = ws / run_id

    if not run_root.exists():
        typer.echo(f"Run not found: {run_root}")
        raise typer.Exit(1)

    if stage:
        report = evaluate_stage(run_root, stage)
        typer.echo(report.summary())
        for check in report.checks:
            status = "✓" if check.passed else "✗"
            line = f"  {status} {check.name}: {check.description}"
            if check.details:
                line += f" ({check.details})"
            typer.echo(line)
    else:
        reports = evaluate_run(run_root)
        all_passed = all(r.passed for r in reports)
        for report in reports:
            typer.echo(report.summary())
        typer.echo(f"\nOverall: {'PASS' if all_passed else 'FAIL'}")
