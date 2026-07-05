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


def verify_cmd(
    run_id: str,
    stage: str = "",
    config_path: "Path | None" = None,
    workspace: "Path | None" = None,
) -> None:
    """Verify a run (or one stage) against its schema and exit non-zero on
    failure.

    A gate-friendly wrapper over the ``evaluate`` schema checks: it resolves the
    workspace from ``--config``/``--workspace``, prints the report, and raises a
    non-zero exit when validation fails so manifest gates can rely on it (#18).
    """
    from research_pipeline.config.loader import load_config
    from research_pipeline.evaluation.schema_eval import evaluate_run, evaluate_stage

    ws = (
        workspace if workspace is not None else Path(load_config(config_path).workspace)
    )
    run_root = ws / run_id
    if not run_root.exists():
        typer.echo(f"Run not found: {run_root}", err=True)
        raise typer.Exit(1)

    if stage:
        report = evaluate_stage(run_root, stage)
        typer.echo(report.summary())
        for check in report.checks:
            status = "PASS" if check.passed else "FAIL"
            typer.echo(f"  [{status}] {check.name}: {check.description}")
        passed = report.passed
    else:
        reports = evaluate_run(run_root)
        for report in reports:
            typer.echo(report.summary())
        passed = all(r.passed for r in reports)

    typer.echo(f"VERIFY: {'PASS' if passed else 'FAIL'}")
    if not passed:
        raise typer.Exit(1)
