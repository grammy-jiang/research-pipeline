"""Tests for the verify gate command (#18)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import typer

from research_pipeline.cli.cmd_evaluate import verify_cmd


def test_verify_registered_on_app() -> None:
    command = typer.main.get_command(
        __import__("research_pipeline.cli.app", fromlist=["app"]).app
    )
    assert "verify" in command.commands  # type: ignore[attr-defined]
    verify = command.commands["verify"]  # type: ignore[attr-defined]
    assert {"run_id", "stage", "config"} <= {p.name for p in verify.params}


def test_verify_missing_run_exits(tmp_path: Path) -> None:
    with pytest.raises(typer.Exit):
        verify_cmd(run_id="nope", workspace=tmp_path)


@patch("research_pipeline.evaluation.schema_eval.evaluate_stage")
def test_verify_stage_failure_exits_nonzero(mock_eval, tmp_path: Path) -> None:
    (tmp_path / "run1").mkdir()
    report = MagicMock()
    report.passed = False
    report.checks = []
    report.summary.return_value = "summary"
    mock_eval.return_value = report
    with pytest.raises(typer.Exit):
        verify_cmd(run_id="run1", stage="plan", workspace=tmp_path)


@patch("research_pipeline.evaluation.schema_eval.evaluate_stage")
def test_verify_stage_pass_ok(mock_eval, tmp_path: Path) -> None:
    (tmp_path / "run1").mkdir()
    report = MagicMock()
    report.passed = True
    report.checks = []
    report.summary.return_value = "summary"
    mock_eval.return_value = report
    # A passing stage must not raise.
    verify_cmd(run_id="run1", stage="plan", workspace=tmp_path)
