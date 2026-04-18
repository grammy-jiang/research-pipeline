"""Tests for smaller CLI handler modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ── cmd_evaluate ──────────────────────────────────────────────


class TestEvaluateCmd:
    """Tests for the evaluate command handler."""

    @patch(
        "research_pipeline.evaluation.schema_eval.evaluate_run",
        autospec=True,
    )
    def test_evaluate_run_called(
        self, mock_eval_run: MagicMock, tmp_path: Path
    ) -> None:
        """evaluate_run is called when no stage is specified."""
        from research_pipeline.cli.cmd_evaluate import evaluate_cmd

        run_root = tmp_path / "run-001"
        run_root.mkdir()

        @dataclass
        class FakeCheck:
            name: str = "check1"
            description: str = "desc"
            passed: bool = True
            details: str = ""
            severity: str = "error"

        @dataclass
        class FakeReport:
            stage: str = "plan"
            checks: list = field(default_factory=lambda: [FakeCheck()])
            passed: bool = True

            def summary(self) -> str:
                return f"{self.stage}: PASS"

        mock_eval_run.return_value = [FakeReport()]

        evaluate_cmd(run_id="run-001", stage="", workspace=str(tmp_path))
        mock_eval_run.assert_called_once_with(run_root)

    @patch(
        "research_pipeline.evaluation.schema_eval.evaluate_stage",
        autospec=True,
    )
    def test_evaluate_stage_called(
        self, mock_eval_stage: MagicMock, tmp_path: Path
    ) -> None:
        """evaluate_stage is called when stage is specified."""
        from research_pipeline.cli.cmd_evaluate import evaluate_cmd

        run_root = tmp_path / "run-002"
        run_root.mkdir()

        @dataclass
        class FakeCheck:
            name: str = "check1"
            description: str = "desc"
            passed: bool = True
            details: str = ""

        @dataclass
        class FakeReport:
            stage: str = "plan"
            checks: list = field(default_factory=lambda: [FakeCheck()])

            def summary(self) -> str:
                return "plan: PASS"

        mock_eval_stage.return_value = FakeReport()

        evaluate_cmd(run_id="run-002", stage="plan", workspace=str(tmp_path))
        mock_eval_stage.assert_called_once_with(run_root, "plan")

    def test_nonexistent_run_exits(self, tmp_path: Path) -> None:
        """Missing run directory raises typer.Exit."""
        import typer

        from research_pipeline.cli.cmd_evaluate import evaluate_cmd

        with pytest.raises((SystemExit, typer.Exit)):
            evaluate_cmd(run_id="nonexistent", stage="", workspace=str(tmp_path))


# ── cmd_run ───────────────────────────────────────────────────


class TestRunFull:
    """Tests for the run_full handler."""

    @patch(
        "research_pipeline.cli.cmd_run.run_pipeline",
        autospec=True,
    )
    @patch(
        "research_pipeline.cli.cmd_run.load_config",
        autospec=True,
    )
    def test_run_pipeline_called(
        self, mock_config: MagicMock, mock_pipeline: MagicMock
    ) -> None:
        """run_pipeline is invoked with topic and config."""
        from research_pipeline.cli.cmd_run import run_full

        fake_config = MagicMock()
        fake_config.workspace = "runs"
        fake_config.sources = MagicMock()
        fake_config.gates = MagicMock()
        mock_config.return_value = fake_config

        fake_manifest = MagicMock()
        fake_manifest.run_id = "r-123"
        fake_manifest.stages = {}
        mock_pipeline.return_value = fake_manifest

        run_full(topic="transformer architectures")
        mock_pipeline.assert_called_once()
        call_kwargs = mock_pipeline.call_args
        assert call_kwargs.kwargs["topic"] == "transformer architectures"


# ── cmd_blinding_audit ────────────────────────────────────────


class TestBlindingAuditHandler:
    """Tests for handle_blinding_audit."""

    @patch(
        "research_pipeline.cli.cmd_blinding_audit.run_blinding_audit_for_workspace",
        autospec=True,
    )
    def test_handler_calls_audit(self, mock_audit: MagicMock, tmp_path: Path) -> None:
        """run_blinding_audit_for_workspace is called with workspace."""
        from research_pipeline.cli.cmd_blinding_audit import (
            handle_blinding_audit,
        )

        fake_result = MagicMock()
        fake_result.run_id = "r1"
        fake_result.timestamp = "2024-01-01"
        fake_result.paper_scores = []
        fake_result.aggregate_score = 0.1
        fake_result.high_contamination_papers = []
        fake_result.recommendation = "Low risk"
        mock_audit.return_value = fake_result

        handle_blinding_audit(tmp_path)
        mock_audit.assert_called_once_with(
            tmp_path,
            run_id=None,
            contamination_threshold=pytest.approx(0.4),
            store_results=True,
        )

    @patch(
        "research_pipeline.cli.cmd_blinding_audit.run_blinding_audit_for_workspace",
        autospec=True,
    )
    def test_handler_json_output(self, mock_audit: MagicMock, tmp_path: Path) -> None:
        """output_json=True produces JSON via to_dict()."""
        from research_pipeline.cli.cmd_blinding_audit import (
            handle_blinding_audit,
        )

        fake_result = MagicMock()
        fake_result.to_dict.return_value = {"run_id": "r1"}
        mock_audit.return_value = fake_result

        handle_blinding_audit(tmp_path, output_json=True)
        fake_result.to_dict.assert_called_once()


# ── cmd_consolidate ───────────────────────────────────────────


class TestConsolidateCmd:
    """Tests for run_consolidate_cmd handler."""

    @patch(
        "research_pipeline.cli.cmd_consolidate.run_consolidation",
        autospec=True,
    )
    @patch(
        "research_pipeline.cli.cmd_consolidate.load_config",
        autospec=True,
    )
    def test_consolidation_called(
        self, mock_config: MagicMock, mock_consolidate: MagicMock
    ) -> None:
        """run_consolidation is called with workspace."""
        from research_pipeline.cli.cmd_consolidate import run_consolidate_cmd

        fake_config = MagicMock()
        fake_config.workspace = "runs"
        mock_config.return_value = fake_config

        fake_result = MagicMock()
        fake_result.episodes_before = 10
        fake_result.episodes_after = 8
        fake_result.rules_created = 2
        fake_result.rules_updated = 1
        fake_result.entries_pruned = 3
        fake_result.drift_metrics = []
        mock_consolidate.return_value = fake_result

        run_consolidate_cmd()
        mock_consolidate.assert_called_once()
