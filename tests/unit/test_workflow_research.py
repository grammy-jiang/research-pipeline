"""Tests for the research workflow orchestrator.

Uses mocked pipeline stages, sampling, and elicitation to test
the harness logic independently of LLM quality (harness sensitivity
principle from Harness-Native SE).
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from mcp_server.workflow.monitoring import IterationMetrics
from mcp_server.workflow.research import (
    _build_result,
    _parse_json_response,
    run_research_workflow,
)
from mcp_server.workflow.state import (
    StageStatus,
    WorkflowState,
    load_state,
    save_state,
)
from mcp_server.workflow.telemetry import WorkflowTelemetry


class TestParseJsonResponse:
    """JSON response parsing from sampling output."""

    def test_plain_json(self) -> None:
        result = _parse_json_response('{"key": "value"}')
        assert result == {"key": "value"}

    def test_markdown_code_fence(self) -> None:
        text = '```json\n{"key": "value"}\n```'
        result = _parse_json_response(text)
        assert result == {"key": "value"}

    def test_json_in_text(self) -> None:
        text = 'Here is the analysis:\n{"rating": 4, "findings": ["good"]}\nDone.'
        result = _parse_json_response(text)
        assert result["rating"] == 4

    def test_invalid_json_returns_raw(self) -> None:
        text = "This is not JSON at all."
        result = _parse_json_response(text)
        assert "raw_response" in result

    def test_empty_string(self) -> None:
        result = _parse_json_response("")
        assert "raw_response" in result


class TestBuildResult:
    """Result builder tests."""

    def test_basic_result(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            state = WorkflowState(run_id="test", topic="AI", workspace=tmpdir)
            state = state.with_stage_status("plan", StageStatus.VERIFIED)
            state = state.with_stage_status("search", StageStatus.VERIFIED)

            telemetry = WorkflowTelemetry(tmpdir, "test")
            metrics = IterationMetrics()

            result = _build_result(state, telemetry, metrics)
            assert result["success"] is True
            assert "plan" in result["stages_verified"]
            assert "search" in result["stages_verified"]
            assert result["run_id"] == "test"


@pytest.fixture
def workspace(tmp_path: Path) -> str:
    """Create a temporary workspace."""
    return str(tmp_path)


def _create_mock_stage_output(workspace: str, run_id: str) -> None:
    """Create mock stage outputs for the pipeline."""
    run_dir = Path(workspace) / run_id

    # Plan output
    plan_dir = run_dir / "plan"
    plan_dir.mkdir(parents=True)
    (plan_dir / "query_plan.json").write_text(
        json.dumps(
            {
                "must_terms": ["harness", "engineering"],
                "nice_terms": ["AI"],
                "query_variants": ["v1", "v2", "v3"],
            }
        )
    )

    # Search output
    search_dir = run_dir / "search"
    search_dir.mkdir(parents=True)
    candidates = [
        {"paper_id": "2301.00001", "title": "Paper A", "abstract": "About harness"},
        {"paper_id": "2301.00002", "title": "Paper B", "abstract": "About AI"},
    ]
    (search_dir / "candidates.jsonl").write_text(
        "\n".join(json.dumps(c) for c in candidates)
    )

    # Screen output
    screen_dir = run_dir / "screen"
    screen_dir.mkdir(parents=True)
    (screen_dir / "shortlist.json").write_text(
        json.dumps([{"paper_id": "2301.00001", "score": 0.9}])
    )

    # Download output
    pdf_dir = run_dir / "download" / "pdf"
    pdf_dir.mkdir(parents=True)
    (pdf_dir / "2301.00001.pdf").write_bytes(b"x" * 15_000)

    # Convert output
    md_dir = run_dir / "convert" / "markdown"
    md_dir.mkdir(parents=True)
    (md_dir / "2301.00001.md").write_text(
        "# Paper A\n## Abstract\nContent\n" + "x" * 600
    )

    # Extract output
    extract_dir = run_dir / "extract"
    extract_dir.mkdir(parents=True)
    (extract_dir / "chunks.json").write_text("[]")


class TestWorkflowWithoutSampling:
    """Test workflow degradation when sampling is unavailable."""

    def test_no_ctx_runs_pipeline_only(self, workspace: str) -> None:
        """Without a context, workflow should complete pipeline stages only."""
        run_id = "test-no-ctx"

        with patch("mcp_server.workflow.research._execute_pipeline_stage") as mock_exec:
            mock_exec.return_value = {
                "success": True,
                "message": "Stage completed",
                "artifacts": {},
            }
            _create_mock_stage_output(workspace, run_id)

            result = asyncio.run(
                run_research_workflow(
                    topic="test topic",
                    ctx=None,
                    workspace=workspace,
                    run_id=run_id,
                )
            )

        assert result["success"] is True
        assert "run_id" in result


class TestWorkflowRecovery:
    """Test crash-recovery (recovery dominance principle)."""

    def test_resume_from_saved_state(self, workspace: str) -> None:
        """Workflow should resume from the last persisted state."""
        run_id = "resume-test"

        # Create a state where plan+search are already verified
        state = WorkflowState(
            run_id=run_id,
            topic="test",
            workspace=workspace,
        )
        state = state.with_stage_status("plan", StageStatus.VERIFIED)
        state = state.with_stage_status("search", StageStatus.VERIFIED)
        state = state.model_copy(update={"current_stage": "search"})
        save_state(state)

        _create_mock_stage_output(workspace, run_id)

        with patch("mcp_server.workflow.research._execute_pipeline_stage") as mock_exec:
            mock_exec.return_value = {
                "success": True,
                "message": "Stage completed",
                "artifacts": {},
            }

            result = asyncio.run(
                run_research_workflow(
                    topic="test",
                    ctx=None,
                    workspace=workspace,
                    run_id=run_id,
                    resume=True,
                )
            )

        assert result["success"] is True


class TestWorkflowStateTransitions:
    """Test that the workflow correctly handles state transitions."""

    def test_stage_failure_persists_state(self, workspace: str) -> None:
        """Failed stage should still persist state for recovery."""
        run_id = "fail-test"

        with patch("mcp_server.workflow.research._execute_pipeline_stage") as mock_exec:
            mock_exec.return_value = {
                "success": False,
                "message": "Plan generation failed",
            }
            _create_mock_stage_output(workspace, run_id)

            result = asyncio.run(
                run_research_workflow(
                    topic="test",
                    ctx=None,
                    workspace=workspace,
                    run_id=run_id,
                )
            )

        assert result["success"] is False
        # State should be persisted
        loaded = load_state(workspace, run_id)
        assert loaded is not None


class TestWorkflowHarnessIntegration:
    """Integration tests for harness layer behavior."""

    def test_telemetry_emitted(self, workspace: str) -> None:
        """Verify that telemetry events are produced."""
        run_id = "telemetry-test"
        _create_mock_stage_output(workspace, run_id)

        with patch("mcp_server.workflow.research._execute_pipeline_stage") as mock_exec:
            mock_exec.return_value = {
                "success": True,
                "message": "OK",
                "artifacts": {},
            }

            result = asyncio.run(
                run_research_workflow(
                    topic="test",
                    ctx=None,
                    workspace=workspace,
                    run_id=run_id,
                )
            )

        assert result["success"] is True
        # Telemetry log should exist
        tel_path = Path(workspace) / run_id / "workflow" / "telemetry.jsonl"
        assert tel_path.exists()
        lines = tel_path.read_text().strip().split("\n")
        assert len(lines) > 0

    def test_verification_runs_for_stages(self, workspace: str) -> None:
        """Disk-based stages should have verification events."""
        run_id = "verify-test"
        _create_mock_stage_output(workspace, run_id)

        with patch("mcp_server.workflow.research._execute_pipeline_stage") as mock_exec:
            mock_exec.return_value = {
                "success": True,
                "message": "OK",
                "artifacts": {},
            }

            result = asyncio.run(
                run_research_workflow(
                    topic="test",
                    ctx=None,
                    workspace=workspace,
                    run_id=run_id,
                )
            )

        assert result["success"] is True

        # Check telemetry for verification events
        tel_path = Path(workspace) / run_id / "workflow" / "telemetry.jsonl"
        events = [json.loads(line) for line in tel_path.read_text().strip().split("\n")]
        verification_events = [
            e for e in events if e.get("event_type") == "verification"
        ]
        assert len(verification_events) >= 1
