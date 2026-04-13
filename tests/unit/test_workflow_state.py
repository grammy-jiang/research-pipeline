"""Tests for workflow state model, transitions, and persistence."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from mcp_server.workflow.state import (
    ContextBudget,
    ExecutionRecord,
    GovernanceError,
    StageStatus,
    WorkflowStage,
    WorkflowState,
    load_state,
    save_state,
    transition_state,
)


class TestStageStatus:
    """StageStatus enum tests."""

    def test_values(self) -> None:
        assert StageStatus.PENDING.value == "pending"
        assert StageStatus.VERIFIED.value == "verified"

    def test_all_statuses(self) -> None:
        expected = {"pending", "running", "completed", "verified", "failed", "skipped"}
        assert {s.value for s in StageStatus} == expected


class TestWorkflowStage:
    """WorkflowStage enum tests."""

    def test_core_stages_exist(self) -> None:
        stages = {s.value for s in WorkflowStage}
        for expected in (
            "plan",
            "search",
            "screen",
            "download",
            "convert",
            "extract",
            "analyze",
            "synthesize",
            "report",
        ):
            assert expected in stages

    def test_all_stages_have_transitions(self) -> None:
        from mcp_server.workflow.state import VALID_TRANSITIONS

        for stage in WorkflowStage:
            assert stage.value in VALID_TRANSITIONS


class TestExecutionRecord:
    """ExecutionRecord model tests."""

    def test_create_minimal(self) -> None:
        record = ExecutionRecord(
            stage="plan",
            intent="Test plan",
            observation="Observed",
            inference="Inferred",
        )
        assert record.stage == "plan"
        assert record.timestamp  # Auto-generated

    def test_roundtrip(self) -> None:
        record = ExecutionRecord(
            stage="search",
            intent="Search papers",
            observation="Found 42 candidates",
            inference="Proceeding to screen",
            artifacts_produced=["/path/to/candidates.jsonl"],
            verification_result="passed",
            elapsed_seconds=3.14,
            iteration=1,
            metadata={"source": "arxiv"},
        )
        data = record.model_dump()
        restored = ExecutionRecord(**data)
        assert restored.stage == record.stage
        assert restored.metadata == {"source": "arxiv"}


class TestContextBudget:
    """ContextBudget model tests."""

    def test_defaults(self) -> None:
        budget = ContextBudget()
        assert budget.total_budget == 2_000_000
        assert budget.remaining == 2_000_000
        assert budget.budget_utilization == 0.0
        assert budget.can_afford(100_000)

    def test_utilization(self) -> None:
        budget = ContextBudget(total_budget=1000, used=750)
        assert budget.budget_utilization == 0.75
        assert budget.remaining == 250
        assert budget.can_afford(250)
        assert not budget.can_afford(251)

    def test_zero_budget(self) -> None:
        budget = ContextBudget(total_budget=0, used=0)
        assert budget.budget_utilization == 1.0
        assert budget.remaining == 0
        assert not budget.can_afford(1)


class TestWorkflowState:
    """WorkflowState model tests."""

    def test_create_initializes_stages(self) -> None:
        state = WorkflowState(run_id="test", topic="AI memory")
        for stage in WorkflowStage:
            assert state.get_stage_status(stage.value) == StageStatus.PENDING

    def test_with_stage_status_returns_new_state(self) -> None:
        state = WorkflowState(run_id="test", topic="AI")
        new_state = state.with_stage_status("plan", StageStatus.RUNNING)
        # Original unchanged
        assert state.get_stage_status("plan") == StageStatus.PENDING
        # New state updated
        assert new_state.get_stage_status("plan") == StageStatus.RUNNING

    def test_with_execution_record_appends(self) -> None:
        state = WorkflowState(run_id="test", topic="AI")
        assert len(state.execution_log) == 0

        record = ExecutionRecord(
            stage="plan", intent="test", observation="ok", inference="good"
        )
        new_state = state.with_execution_record(record)
        assert len(new_state.execution_log) == 1
        assert len(state.execution_log) == 0  # Original unchanged

    def test_with_fingerprint(self) -> None:
        state = WorkflowState(run_id="test", topic="AI")
        new_state = state.with_fingerprint("test_key", "some content")
        assert "test_key" in new_state.content_fingerprints
        assert len(new_state.content_fingerprints["test_key"]) == 32  # MD5

    def test_with_budget_update(self) -> None:
        state = WorkflowState(run_id="test", topic="AI")
        new_state = state.with_budget_update(1000, paper_content_tokens=800)
        assert new_state.context_budget.used == 1000
        assert new_state.context_budget.paper_content_tokens == 800
        assert state.context_budget.used == 0  # Original unchanged

    def test_json_roundtrip(self) -> None:
        state = WorkflowState(
            run_id="test-123",
            topic="harness engineering",
            system_building=True,
            max_iterations=5,
        )
        data = json.loads(state.model_dump_json())
        restored = WorkflowState(**data)
        assert restored.run_id == state.run_id
        assert restored.topic == state.topic
        assert restored.system_building is True
        assert restored.max_iterations == 5


class TestTransitions:
    """State transition governance tests."""

    def _make_verified_state(self, stage: str) -> WorkflowState:
        """Create a state where the given stage is verified."""
        state = WorkflowState(run_id="test", topic="AI")
        state = state.with_stage_status(stage, StageStatus.VERIFIED)
        return state.model_copy(update={"current_stage": stage})

    def test_valid_plan_to_search(self) -> None:
        state = self._make_verified_state("plan")
        new_state = transition_state(state, "search")
        assert new_state.current_stage == "search"

    def test_valid_screen_to_download(self) -> None:
        state = self._make_verified_state("screen")
        new_state = transition_state(state, "download")
        assert new_state.current_stage == "download"

    def test_invalid_transition_raises(self) -> None:
        state = self._make_verified_state("plan")
        with pytest.raises(GovernanceError, match="Invalid transition"):
            transition_state(state, "download")

    def test_unverified_stage_raises(self) -> None:
        state = WorkflowState(run_id="test", topic="AI")
        state = state.with_stage_status("plan", StageStatus.COMPLETED)
        state = state.model_copy(update={"current_stage": "plan"})
        with pytest.raises(GovernanceError, match="must be 'verified'"):
            transition_state(state, "search")

    def test_terminal_stage_no_transitions(self) -> None:
        state = self._make_verified_state("report")
        with pytest.raises(GovernanceError, match="none \\(terminal stage\\)"):
            transition_state(state, "plan")

    def test_screen_branches(self) -> None:
        """Screen can branch to quality, expand, or download."""
        state = self._make_verified_state("screen")
        for target in ["quality", "expand", "download"]:
            new_state = transition_state(state, target)
            assert new_state.current_stage == target

    def test_synthesize_branches(self) -> None:
        """Synthesize can go to iterate or report."""
        state = self._make_verified_state("synthesize")
        for target in ["iterate", "report"]:
            new_state = transition_state(state, target)
            assert new_state.current_stage == target

    def test_iterate_back_to_plan(self) -> None:
        """Iterate loops back to plan."""
        state = self._make_verified_state("iterate")
        new_state = transition_state(state, "plan")
        assert new_state.current_stage == "plan"


class TestPersistence:
    """State persistence (recovery layer) tests."""

    def test_save_and_load(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            state = WorkflowState(
                run_id="persist-test",
                topic="testing persistence",
                workspace=tmpdir,
            )
            state = state.with_stage_status("plan", StageStatus.VERIFIED)

            path = save_state(state)
            assert path.exists()

            loaded = load_state(tmpdir, "persist-test")
            assert loaded is not None
            assert loaded.run_id == "persist-test"
            assert loaded.get_stage_status("plan") == StageStatus.VERIFIED

    def test_load_missing_returns_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            assert load_state(tmpdir, "nonexistent") is None

    def test_load_corrupted_returns_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            state_dir = Path(tmpdir) / "bad" / "workflow"
            state_dir.mkdir(parents=True)
            (state_dir / "state.json").write_text("{invalid json!!!")
            assert load_state(tmpdir, "bad") is None

    def test_save_creates_directories(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            state = WorkflowState(
                run_id="new-run",
                topic="test",
                workspace=tmpdir,
            )
            path = save_state(state)
            assert path.exists()
            assert path.parent.name == "workflow"
