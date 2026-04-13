"""Tests for workflow telemetry (three-surface pattern)."""

from __future__ import annotations

import json
import tempfile

from mcp_server.workflow.telemetry import (
    TelemetryEvent,
    TelemetrySurface,
    WorkflowTelemetry,
)


class TestTelemetryEvent:
    """TelemetryEvent construction tests."""

    def test_creates_dict(self) -> None:
        event = TelemetryEvent(
            surface=TelemetrySurface.COGNITIVE.value,
            event_type="sampling_request",
            stage="analyze",
            message="test",
        )
        assert isinstance(event, dict)
        assert event["surface"] == "cognitive"
        assert event["event_type"] == "sampling_request"
        assert "timestamp" in event

    def test_metadata(self) -> None:
        event = TelemetryEvent(
            surface=TelemetrySurface.OPERATIONAL.value,
            event_type="stage_complete",
            stage="plan",
            message="done",
            metadata={"elapsed": 1.5},
        )
        assert event["metadata"]["elapsed"] == 1.5


class TestWorkflowTelemetry:
    """WorkflowTelemetry emission and persistence tests."""

    def test_cognitive_surface(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tel = WorkflowTelemetry(tmpdir, "test-run")

            tel.log_sampling_request("analyze", "Paper X", 5000)
            tel.log_sampling_response("analyze", "Analysis of X", 2000)
            tel.log_analysis_decision("analyze", "Include paper", "High relevance")

            events = tel.events
            assert len(events) == 3
            assert all(e["surface"] == "cognitive" for e in events)

    def test_operational_surface(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tel = WorkflowTelemetry(tmpdir, "test-run")

            tel.log_stage_start("plan", "Generate query plan")
            tel.log_stage_complete("plan", 2.5, 3)
            tel.log_verification_result("plan", True, "All checks passed")

            events = tel.events
            assert len(events) == 3
            assert all(e["surface"] == "operational" for e in events)

    def test_contextual_surface(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tel = WorkflowTelemetry(tmpdir, "test-run")

            tel.log_budget_update("analyze", 5000, 95000, 0.05)
            tel.log_iteration_state(1, 3, 10, 2)
            tel.log_user_decision("screen", "Approved shortlist")
            tel.log_doom_loop_check("iterate", False, 0.3)

            events = tel.events
            assert len(events) == 4
            assert all(e["surface"] == "contextual" for e in events)

    def test_jsonl_persistence(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tel = WorkflowTelemetry(tmpdir, "persist-run")

            tel.log_stage_start("plan", "Test persistence")
            tel.log_stage_complete("plan", 1.0, 1)

            log_path = tel.flush()
            assert log_path.exists()

            lines = log_path.read_text().strip().split("\n")
            assert len(lines) == 2

            for line in lines:
                event = json.loads(line)
                assert "surface" in event
                assert "timestamp" in event

    def test_failure_logging(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tel = WorkflowTelemetry(tmpdir, "fail-run")
            tel.log_stage_failed("download", "Connection timeout")

            events = tel.events
            assert len(events) == 1
            assert events[0]["event_type"] == "stage_failed"
            assert "timeout" in events[0]["message"].lower()

    def test_verification_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tel = WorkflowTelemetry(tmpdir, "verify-run")
            tel.log_verification_result("plan", False, "Missing must_terms")

            events = tel.events
            assert events[0]["metadata"]["passed"] is False

    def test_budget_warning_levels(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tel = WorkflowTelemetry(tmpdir, "budget-run")

            # Normal
            tel.log_budget_update("analyze", 5000, 95000, 0.05)
            # Warning (>75%)
            tel.log_budget_update("analyze", 80000, 20000, 0.80)
            # Critical (>90%)
            tel.log_budget_update("analyze", 95000, 5000, 0.95)

            events = tel.events
            assert len(events) == 3

    def test_events_are_independent_copies(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tel = WorkflowTelemetry(tmpdir, "copy-run")
            tel.log_stage_start("plan", "Test")

            events1 = tel.events
            events2 = tel.events
            assert events1 is not events2

    def test_doom_loop_detected(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tel = WorkflowTelemetry(tmpdir, "doom-run")
            tel.log_doom_loop_check("iterate", True, 0.95, iteration=2)

            events = tel.events
            assert events[0]["metadata"]["is_loop"] is True
            assert events[0]["metadata"]["similarity"] == 0.95
