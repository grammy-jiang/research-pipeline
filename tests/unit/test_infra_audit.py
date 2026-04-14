"""Unit tests for infra.audit module."""

import json
from pathlib import Path

from research_pipeline.infra.audit import AuditLogger, EventType


class TestAuditLogger:
    def test_emit_creates_log_file(self, tmp_path: Path) -> None:
        audit = AuditLogger(tmp_path)
        audit.emit(EventType.STAGE_STARTED, stage="plan")
        assert audit.log_path.exists()

    def test_emit_writes_valid_jsonl(self, tmp_path: Path) -> None:
        audit = AuditLogger(tmp_path)
        audit.emit(EventType.STAGE_STARTED, stage="screen", run_id="r1")
        audit.emit(
            EventType.STAGE_COMPLETED,
            stage="screen",
            run_id="r1",
            details={"selected": 50},
        )
        lines = audit.log_path.read_text().strip().split("\n")
        assert len(lines) == 2
        for line in lines:
            parsed = json.loads(line)
            assert "timestamp" in parsed
            assert "event_type" in parsed

    def test_emit_returns_event_dict(self, tmp_path: Path) -> None:
        audit = AuditLogger(tmp_path)
        event = audit.emit(
            EventType.DECISION,
            stage="download",
            details={"reason": "over quota"},
        )
        assert event["event_type"] == "decision"
        assert event["stage"] == "download"
        assert event["details"]["reason"] == "over quota"

    def test_emit_omits_empty_optional_fields(self, tmp_path: Path) -> None:
        audit = AuditLogger(tmp_path)
        event = audit.emit(EventType.STAGE_STARTED)
        assert "stage" not in event
        assert "run_id" not in event
        assert "details" not in event

    def test_emit_accepts_custom_event_type_string(self, tmp_path: Path) -> None:
        audit = AuditLogger(tmp_path)
        event = audit.emit("custom_event", stage="extract")
        assert event["event_type"] == "custom_event"

    def test_read_events_empty_when_no_file(self, tmp_path: Path) -> None:
        audit = AuditLogger(tmp_path)
        assert audit.read_events() == []

    def test_read_events_returns_all_events(self, tmp_path: Path) -> None:
        audit = AuditLogger(tmp_path)
        audit.emit(EventType.STAGE_STARTED, stage="plan")
        audit.emit(EventType.STAGE_COMPLETED, stage="plan")
        audit.emit(EventType.ARTIFACT_CREATED, stage="plan", details={"path": "q.json"})
        events = audit.read_events()
        assert len(events) == 3
        assert events[0]["event_type"] == "stage_started"
        assert events[2]["details"]["path"] == "q.json"

    def test_disabled_logger_does_not_write(self, tmp_path: Path) -> None:
        audit = AuditLogger(tmp_path, enabled=False)
        event = audit.emit(EventType.STAGE_STARTED, stage="plan")
        assert not audit.log_path.exists()
        # Still returns the event dict for convenience
        assert event["event_type"] == "stage_started"

    def test_log_path_location(self, tmp_path: Path) -> None:
        audit = AuditLogger(tmp_path)
        assert audit.log_path == tmp_path / "logs" / "audit.jsonl"

    def test_enabled_property(self, tmp_path: Path) -> None:
        assert AuditLogger(tmp_path, enabled=True).enabled is True
        assert AuditLogger(tmp_path, enabled=False).enabled is False

    def test_config_loaded_event(self, tmp_path: Path) -> None:
        audit = AuditLogger(tmp_path)
        event = audit.emit(
            EventType.CONFIG_LOADED,
            details={"backend": "docling", "sources": ["arxiv", "scholar"]},
        )
        assert event["details"]["backend"] == "docling"
        assert event["details"]["sources"] == ["arxiv", "scholar"]
