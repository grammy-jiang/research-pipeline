"""Structured audit event logging for pipeline operations.

Provides a lightweight audit trail of pipeline decisions and stage
transitions.  Each event is a single JSONL line written to
``<run_root>/logs/audit.jsonl`` with a fixed schema so downstream
tools can query what happened, when, and with what outcome.

Usage::

    from research_pipeline.infra.audit import AuditLogger

    audit = AuditLogger(run_root)
    audit.emit("stage_started", stage="screen", details={"candidates": 200})
    audit.emit("stage_completed", stage="screen", details={"selected": 50})
"""

import json
import logging
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_AUDIT_FILENAME = "audit.jsonl"


class EventType(StrEnum):
    """Well-known audit event types."""

    STAGE_STARTED = "stage_started"
    STAGE_COMPLETED = "stage_completed"
    STAGE_FAILED = "stage_failed"
    DECISION = "decision"
    ARTIFACT_CREATED = "artifact_created"
    CONFIG_LOADED = "config_loaded"


class AuditLogger:
    """Append-only JSONL audit logger scoped to a pipeline run.

    Args:
        run_root: Root directory of the pipeline run.  The audit log
            is written to ``<run_root>/logs/audit.jsonl``.
        enabled: When ``False`` all ``emit()`` calls are no-ops.
            Defaults to ``True``.
    """

    def __init__(self, run_root: Path, *, enabled: bool = True) -> None:
        self._enabled = enabled
        self._log_dir = run_root / "logs"
        self._log_path = self._log_dir / _AUDIT_FILENAME

    @property
    def log_path(self) -> Path:
        """Path to the audit JSONL file."""
        return self._log_path

    @property
    def enabled(self) -> bool:
        """Whether audit logging is active."""
        return self._enabled

    def emit(
        self,
        event_type: str | EventType,
        *,
        stage: str = "",
        run_id: str = "",
        details: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Record a single audit event.

        Args:
            event_type: Type of event (use :class:`EventType` members
                or a custom string).
            stage: Pipeline stage name (e.g. ``"screen"``).
            run_id: Pipeline run identifier.
            details: Arbitrary key-value data for the event.

        Returns:
            The event dict that was written (useful for testing).
        """
        event: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "event_type": (
                event_type.value
                if isinstance(event_type, EventType)
                else str(event_type)
            ),
        }
        if run_id:
            event["run_id"] = run_id
        if stage:
            event["stage"] = stage
        if details:
            event["details"] = details

        if self._enabled:
            self._write(event)

        return event

    def _write(self, event: dict[str, Any]) -> None:
        """Append *event* as a single JSON line to the audit log."""
        try:
            self._log_dir.mkdir(parents=True, exist_ok=True)
            with self._log_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(event, default=str) + "\n")
        except OSError:
            logger.warning("Failed to write audit event", exc_info=True)

    def read_events(self) -> list[dict[str, Any]]:
        """Read all events from the audit log.

        Returns:
            List of event dicts in chronological order.
            Empty list if the log does not exist.
        """
        if not self._log_path.exists():
            return []
        events: list[dict[str, Any]] = []
        with self._log_path.open(encoding="utf-8") as fh:
            for line in fh:
                stripped = line.strip()
                if stripped:
                    events.append(json.loads(stripped))
        return events
