"""Append-only JSONL telemetry for briefing runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from research_pipeline.briefing.io import ensure_parent
from research_pipeline.briefing.normalize import utc_now_iso


class BriefingTelemetry:
    """Small append-only telemetry writer."""

    def __init__(self, path: Path) -> None:
        self.path = path
        ensure_parent(path)

    def emit(self, event_type: str, **payload: Any) -> None:
        """Append a telemetry event."""
        row = {"timestamp": utc_now_iso(), "event_type": event_type, **payload}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
