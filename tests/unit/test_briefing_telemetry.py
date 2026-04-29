"""Tests for briefing telemetry JSONL logging."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from research_pipeline.briefing.telemetry import BriefingTelemetry


def test_telemetry_creates_path_if_missing() -> None:
    """BriefingTelemetry creates path parent directories if missing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "telemetry" / "logs" / "events.jsonl"
        assert not path.exists()
        assert not path.parent.exists()

        telemetry = BriefingTelemetry(path)
        telemetry.emit("test_event", key="value")

        assert path.exists()
        assert path.parent.exists()


def test_telemetry_emit_writes_jsonl() -> None:
    """BriefingTelemetry.emit() writes one JSON object per line."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "events.jsonl"
        telemetry = BriefingTelemetry(path)

        telemetry.emit("event_type_1", field1="value1")
        telemetry.emit("event_type_2", field2="value2")

        # Read raw lines
        lines = path.read_text().strip().split("\n")
        assert len(lines) == 2

        # Parse as JSON
        events = [json.loads(line) for line in lines]
        assert events[0]["event_type"] == "event_type_1"
        assert events[0]["field1"] == "value1"
        assert events[1]["event_type"] == "event_type_2"
        assert events[1]["field2"] == "value2"


def test_telemetry_event_includes_timestamp() -> None:
    """BriefingTelemetry.emit() includes ISO 8601 timestamp."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "events.jsonl"
        telemetry = BriefingTelemetry(path)

        telemetry.emit("test_event")

        line = path.read_text().strip()
        event = json.loads(line)

        assert "timestamp" in event
        # Check ISO 8601 format YYYY-MM-DDTHH:MM:SSZ
        assert event["timestamp"].endswith("Z")
        assert "T" in event["timestamp"]
        assert len(event["timestamp"]) == 20  # 2025-01-20T12:00:00Z


def test_telemetry_event_includes_event_type() -> None:
    """BriefingTelemetry.emit() includes event_type field."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "events.jsonl"
        telemetry = BriefingTelemetry(path)

        telemetry.emit("my_event_type", data="test")

        line = path.read_text().strip()
        event = json.loads(line)

        assert event["event_type"] == "my_event_type"


def test_telemetry_payload_fields_included() -> None:
    """BriefingTelemetry.emit() includes all custom payload fields."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "events.jsonl"
        telemetry = BriefingTelemetry(path)

        telemetry.emit(
            "test_event",
            field1="value1",
            field2=42,
            field3=True,
            field4={"nested": "object"},
        )

        line = path.read_text().strip()
        event = json.loads(line)

        assert event["field1"] == "value1"
        assert event["field2"] == 42
        assert event["field3"] is True
        assert event["field4"] == {"nested": "object"}


def test_telemetry_append_only() -> None:
    """BriefingTelemetry maintains append-only JSONL file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "events.jsonl"
        telemetry = BriefingTelemetry(path)

        # Emit first event
        telemetry.emit("event1", seq=1)

        # Read and check
        lines_after_first = path.read_text().strip().split("\n")
        assert len(lines_after_first) == 1

        # Emit second event with new instance
        telemetry2 = BriefingTelemetry(path)
        telemetry2.emit("event2", seq=2)

        # Read and check both present
        lines_after_second = path.read_text().strip().split("\n")
        assert len(lines_after_second) == 2
        assert json.loads(lines_after_second[0])["seq"] == 1
        assert json.loads(lines_after_second[1])["seq"] == 2


def test_telemetry_events_ordered() -> None:
    """BriefingTelemetry events are appended in call order."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "events.jsonl"
        telemetry = BriefingTelemetry(path)

        # Emit events with sequence numbers
        for i in range(5):
            telemetry.emit("event", sequence=i)

        # Read and verify order
        lines = path.read_text().strip().split("\n")
        events = [json.loads(line) for line in lines]

        for i, event in enumerate(events):
            assert event["sequence"] == i


def test_telemetry_json_keys_sorted() -> None:
    """BriefingTelemetry JSON keys are sorted for deterministic output."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "events.jsonl"
        telemetry = BriefingTelemetry(path)

        # Emit with keys in non-alphabetical order
        telemetry.emit("test", zebra="z", apple="a", middle="m")

        line = path.read_text().strip()

        # Keys should be sorted alphabetically
        assert line.index("apple") < line.index("middle") < line.index("zebra")


def test_telemetry_handles_unicode_payload() -> None:
    """BriefingTelemetry correctly encodes unicode payload."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "events.jsonl"
        telemetry = BriefingTelemetry(path)

        telemetry.emit("test_event", message="Hello 世界 🌍")

        line = path.read_text().strip()
        event = json.loads(line)

        assert event["message"] == "Hello 世界 🌍"


def test_telemetry_handles_special_characters() -> None:
    """BriefingTelemetry correctly escapes JSON special characters."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "events.jsonl"
        telemetry = BriefingTelemetry(path)

        telemetry.emit(
            "test_event",
            quoted='String with "quotes"',
            backslash="Path\\like\\this",
            newline="Line1\nLine2",
        )

        line = path.read_text().strip()
        event = json.loads(line)

        assert event["quoted"] == 'String with "quotes"'
        assert event["backslash"] == "Path\\like\\this"
        assert event["newline"] == "Line1\nLine2"


def test_telemetry_no_empty_payload() -> None:
    """BriefingTelemetry.emit() works with no custom payload fields."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "events.jsonl"
        telemetry = BriefingTelemetry(path)

        telemetry.emit("bare_event")

        line = path.read_text().strip()
        event = json.loads(line)

        assert "timestamp" in event
        assert event["event_type"] == "bare_event"
        # Only timestamp and event_type should be present
        assert len(event) == 2


def test_telemetry_multiple_instances_same_file() -> None:
    """Multiple BriefingTelemetry instances can safely write to same file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "events.jsonl"

        # Create two telemetry instances
        telemetry1 = BriefingTelemetry(path)
        telemetry2 = BriefingTelemetry(path)

        # Interleave emissions
        telemetry1.emit("instance1_event", source=1)
        telemetry2.emit("instance2_event", source=2)
        telemetry1.emit("instance1_event2", source=1)

        lines = path.read_text().strip().split("\n")
        events = [json.loads(line) for line in lines]

        assert len(events) == 3
        assert events[0]["source"] == 1
        assert events[1]["source"] == 2
        assert events[2]["source"] == 1


def test_telemetry_large_payload() -> None:
    """BriefingTelemetry handles large payload correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "events.jsonl"
        telemetry = BriefingTelemetry(path)

        large_string = "x" * 10000
        large_list = list(range(1000))

        telemetry.emit(
            "large_payload",
            large_string=large_string,
            large_list=large_list,
        )

        line = path.read_text().strip()
        event = json.loads(line)

        assert event["large_string"] == large_string
        assert event["large_list"] == large_list


def test_telemetry_null_values_in_payload() -> None:
    """BriefingTelemetry correctly serializes None/null values."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "events.jsonl"
        telemetry = BriefingTelemetry(path)

        telemetry.emit("test_event", nullable_field=None, other="value")

        line = path.read_text().strip()
        event = json.loads(line)

        assert event["nullable_field"] is None
        assert event["other"] == "value"


def test_telemetry_numeric_types() -> None:
    """BriefingTelemetry preserves numeric types in JSON."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "events.jsonl"
        telemetry = BriefingTelemetry(path)

        telemetry.emit(
            "test_event",
            integer=42,
            floating=3.14159,
            zero=0,
            negative=-100,
        )

        line = path.read_text().strip()
        event = json.loads(line)

        assert event["integer"] == 42
        assert event["floating"] == 3.14159
        assert event["zero"] == 0
        assert event["negative"] == -100
