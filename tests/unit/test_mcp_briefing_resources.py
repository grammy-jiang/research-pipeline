"""Tests for MCP briefing resources (G05)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from research_pipeline.mcp_server import resources

BRIEFING_RESOURCES = [
    "list_briefings",
    "get_briefing_daily",
    "get_briefing_ranked",
    "get_briefing_telemetry",
    "get_briefing_validation",
    "get_briefing_workflow_state",
]


def test_all_briefing_resources_exist() -> None:
    for name in BRIEFING_RESOURCES:
        assert hasattr(resources, name), f"Missing briefing resource: {name}"
        assert callable(getattr(resources, name))


def _isolate_workspace(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    monkeypatch.setattr(resources, "DEFAULT_WORKSPACE", str(tmp_path))
    return tmp_path


def test_list_briefings_empty(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _isolate_workspace(monkeypatch, tmp_path)
    payload = json.loads(resources.list_briefings())
    assert payload["briefings"] == []
    assert "root" in payload


def test_list_briefings_with_dates(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _isolate_workspace(monkeypatch, tmp_path)
    (tmp_path / "briefings" / "2026-04-20").mkdir(parents=True)
    (tmp_path / "briefings" / "2026-04-21").mkdir(parents=True)
    payload = json.loads(resources.list_briefings())
    assert [b["date"] for b in payload["briefings"]] == ["2026-04-20", "2026-04-21"]
    for b in payload["briefings"]:
        assert b["daily_report"].endswith("reports/daily.md")
        assert b["validation"].endswith("validation/validation.json")


def test_get_briefing_daily_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _isolate_workspace(monkeypatch, tmp_path)
    payload = json.loads(resources.get_briefing_daily("2026-04-20"))
    assert "error" in payload


def test_get_briefing_daily_returns_text(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _isolate_workspace(monkeypatch, tmp_path)
    daily = tmp_path / "briefings" / "2026-04-20" / "reports" / "daily.md"
    daily.parent.mkdir(parents=True)
    daily.write_text("# Daily Brief 2026-04-20\n")
    out = resources.get_briefing_daily("2026-04-20")
    assert "# Daily Brief 2026-04-20" in out


def test_get_briefing_ranked(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _isolate_workspace(monkeypatch, tmp_path)
    rp = tmp_path / "briefings" / "2026-04-20" / "ranked" / "ranked_clusters.jsonl"
    rp.parent.mkdir(parents=True)
    rp.write_text('{"cluster_id": "c1"}\n')
    assert "c1" in resources.get_briefing_ranked("2026-04-20")


def test_get_briefing_ranked_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _isolate_workspace(monkeypatch, tmp_path)
    assert "error" in json.loads(resources.get_briefing_ranked("2026-04-20"))


def test_get_briefing_telemetry(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _isolate_workspace(monkeypatch, tmp_path)
    tp = tmp_path / "briefings" / "2026-04-20" / "telemetry.jsonl"
    tp.parent.mkdir(parents=True)
    tp.write_text('{"stage":"polled"}\n')
    assert "polled" in resources.get_briefing_telemetry("2026-04-20")
    # missing case
    assert "error" in json.loads(resources.get_briefing_telemetry("2099-01-01"))


def test_get_briefing_validation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _isolate_workspace(monkeypatch, tmp_path)
    vp = tmp_path / "briefings" / "2026-04-20" / "validation" / "validation.json"
    vp.parent.mkdir(parents=True)
    vp.write_text(json.dumps({"passed": True}))
    out = json.loads(resources.get_briefing_validation("2026-04-20"))
    assert out["passed"] is True
    # missing case
    assert "error" in json.loads(resources.get_briefing_validation("2099-01-01"))


def test_get_briefing_workflow_state(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _isolate_workspace(monkeypatch, tmp_path)
    wp = tmp_path / "briefings" / "2026-04-20" / "workflow_state.json"
    wp.parent.mkdir(parents=True)
    wp.write_text(json.dumps({"run_date": "2026-04-20", "current_stage": "validated"}))
    out = json.loads(resources.get_briefing_workflow_state("2026-04-20"))
    assert out["current_stage"] == "validated"
    # missing case
    assert "error" in json.loads(resources.get_briefing_workflow_state("2099-01-01"))


def test_briefing_resources_are_read_only(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Briefing resource handlers must not create or mutate files."""
    _isolate_workspace(monkeypatch, tmp_path)
    before = list(tmp_path.rglob("*"))
    resources.list_briefings()
    resources.get_briefing_daily("2026-04-20")
    resources.get_briefing_ranked("2026-04-20")
    resources.get_briefing_telemetry("2026-04-20")
    resources.get_briefing_validation("2026-04-20")
    resources.get_briefing_workflow_state("2026-04-20")
    after = list(tmp_path.rglob("*"))
    assert before == after
