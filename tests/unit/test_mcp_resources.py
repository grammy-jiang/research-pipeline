"""Tests for MCP resource handlers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mcp_server import resources
from mcp_server.server import mcp


class TestResourceRegistration:
    def test_resource_count(self) -> None:
        """All 15 resources should be registered."""
        registered = mcp._resource_manager._resources
        templates = mcp._resource_manager._templates
        total = len(registered) + len(templates)
        assert total == 15, (
            f"Expected 15 resources/templates, got {total}: "
            f"{len(registered)} static + {len(templates)} templates"
        )

    def test_static_resources(self) -> None:
        """Static resources (no URI params) should be registered."""
        registered = set(str(u) for u in mcp._resource_manager._resources)
        expected_static = {"runs://list", "config://current", "index://papers"}
        assert expected_static.issubset(registered), (
            f"Missing static resources: {expected_static - registered}"
        )


class TestListRuns:
    def test_empty_workspace(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(resources, "DEFAULT_RUNS_DIR", str(tmp_path / "runs"))
        monkeypatch.setattr(resources, "DEFAULT_WORKSPACE", str(tmp_path / "workspace"))
        result = json.loads(resources.list_runs())
        assert result["runs"] == []

    def test_with_runs(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        runs_dir = tmp_path / "runs"
        runs_dir.mkdir()
        (runs_dir / "run-001").mkdir()
        manifest = {"topic": "test topic", "stages": {"plan": {}, "search": {}}}
        (runs_dir / "run-001" / "run_manifest.json").write_text(json.dumps(manifest))
        monkeypatch.setattr(resources, "DEFAULT_RUNS_DIR", str(runs_dir))
        result = json.loads(resources.list_runs())
        assert len(result["runs"]) == 1
        assert result["runs"][0]["run_id"] == "run-001"
        assert result["runs"][0]["topic"] == "test topic"


class TestGetRunManifest:
    def test_missing_run(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(resources, "DEFAULT_RUNS_DIR", str(tmp_path))
        result = json.loads(resources.get_run_manifest("nonexistent"))
        assert "error" in result

    def test_existing_manifest(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run_dir = tmp_path / "test-run"
        run_dir.mkdir()
        manifest = {"topic": "test", "stages": {}}
        (run_dir / "run_manifest.json").write_text(json.dumps(manifest))
        monkeypatch.setattr(resources, "DEFAULT_RUNS_DIR", str(tmp_path))
        result = json.loads(resources.get_run_manifest("test-run"))
        assert result["topic"] == "test"


class TestGetRunPlan:
    def test_missing_plan(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run_dir = tmp_path / "test-run"
        run_dir.mkdir()
        monkeypatch.setattr(resources, "DEFAULT_RUNS_DIR", str(tmp_path))
        result = json.loads(resources.get_run_plan("test-run"))
        assert "error" in result


class TestGetRunCandidates:
    def test_existing_candidates(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run_dir = tmp_path / "test-run" / "search"
        run_dir.mkdir(parents=True)
        (run_dir / "candidates.jsonl").write_text('{"id": "1"}\n{"id": "2"}\n')
        monkeypatch.setattr(resources, "DEFAULT_RUNS_DIR", str(tmp_path))
        result = resources.get_run_candidates("test-run")
        assert '{"id": "1"}' in result


class TestGetPaperMarkdown:
    def test_from_convert_markdown(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        md_dir = tmp_path / "test-run" / "convert" / "markdown"
        md_dir.mkdir(parents=True)
        (md_dir / "2401.12345.md").write_text("# Test Paper")
        monkeypatch.setattr(resources, "DEFAULT_RUNS_DIR", str(tmp_path))
        result = resources.get_paper_markdown("test-run", "2401.12345")
        assert result == "# Test Paper"


class TestGetCurrentConfig:
    def test_fallback_to_example(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Should return something (either config.toml or config.example.toml)."""
        result = resources.get_current_config()
        # Should not be empty; example config exists in repo
        assert len(result) > 0


class TestGetGlobalIndex:
    def test_returns_json(self) -> None:
        result = json.loads(resources.get_global_index())
        assert "count" in result
        assert "papers" in result
