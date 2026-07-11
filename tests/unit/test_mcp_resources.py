"""Tests for MCP resource handlers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from research_pipeline.mcp_server import resources
from research_pipeline.mcp_server.server import mcp


class TestResourceRegistration:
    def test_resource_count(self) -> None:
        """All resources should be registered."""
        registered = mcp._resource_manager._resources
        templates = mcp._resource_manager._templates
        total = len(registered) + len(templates)
        assert total == 21, (
            f"Expected 21 resources/templates, got {total}: "
            f"{len(registered)} static + {len(templates)} templates"
        )

    def test_static_resources(self) -> None:
        """Static resources (no URI params) should be registered."""
        registered = {str(u) for u in mcp._resource_manager._resources}
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
        monkeypatch.setattr(resources, "DEFAULT_WORKSPACE", str(tmp_path))
        # A missing run must raise (surfaced as a JSON-RPC error), not return
        # a success-shaped error blob. See #42.
        with pytest.raises(ValueError, match="not found"):
            resources.get_run_manifest("nonexistent")

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
        with pytest.raises(ValueError, match="No plan"):
            resources.get_run_plan("test-run")


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


class TestPathConfinement:
    """Caller-supplied identifiers cannot escape the workspace root (#40)."""

    def test_validate_id_rejects_traversal(self) -> None:
        for bad in ("../etc", "a/../../b", "/abs", "", "x\x00y"):
            with pytest.raises(ValueError, match="Invalid"):
                resources._validate_id(bad, "run_id")

    def test_validate_id_allows_normal(self) -> None:
        assert resources._validate_id("run-001", "run_id") == "run-001"
        # old-style arXiv ids contain a slash but no traversal
        assert resources._validate_id("hep-th/9901001", "paper_id") == "hep-th/9901001"

    def test_safe_join_contains(self, tmp_path: Path) -> None:
        assert (
            resources._safe_join(tmp_path, "sub", "f.txt")
            == (tmp_path / "sub" / "f.txt").resolve()
        )

    def test_safe_join_rejects_escape(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="escapes root"):
            resources._safe_join(tmp_path, "..", "..", "etc")

    def test_get_run_root_rejects_traversal(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(resources, "DEFAULT_RUNS_DIR", str(tmp_path))
        with pytest.raises(ValueError, match="Invalid run_id"):
            resources._get_run_root("../../etc")

    def test_paper_reader_rejects_traversal(self) -> None:
        with pytest.raises(ValueError, match="Invalid paper_id"):
            resources.get_paper_markdown("run", "../../secret")


class TestResourceCaps:
    """Oversized resource reads are size-capped (#44)."""

    def test_cap_text_passthrough_under_limit(self) -> None:
        assert resources._cap_text("hello", "x") == "hello"

    def test_cap_text_truncates_over_limit(self) -> None:
        big = "a" * (resources._MAX_RESOURCE_BYTES + 1000)
        out = resources._cap_text(big, "big.md")
        assert len(out.encode("utf-8")) < len(big.encode("utf-8"))
        assert "truncated" in out
        assert "big.md" in out

    def test_cap_bytes_passthrough_under_limit(self) -> None:
        assert resources._cap_bytes(b"pdf", "x") == b"pdf"

    def test_cap_bytes_truncates_over_limit(self) -> None:
        big = b"a" * (resources._MAX_RESOURCE_BYTES + 1000)
        assert (
            len(resources._cap_bytes(big, "big.pdf")) == resources._MAX_RESOURCE_BYTES
        )


class TestCapJsonl:
    """JSONL truncation must stay on record boundaries (#121)."""

    def test_under_cap_is_unchanged(self) -> None:
        text = '{"a": 1}\n{"b": 2}\n'
        assert resources._cap_jsonl(text, "candidates") == text

    def test_keeps_only_whole_records(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(resources, "_MAX_RESOURCE_BYTES", 60)
        text = "".join(json.dumps({"i": i, "pad": "xxxx"}) + "\n" for i in range(20))
        out = resources._cap_jsonl(text, "candidates")
        # Every returned line is still valid JSON — no mid-record slice, no prose.
        assert out  # kept at least one record
        for line in out.splitlines():
            json.loads(line)
        assert len(out.encode("utf-8")) <= 60


class TestResourceNotFoundBoundary:
    """Resource-read failures surface the spec -32002 error + uri (#121)."""

    def test_missing_run_raises_mcp_error_32002(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from mcp.shared.exceptions import McpError

        from research_pipeline.mcp_server import server

        monkeypatch.setattr(resources, "DEFAULT_RUNS_DIR", str(tmp_path))
        monkeypatch.setattr(resources, "DEFAULT_WORKSPACE", str(tmp_path))
        with pytest.raises(McpError) as excinfo:
            server.resource_run_manifest(run_id="does-not-exist")
        err = excinfo.value.error
        assert err.code == -32002
        assert err.data == {"uri": "runs://does-not-exist/manifest"}

    def test_paper_resource_resolves_both_params_in_uri(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from mcp.shared.exceptions import McpError

        from research_pipeline.mcp_server import server

        monkeypatch.setattr(resources, "DEFAULT_RUNS_DIR", str(tmp_path))
        with pytest.raises(McpError) as excinfo:
            server.resource_paper_markdown(run_id="r1", paper_id="2401.00001")
        assert excinfo.value.error.data == {"uri": "runs://r1/markdown/2401.00001"}


class TestListResourceCaps:
    """List resources are size-capped like every other large read (#121)."""

    def test_list_runs_is_capped(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        runs_dir = tmp_path / "runs"
        runs_dir.mkdir()
        for i in range(50):
            run = runs_dir / f"run-{i:04d}"
            run.mkdir()
            (run / "run_manifest.json").write_text(json.dumps({"topic": "x" * 200}))
        monkeypatch.setattr(resources, "DEFAULT_RUNS_DIR", str(runs_dir))
        monkeypatch.setattr(resources, "_MAX_RESOURCE_BYTES", 500)
        out = resources.list_runs()
        assert "truncated" in out
        assert len(out.encode("utf-8")) < 5000
