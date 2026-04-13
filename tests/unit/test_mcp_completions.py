"""Tests for MCP completion handlers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from mcp.types import CompletionArgument, PromptReference

from mcp_server.completions import (
    _list_backends,
    _list_paper_ids,
    _list_run_ids,
    handle_completion,
)


class TestListRunIds:
    def test_empty_dirs(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        import mcp_server.completions as mod

        monkeypatch.setattr(mod, "DEFAULT_RUNS_DIRS", [str(tmp_path / "empty")])
        assert _list_run_ids() == []

    def test_finds_runs(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        import mcp_server.completions as mod

        runs = tmp_path / "runs"
        runs.mkdir()
        (runs / "run-001").mkdir()
        (runs / "run-002").mkdir()
        (runs / ".hidden").mkdir()
        monkeypatch.setattr(mod, "DEFAULT_RUNS_DIRS", [str(runs)])
        result = _list_run_ids()
        assert result == ["run-001", "run-002"]

    def test_prefix_filter(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import mcp_server.completions as mod

        runs = tmp_path / "runs"
        runs.mkdir()
        (runs / "alpha-001").mkdir()
        (runs / "beta-002").mkdir()
        monkeypatch.setattr(mod, "DEFAULT_RUNS_DIRS", [str(runs)])
        result = _list_run_ids("alpha")
        assert result == ["alpha-001"]


class TestListPaperIds:
    def test_finds_papers(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import mcp_server.completions as mod

        pdf_dir = tmp_path / "run-001" / "download" / "pdf"
        pdf_dir.mkdir(parents=True)
        (pdf_dir / "2401.12345.pdf").touch()
        (pdf_dir / "2401.67890.pdf").touch()
        monkeypatch.setattr(mod, "DEFAULT_RUNS_DIRS", [str(tmp_path)])
        result = _list_paper_ids("run-001")
        assert "2401.12345" in result
        assert "2401.67890" in result


class TestListBackends:
    def test_returns_list(self) -> None:
        backends = _list_backends()
        assert isinstance(backends, list)
        assert len(backends) > 0


class TestHandleCompletion:
    @pytest.mark.anyio
    async def test_direction_completion(self) -> None:
        ref = PromptReference(type="ref/prompt", name="test")
        arg = CompletionArgument(name="direction", value="c")
        result = await handle_completion(ref, arg)
        assert result is not None
        assert "citations" in result.values

    @pytest.mark.anyio
    async def test_source_completion(self) -> None:
        ref = PromptReference(type="ref/prompt", name="test")
        arg = CompletionArgument(name="source", value="ar")
        result = await handle_completion(ref, arg)
        assert result is not None
        assert "arxiv" in result.values

    @pytest.mark.anyio
    async def test_backend_completion(self) -> None:
        ref = PromptReference(type="ref/prompt", name="test")
        arg = CompletionArgument(name="backend", value="py")
        result = await handle_completion(ref, arg)
        assert result is not None
        assert "pymupdf4llm" in result.values

    @pytest.mark.anyio
    async def test_unknown_argument(self) -> None:
        ref = PromptReference(type="ref/prompt", name="test")
        arg = CompletionArgument(name="unknown_arg", value="x")
        result = await handle_completion(ref, arg)
        assert result is None

    @pytest.mark.anyio
    async def test_run_id_completion(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import mcp_server.completions as mod

        runs = tmp_path / "runs"
        runs.mkdir()
        (runs / "test-run-001").mkdir()
        monkeypatch.setattr(mod, "DEFAULT_RUNS_DIRS", [str(runs)])
        ref = PromptReference(type="ref/prompt", name="test")
        arg = CompletionArgument(name="run_id", value="test")
        result = await handle_completion(ref, arg)
        assert result is not None
        assert "test-run-001" in result.values

    @pytest.mark.anyio
    async def test_topic_completion(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import mcp_server.completions as mod

        run_dir = tmp_path / "runs" / "run-001" / "plan"
        run_dir.mkdir(parents=True)
        (run_dir / "query_plan.json").write_text(
            json.dumps({"topic": "transformer attention"})
        )
        monkeypatch.setattr(mod, "DEFAULT_RUNS_DIRS", [str(tmp_path / "runs")])
        ref = PromptReference(type="ref/prompt", name="test")
        arg = CompletionArgument(name="topic", value="trans")
        result = await handle_completion(ref, arg)
        assert result is not None
        assert "transformer attention" in result.values
