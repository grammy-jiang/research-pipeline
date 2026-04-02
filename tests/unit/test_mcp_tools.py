"""Tests for MCP tool implementations."""

import json
from pathlib import Path

from mcp_server.schemas import (
    ConvertFileInput,
    GetRunManifestInput,
    PlanTopicInput,
    ScreenCandidatesInput,
    SearchInput,
)
from mcp_server.tools import (
    _resolve_run_id,
    _resolve_workspace,
    convert_file,
    get_run_manifest,
    plan_topic,
    screen_candidates,
    search,
)


class TestResolveWorkspace:
    def test_relative_path(self) -> None:
        result = _resolve_workspace("./workspace")
        assert result.is_absolute()
        assert result.name == "workspace"

    def test_absolute_path(self) -> None:
        result = _resolve_workspace("/tmp/ws")
        assert result == Path("/tmp/ws")

    def test_tilde_expansion(self) -> None:
        result = _resolve_workspace("~/workspace")
        assert result.is_absolute()
        assert "~" not in str(result)


class TestResolveRunId:
    def test_provided_id(self) -> None:
        assert _resolve_run_id("my-run") == "my-run"

    def test_auto_generate(self) -> None:
        rid = _resolve_run_id("")
        assert rid  # non-empty
        assert "T" in rid  # ISO-ish format


class TestPlanTopic:
    def test_creates_plan(self, tmp_path: Path) -> None:
        result = plan_topic(
            PlanTopicInput(
                topic="multimodal RAG",
                workspace=str(tmp_path),
                run_id="test-run",
            )
        )
        assert result.success is True
        assert "test-run" in result.artifacts.get("run_id", "")

        plan_path = Path(result.artifacts["query_plan"])
        assert plan_path.exists()
        plan_data = json.loads(plan_path.read_text())
        assert plan_data["topic_raw"] == "multimodal RAG"

    def test_empty_topic(self, tmp_path: Path) -> None:
        result = plan_topic(
            PlanTopicInput(
                topic="",
                workspace=str(tmp_path),
                run_id="test-empty",
            )
        )
        # Empty topic still creates a plan (query builder handles it)
        assert result.success is True


class TestSearch:
    def test_no_plan_fails(self, tmp_path: Path) -> None:
        result = search(
            SearchInput(
                workspace=str(tmp_path),
                run_id="no-plan-run",
            )
        )
        assert result.success is False
        assert "plan" in result.message.lower()


class TestScreenCandidates:
    def test_no_candidates_fails(self, tmp_path: Path) -> None:
        result = screen_candidates(
            ScreenCandidatesInput(
                workspace=str(tmp_path),
                run_id="no-search-run",
            )
        )
        assert result.success is False
        assert (
            "candidates" in result.message.lower() or "search" in result.message.lower()
        )


class TestGetRunManifest:
    def test_no_manifest(self, tmp_path: Path) -> None:
        result = get_run_manifest(
            GetRunManifestInput(
                workspace=str(tmp_path),
                run_id="nonexistent",
            )
        )
        assert result.success is False


class TestConvertFile:
    def test_missing_file(self) -> None:
        result = convert_file(ConvertFileInput(pdf_path="/tmp/nonexistent_paper.pdf"))
        assert result.success is False
        assert "not found" in result.message.lower()
