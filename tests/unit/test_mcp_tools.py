"""Tests for MCP tool implementations."""

import json
from pathlib import Path

from mcp_server.schemas import (
    ConvertFileInput,
    ConvertFineInput,
    ConvertRoughInput,
    EvaluateQualityInput,
    ExpandCitationsInput,
    GetRunManifestInput,
    ManageIndexInput,
    PlanTopicInput,
    ScreenCandidatesInput,
    SearchInput,
)
from mcp_server.tools import (
    _resolve_run_id,
    _resolve_workspace,
    convert_file,
    convert_fine,
    convert_rough,
    evaluate_quality,
    expand_citations,
    get_run_manifest,
    manage_index,
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


class TestExpandCitations:
    def test_no_paper_ids(self, tmp_path: Path) -> None:
        result = expand_citations(
            ExpandCitationsInput(
                paper_ids=[],
                workspace=str(tmp_path),
                run_id="test-expand",
            )
        )
        assert result.success is False
        assert "paper" in result.message.lower()


class TestEvaluateQuality:
    def test_no_candidates(self, tmp_path: Path) -> None:
        result = evaluate_quality(
            EvaluateQualityInput(
                workspace=str(tmp_path),
                run_id="test-quality",
            )
        )
        assert result.success is False
        assert (
            "candidates" in result.message.lower() or "search" in result.message.lower()
        )


class TestConvertRough:
    def test_no_download_manifest(self, tmp_path: Path) -> None:
        result = convert_rough(
            ConvertRoughInput(
                workspace=str(tmp_path),
                run_id="test-rough",
            )
        )
        assert result.success is False
        assert "download" in result.message.lower()


class TestConvertFine:
    def test_no_paper_ids(self, tmp_path: Path) -> None:
        result = convert_fine(
            ConvertFineInput(
                paper_ids=[],
                workspace=str(tmp_path),
                run_id="test-fine",
            )
        )
        assert result.success is False
        assert "paper" in result.message.lower()

    def test_no_download_manifest(self, tmp_path: Path) -> None:
        result = convert_fine(
            ConvertFineInput(
                paper_ids=["2401.12345"],
                workspace=str(tmp_path),
                run_id="test-fine",
            )
        )
        assert result.success is False
        assert "download" in result.message.lower()


class TestManageIndex:
    def test_default_usage_message(self, tmp_path: Path) -> None:
        result = manage_index(ManageIndexInput(db_path=str(tmp_path / "test_index.db")))
        assert result.success is True
        assert "list_papers" in result.message.lower() or "gc" in result.message.lower()

    def test_list_empty_index(self, tmp_path: Path) -> None:
        result = manage_index(
            ManageIndexInput(
                list_papers=True,
                db_path=str(tmp_path / "test_index.db"),
            )
        )
        assert result.success is True
        assert result.artifacts.get("count") == 0
