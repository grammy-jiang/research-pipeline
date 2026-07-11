"""Tests for MCP server schemas."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from research_pipeline.mcp_server.schemas import (
    _MCP_ROOT_ENV,
    AnalyzePapersInput,
    CommonParams,
    ConvertFileInput,
    ConvertFineInput,
    ConvertPdfsInput,
    ConvertRoughInput,
    DownloadPdfsInput,
    EvaluateQualityInput,
    ExpandCitationsInput,
    ExtractContentInput,
    GetRunManifestInput,
    ManageIndexInput,
    PlanTopicInput,
    RunPipelineInput,
    ScreenCandidatesInput,
    SearchInput,
    SummarizePapersInput,
    ToolResult,
    _validate_tool_path,
)


class TestCommonParams:
    def test_defaults(self) -> None:
        p = CommonParams()
        assert p.workspace == "./workspace"
        assert p.run_id == ""

    def test_custom_values(self) -> None:
        p = CommonParams(workspace="/tmp/ws", run_id="abc123")
        assert p.workspace == "/tmp/ws"
        assert p.run_id == "abc123"


class TestPlanTopicInput:
    def test_requires_topic(self) -> None:
        with pytest.raises(ValidationError):
            PlanTopicInput()  # type: ignore[call-arg]

    def test_valid(self) -> None:
        p = PlanTopicInput(topic="RAG for long docs")
        assert p.topic == "RAG for long docs"
        assert p.workspace == "./workspace"


class TestSearchInput:
    def test_defaults(self) -> None:
        p = SearchInput()
        assert p.topic == ""
        assert p.resume is False
        assert p.source == ""

    def test_custom_source(self) -> None:
        p = SearchInput(source="scholar")
        assert p.source == "scholar"


class TestScreenCandidatesInput:
    def test_defaults(self) -> None:
        p = ScreenCandidatesInput()
        assert p.resume is False


class TestDownloadPdfsInput:
    def test_defaults(self) -> None:
        p = DownloadPdfsInput()
        assert p.force is False


class TestConvertPdfsInput:
    def test_defaults(self) -> None:
        p = ConvertPdfsInput()
        assert p.force is False


class TestExtractContentInput:
    def test_defaults(self) -> None:
        p = ExtractContentInput()
        assert p.workspace == "./workspace"


class TestSummarizePapersInput:
    def test_defaults(self) -> None:
        p = SummarizePapersInput()
        assert p.workspace == "./workspace"


class TestRunPipelineInput:
    def test_requires_topic(self) -> None:
        with pytest.raises(ValidationError):
            RunPipelineInput()  # type: ignore[call-arg]

    def test_valid(self) -> None:
        p = RunPipelineInput(topic="multimodal RAG")
        assert p.topic == "multimodal RAG"
        assert p.resume is False


class TestGetRunManifestInput:
    def test_defaults(self) -> None:
        p = GetRunManifestInput()
        assert p.workspace == "./workspace"
        assert p.run_id == ""


class TestConvertFileInput:
    def test_requires_pdf_path(self) -> None:
        with pytest.raises(ValidationError):
            ConvertFileInput()  # type: ignore[call-arg]

    def test_valid(self) -> None:
        p = ConvertFileInput(pdf_path="/tmp/paper.pdf")
        assert p.pdf_path == "/tmp/paper.pdf"
        assert p.output_dir == ""


class TestToolResult:
    def test_success(self) -> None:
        r = ToolResult(success=True, message="OK")
        assert r.success is True
        assert r.artifacts == {}

    def test_with_artifacts(self) -> None:
        r = ToolResult(
            success=True,
            message="Done",
            artifacts={"path": "/tmp/out.json"},
        )
        assert r.artifacts["path"] == "/tmp/out.json"

    def test_failure(self) -> None:
        r = ToolResult(success=False, message="Error occurred")
        assert r.success is False


class TestExpandCitationsInput:
    def test_requires_paper_ids(self) -> None:
        with pytest.raises(ValidationError):
            ExpandCitationsInput()  # type: ignore[call-arg]

    def test_valid(self) -> None:
        p = ExpandCitationsInput(paper_ids=["2401.12345"])
        assert p.paper_ids == ["2401.12345"]
        assert p.direction == "both"
        assert p.limit == 50

    def test_custom_direction(self) -> None:
        p = ExpandCitationsInput(
            paper_ids=["2401.12345"], direction="citations", limit=20
        )
        assert p.direction == "citations"
        assert p.limit == 20


class TestEvaluateQualityInput:
    def test_defaults(self) -> None:
        p = EvaluateQualityInput()
        assert p.workspace == "./workspace"
        assert p.run_id == ""


class TestConvertRoughInput:
    def test_defaults(self) -> None:
        p = ConvertRoughInput()
        assert p.force is False
        assert p.workspace == "./workspace"


class TestConvertFineInput:
    def test_requires_paper_ids(self) -> None:
        with pytest.raises(ValidationError):
            ConvertFineInput()  # type: ignore[call-arg]

    def test_valid(self) -> None:
        p = ConvertFineInput(paper_ids=["2401.12345"])
        assert p.paper_ids == ["2401.12345"]
        assert p.force is False
        assert p.backend == ""

    def test_with_backend(self) -> None:
        p = ConvertFineInput(paper_ids=["2401.12345"], backend="marker")
        assert p.backend == "marker"


class TestManageIndexInput:
    def test_defaults(self) -> None:
        p = ManageIndexInput()
        assert p.list_papers is False
        assert p.gc is False
        assert p.db_path == ""

    def test_list_mode(self) -> None:
        p = ManageIndexInput(list_papers=True)
        assert p.list_papers is True

    def test_gc_mode(self) -> None:
        p = ManageIndexInput(gc=True)
        assert p.gc is True

    def test_action_defaults_empty(self) -> None:
        assert ManageIndexInput().action == ""

    def test_action_accepts_list_and_gc(self) -> None:
        assert ManageIndexInput(action="list").action == "list"
        assert ManageIndexInput(action="gc").action == "gc"

    def test_action_rejects_unknown(self) -> None:
        with pytest.raises(ValidationError):
            ManageIndexInput(action="delete")


class TestAnalyzePapersInput:
    def test_defaults(self) -> None:
        p = AnalyzePapersInput()
        assert p.mode == ""
        assert p.collect is False
        assert p.paper_ids == []

    def test_mode_accepts_prepare_and_collect(self) -> None:
        assert AnalyzePapersInput(mode="prepare").mode == "prepare"
        assert AnalyzePapersInput(mode="collect").mode == "collect"

    def test_mode_rejects_unknown(self) -> None:
        with pytest.raises(ValidationError):
            AnalyzePapersInput(mode="analyze")


class TestRunIdTraversalValidator:
    """run_id feeds a filesystem path, so traversal must be rejected (#40)."""

    def test_rejects_traversal(self) -> None:
        for bad in ("../../etc", "a/../b", "/abs/run", "x\x00y"):
            with pytest.raises(ValidationError):
                PlanTopicInput(topic="x", run_id=bad)

    def test_allows_normal_and_empty(self) -> None:
        assert PlanTopicInput(topic="x", run_id="run-001").run_id == "run-001"
        # empty is the auto-generate sentinel
        assert PlanTopicInput(topic="x").run_id == ""


class TestValidateToolPath:
    """The PathStr validator confines every path-shaped tool argument (#103)."""

    def test_allows_empty(self) -> None:
        assert _validate_tool_path("") == ""

    def test_allows_normal_relative(self) -> None:
        assert _validate_tool_path("workspace/run1") == "workspace/run1"

    def test_allows_absolute_without_root(self, tmp_path: Path) -> None:
        assert _validate_tool_path(str(tmp_path)) == str(tmp_path)

    @pytest.mark.parametrize("bad", ["../etc/passwd", "a/../../b", "..", "foo/.."])
    def test_rejects_traversal(self, bad: str) -> None:
        with pytest.raises(ValueError, match="traversal"):
            _validate_tool_path(bad)

    def test_rejects_nul_byte(self) -> None:
        with pytest.raises(ValueError, match="NUL"):
            _validate_tool_path("a\x00b")

    def test_configured_root_allows_inside(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(_MCP_ROOT_ENV, str(tmp_path))
        inside = str(tmp_path / "sub" / "file.pdf")
        assert _validate_tool_path(inside) == inside

    def test_configured_root_rejects_outside(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(_MCP_ROOT_ENV, str(tmp_path))
        with pytest.raises(ValueError, match="escapes"):
            _validate_tool_path("/etc/passwd")


class TestSchemaPathContainment:
    """Path-shaped schema fields reject traversal at construction (#103)."""

    def test_convert_file_rejects_traversal_pdf_path(self) -> None:
        with pytest.raises(ValidationError):
            ConvertFileInput(pdf_path="../../etc/passwd")

    def test_convert_file_rejects_traversal_output_dir(self) -> None:
        with pytest.raises(ValidationError):
            ConvertFileInput(pdf_path="paper.pdf", output_dir="../../tmp")

    def test_convert_file_accepts_normal(self, tmp_path: Path) -> None:
        model = ConvertFileInput(pdf_path=str(tmp_path / "x.pdf"))
        assert model.pdf_path.endswith("x.pdf")

    def test_search_rejects_traversal_workspace(self) -> None:
        with pytest.raises(ValidationError):
            SearchInput(workspace="../secret")

    def test_manage_index_rejects_traversal_db_path(self) -> None:
        with pytest.raises(ValidationError):
            ManageIndexInput(db_path="../../root/.ssh/id_rsa")
