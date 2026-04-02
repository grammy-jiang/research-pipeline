"""Tests for MCP server schemas."""

import pytest
from pydantic import ValidationError

from mcp_server.schemas import (
    CommonParams,
    ConvertFileInput,
    ConvertPdfsInput,
    DownloadPdfsInput,
    ExtractContentInput,
    GetRunManifestInput,
    PlanTopicInput,
    RunPipelineInput,
    ScreenCandidatesInput,
    SearchInput,
    SummarizePapersInput,
    ToolResult,
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
