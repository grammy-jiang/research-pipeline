"""Tests for MCP tool implementations."""

import json
from pathlib import Path

from research_pipeline.mcp_server.schemas import (
    AnalyzePapersInput,
    ConvertFileInput,
    ConvertFineInput,
    ConvertPdfsInput,
    ConvertRoughInput,
    EvaluateQualityInput,
    ExpandCitationsInput,
    GetRunManifestInput,
    ManageIndexInput,
    PlanTopicInput,
    ScreenCandidatesInput,
    SearchInput,
)
from research_pipeline.mcp_server.tools import (
    _resolve_run_id,
    _resolve_workspace,
    analyze_papers,
    convert_file,
    convert_fine,
    convert_pdfs,
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

    def test_action_gc_drives_gc(self, tmp_path: Path) -> None:
        result = manage_index(
            ManageIndexInput(action="gc", db_path=str(tmp_path / "idx.db"))
        )
        assert result.success is True
        assert "removed" in result.artifacts
        assert "garbage collected" in result.message.lower()

    def test_action_list_drives_list(self, tmp_path: Path) -> None:
        result = manage_index(
            ManageIndexInput(action="list", db_path=str(tmp_path / "idx.db"))
        )
        assert result.success is True
        assert result.artifacts.get("count") == 0

    def test_action_wins_over_conflicting_gc_bool(self, tmp_path: Path) -> None:
        # action='list' must win over the deprecated gc=True (list, not gc).
        result = manage_index(
            ManageIndexInput(action="list", gc=True, db_path=str(tmp_path / "idx.db"))
        )
        assert result.success is True
        assert "count" in result.artifacts
        assert "removed" not in result.artifacts

    def test_empty_action_honours_gc_bool(self, tmp_path: Path) -> None:
        # With no explicit action, the deprecated gc bool still drives gc.
        result = manage_index(
            ManageIndexInput(gc=True, db_path=str(tmp_path / "idx.db"))
        )
        assert result.success is True
        assert "removed" in result.artifacts


class TestAnalyzePapers:
    def test_mode_collect_drives_collect(self, tmp_path: Path) -> None:
        result = analyze_papers(
            AnalyzePapersInput(
                workspace=str(tmp_path), run_id="test-analyze", mode="collect"
            )
        )
        assert result.success is False
        assert "no analysis json files" in result.message.lower()

    def test_mode_prepare_drives_prepare(self, tmp_path: Path) -> None:
        result = analyze_papers(
            AnalyzePapersInput(
                workspace=str(tmp_path), run_id="test-analyze", mode="prepare"
            )
        )
        assert result.success is False
        assert "no converted papers" in result.message.lower()

    def test_mode_wins_over_conflicting_collect_bool(self, tmp_path: Path) -> None:
        # mode='prepare' must win over the deprecated collect=True (prepare path).
        result = analyze_papers(
            AnalyzePapersInput(
                workspace=str(tmp_path),
                run_id="test-analyze",
                mode="prepare",
                collect=True,
            )
        )
        assert result.success is False
        assert "no converted papers" in result.message.lower()

    def test_empty_mode_honours_collect_bool(self, tmp_path: Path) -> None:
        # With no explicit mode, the deprecated collect bool still drives collect.
        result = analyze_papers(
            AnalyzePapersInput(
                workspace=str(tmp_path), run_id="test-analyze", collect=True
            )
        )
        assert result.success is False
        assert "no analysis json files" in result.message.lower()

    def test_empty_mode_defaults_to_prepare(self, tmp_path: Path) -> None:
        # No mode and no collect bool → prepare path.
        result = analyze_papers(
            AnalyzePapersInput(workspace=str(tmp_path), run_id="test-analyze")
        )
        assert result.success is False
        assert "no converted papers" in result.message.lower()


class TestConvertPdfs:
    def test_no_download_manifest(self, tmp_path: Path) -> None:
        result = convert_pdfs(
            ConvertPdfsInput(
                workspace=str(tmp_path),
                run_id="test-convert",
            )
        )
        assert result.success is False
        assert "download" in result.message.lower()

    def test_uses_fallback_converter(self, tmp_path: Path) -> None:
        """convert_pdfs delegates to create_converter, enabling FallbackConverter."""
        from unittest.mock import MagicMock, patch

        run_root = tmp_path / "test-convert"
        dl_root = run_root / "download"
        dl_root.mkdir(parents=True)
        manifest_path = dl_root / "download_manifest.jsonl"
        manifest_path.write_text(
            '{"arxiv_id": "2401.00001", "version": "v1",'
            ' "pdf_url": "http://example.com/2401.00001.pdf",'
            ' "local_path": "/fake/2401.00001.pdf",'
            ' "sha256": "abc123", "size_bytes": 1000,'
            ' "downloaded_at": "2024-01-01T00:00:00Z",'
            ' "status": "downloaded"}\n',
            encoding="utf-8",
        )

        with (
            patch(
                "research_pipeline.conversion.factory.create_converter"
            ) as mock_create,
            patch("research_pipeline.config.loader.load_config") as mock_cfg,
        ):
            mock_cfg.return_value = MagicMock(
                workspace=str(tmp_path),
                conversion=MagicMock(backend="pymupdf4llm", fallback_backends=[]),
            )
            mock_converter = MagicMock()
            mock_create.return_value = mock_converter
            mock_converter.convert.return_value = MagicMock(
                status="converted", model_dump=lambda mode=None: {"status": "converted"}
            )

            convert_pdfs(
                ConvertPdfsInput(
                    workspace=str(tmp_path),
                    run_id="test-convert",
                )
            )

        mock_create.assert_called_once()


class TestScrubExc:
    """Tool error messages must not leak internal filesystem paths (#44)."""

    def test_redacts_home_dir(self) -> None:
        from pathlib import Path

        from research_pipeline.mcp_server.tools import _scrub_exc

        home = str(Path.home())
        msg = _scrub_exc(FileNotFoundError(f"{home}/runs/secret/paper.pdf missing"))
        assert home not in msg
        assert "paper.pdf" in msg

    def test_redacts_credentials_in_exception(self) -> None:
        from research_pipeline.mcp_server.tools import _scrub_exc

        fake_key = "sk-" + "0" * 24  # obviously-fake, low-entropy
        msg = _scrub_exc(RuntimeError(f"auth failed for {fake_key}"))
        assert fake_key not in msg
        assert "[REDACTED]" in msg
