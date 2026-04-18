"""Tests for CLI handlers — batch 2 (coverage boost).

Covers 17 previously-untested modules to raise overall coverage.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import click.exceptions
import pytest


# ── helpers ──────────────────────────────────────────────────────────────
def _make_config(tmp_path: Path) -> MagicMock:
    """Return a minimal mock PipelineConfig whose workspace is *tmp_path*."""
    cfg = MagicMock()
    cfg.workspace = str(tmp_path)
    cfg.contact_email = "test@test.com"
    cfg.cache.enabled = False
    cfg.sources.enabled = ["arxiv"]
    cfg.sources.scholar_backend = "scholarly"
    cfg.sources.scholar_min_interval = 0.0
    cfg.sources.serpapi_key = ""
    cfg.sources.serpapi_min_interval = 0.0
    cfg.sources.semantic_scholar_api_key = ""
    cfg.sources.semantic_scholar_min_interval = 0.0
    cfg.sources.openalex_api_key = ""
    cfg.sources.openalex_min_interval = 0.0
    cfg.sources.dblp_min_interval = 0.0
    cfg.sources.huggingface_min_interval = 0.0
    cfg.sources.huggingface_limit = 10
    cfg.arxiv.min_interval_seconds = 0.0
    cfg.arxiv.default_page_size = 10
    cfg.arxiv.base_url = "http://example.com"
    cfg.arxiv.request_timeout_seconds = 5
    cfg.download.max_per_run = 10
    cfg.quality.citation_weight = 0.25
    cfg.quality.venue_weight = 0.25
    cfg.quality.author_weight = 0.25
    cfg.quality.recency_weight = 0.25
    cfg.quality.venue_data_path = None
    cfg.conversion.backend = "pymupdf4llm"
    cfg.llm = MagicMock()
    return cfg


def _dl_entry_dict(
    arxiv_id: str = "2301.00001",
    version: str = "v1",
    status: str = "downloaded",
    local_path: str = "/fake.pdf",
) -> dict:
    """Return a dict valid for DownloadManifestEntry.model_validate."""
    return {
        "arxiv_id": arxiv_id,
        "version": version,
        "status": status,
        "local_path": local_path,
        "pdf_url": "http://example.com/paper.pdf",
        "sha256": "0" * 64,
        "size_bytes": 1024,
        "downloaded_at": "2024-01-01T00:00:00Z",
    }


def _candidate_dict(arxiv_id: str = "2301.00001", **overrides) -> dict:
    """Return a dict valid for CandidateRecord.model_validate."""
    base = {
        "arxiv_id": arxiv_id,
        "version": "v1",
        "title": "Test Paper",
        "authors": ["Author A"],
        "published": "2023-01-01T00:00:00Z",
        "updated": "2023-01-01T00:00:00Z",
        "categories": ["cs.AI"],
        "primary_category": "cs.AI",
        "abstract": "Test abstract",
        "abs_url": "http://example.com/abs",
        "pdf_url": "http://example.com/pdf",
    }
    base.update(overrides)
    return base


def _conv_entry_dict(
    arxiv_id: str = "2301.00001",
    status: str = "converted",
    markdown_path: str = "/fake.md",
    **overrides,
) -> dict:
    """Return a dict valid for ConvertManifestEntry.model_validate."""
    base = {
        "arxiv_id": arxiv_id,
        "version": "v1",
        "pdf_path": "/fake.pdf",
        "pdf_sha256": "0" * 64,
        "markdown_path": markdown_path,
        "converter_name": "pymupdf4llm",
        "converter_version": "1.0.0",
        "converter_config_hash": "abc123",
        "converted_at": "2024-01-01T00:00:00Z",
        "status": status,
    }
    base.update(overrides)
    return base


def _init_run_side_effect(ws: Path, run_id: str | None = None):
    """Side effect for init_run that creates the run dir."""
    rid = run_id or "test-run"
    run_root = ws / rid
    run_root.mkdir(parents=True, exist_ok=True)
    return rid, run_root


# ── 1. cmd_cite_context ──────────────────────────────────────────────────
class TestCmdCiteContext:
    """Tests for cmd_cite_context."""

    @patch("research_pipeline.cli.cmd_cite_context.init_run")
    @patch("research_pipeline.cli.cmd_cite_context.load_config")
    def test_missing_run_dir_exits(self, mock_cfg, mock_init, tmp_path):
        from research_pipeline.cli.cmd_cite_context import cite_context_command

        mock_cfg.return_value = _make_config(tmp_path)
        # init_run returns a non-existent dir
        mock_init.return_value = ("run1", tmp_path / "nonexistent")

        with pytest.raises(click.exceptions.Exit):
            cite_context_command(
                run_id="run1",
                context_window=1,
                output=None,
                config_path=Path("config.toml"),
            )

    @patch("research_pipeline.cli.cmd_cite_context.init_run")
    @patch("research_pipeline.cli.cmd_cite_context.load_config")
    def test_no_md_files_exits(self, mock_cfg, mock_init, tmp_path):
        from research_pipeline.cli.cmd_cite_context import cite_context_command

        mock_cfg.return_value = _make_config(tmp_path)
        run_dir = tmp_path / "run1"
        run_dir.mkdir()
        mock_init.return_value = ("run1", run_dir)
        # Create convert dir but no .md files
        (run_dir / "convert").mkdir(parents=True)

        with pytest.raises(click.exceptions.Exit):
            cite_context_command(
                run_id="run1",
                context_window=1,
                output=None,
                config_path=Path("config.toml"),
            )

    @patch("research_pipeline.cli.cmd_cite_context.group_by_marker")
    @patch("research_pipeline.cli.cmd_cite_context.contexts_to_dicts")
    @patch("research_pipeline.cli.cmd_cite_context.extract_citation_contexts")
    @patch("research_pipeline.cli.cmd_cite_context.init_run")
    @patch("research_pipeline.cli.cmd_cite_context.load_config")
    def test_happy_path(
        self, mock_cfg, mock_init, mock_extract, mock_to_dicts, mock_group, tmp_path
    ):
        from research_pipeline.cli.cmd_cite_context import cite_context_command

        mock_cfg.return_value = _make_config(tmp_path)
        run_dir = tmp_path / "run1"
        run_dir.mkdir()
        mock_init.return_value = ("run1", run_dir)

        # "convert" stage maps to "convert/markdown" in STAGE_SUBDIRS
        convert_dir = run_dir / "convert" / "markdown"
        convert_dir.mkdir(parents=True)
        md_file = convert_dir / "paper1.md"
        md_file.write_text("Some text [1] citing something.", encoding="utf-8")

        ctx_obj = MagicMock()
        mock_extract.return_value = [ctx_obj]
        mock_to_dicts.return_value = [{"marker": "[1]", "text": "citing"}]
        mock_group.return_value = {"[1]": [ctx_obj]}

        out_file = tmp_path / "out.json"
        cite_context_command(
            run_id="run1",
            context_window=1,
            output=out_file,
            config_path=Path("config.toml"),
        )

        assert out_file.exists()
        mock_extract.assert_called_once()
        mock_to_dicts.assert_called_once()


# ── 2. cmd_confidence_layers ─────────────────────────────────────────────
class TestCmdConfidenceLayers:
    """Tests for cmd_confidence_layers."""

    @patch("research_pipeline.cli.cmd_confidence_layers.create_llm_provider")
    @patch("research_pipeline.cli.cmd_confidence_layers.read_jsonl")
    @patch("research_pipeline.cli.cmd_confidence_layers.init_run")
    @patch("research_pipeline.cli.cmd_confidence_layers.load_config")
    def test_missing_claims_exits(
        self, mock_cfg, mock_init, mock_read, mock_llm, tmp_path
    ):
        from research_pipeline.cli.cmd_confidence_layers import run_confidence_layers

        mock_cfg.return_value = _make_config(tmp_path)
        run_dir = tmp_path / "run1"
        mock_init.return_value = ("run1", run_dir)
        # Create the summarize/claims dir but no file
        claims_dir = run_dir / "summarize" / "claims"
        claims_dir.mkdir(parents=True)

        with pytest.raises(click.exceptions.Exit):
            run_confidence_layers(
                config_path=None,
                workspace=tmp_path,
                run_id="run1",
                l4_threshold=0.5,
                damping=0.8,
                calibrate=False,
            )

    @patch("research_pipeline.cli.cmd_confidence_layers.write_jsonl")
    @patch("research_pipeline.cli.cmd_confidence_layers.batch_calibration_report")
    @patch("research_pipeline.cli.cmd_confidence_layers.score_batch_layered")
    @patch("research_pipeline.cli.cmd_confidence_layers.create_llm_provider")
    @patch("research_pipeline.cli.cmd_confidence_layers.read_jsonl")
    @patch("research_pipeline.cli.cmd_confidence_layers.init_run")
    @patch("research_pipeline.cli.cmd_confidence_layers.load_config")
    def test_happy_path(
        self,
        mock_cfg,
        mock_init,
        mock_read,
        mock_llm,
        mock_score,
        mock_report,
        mock_write,
        tmp_path,
    ):
        from research_pipeline.cli.cmd_confidence_layers import run_confidence_layers

        mock_cfg.return_value = _make_config(tmp_path)
        run_dir = tmp_path / "run1"
        mock_init.return_value = ("run1", run_dir)

        claims_dir = run_dir / "summarize" / "claims"
        claims_dir.mkdir(parents=True)
        claims_path = claims_dir / "claim_decomposition.jsonl"
        claims_path.write_text("", encoding="utf-8")

        claim_mock = MagicMock()
        claim_mock.confidence_score = 0.8
        decomp = MagicMock()
        decomp.claims = [claim_mock]
        mock_read.return_value = [{"paper_id": "p1", "claims": []}]

        with patch(
            "research_pipeline.cli.cmd_confidence_layers.ClaimDecomposition"
        ) as MockCD:
            MockCD.model_validate.return_value = decomp
            mock_llm.return_value = None

            result_item = MagicMock()
            result_item.final_score = 0.75
            result_item.l4 = MagicMock(triggered=False)
            result_item.l2 = MagicMock(decision=MagicMock(value="skip"))
            result_item.to_dict.return_value = {"score": 0.75}
            mock_score.return_value = [result_item]

            report_obj = MagicMock()
            report_obj.ece = 0.05
            report_obj.brier = 0.1
            report_obj.auroc = 0.9
            report_obj.to_dict.return_value = {"ece": 0.05}
            mock_report.return_value = report_obj

            run_confidence_layers(
                config_path=None,
                workspace=tmp_path,
                run_id="run1",
                l4_threshold=0.5,
                damping=0.8,
                calibrate=False,
            )

            mock_score.assert_called_once()
            mock_report.assert_called_once()

    @patch("research_pipeline.cli.cmd_confidence_layers.write_jsonl")
    @patch("research_pipeline.cli.cmd_confidence_layers.batch_calibration_report")
    @patch("research_pipeline.cli.cmd_confidence_layers.score_batch_layered")
    @patch("research_pipeline.cli.cmd_confidence_layers.create_llm_provider")
    @patch("research_pipeline.cli.cmd_confidence_layers.read_jsonl")
    @patch("research_pipeline.cli.cmd_confidence_layers.init_run")
    @patch("research_pipeline.cli.cmd_confidence_layers.load_config")
    def test_no_claims_exits_zero(
        self,
        mock_cfg,
        mock_init,
        mock_read,
        mock_llm,
        mock_score,
        mock_report,
        mock_write,
        tmp_path,
    ):
        from research_pipeline.cli.cmd_confidence_layers import run_confidence_layers

        mock_cfg.return_value = _make_config(tmp_path)
        run_dir = tmp_path / "run1"
        mock_init.return_value = ("run1", run_dir)

        claims_dir = run_dir / "summarize" / "claims"
        claims_dir.mkdir(parents=True)
        (claims_dir / "claim_decomposition.jsonl").write_text("", encoding="utf-8")

        decomp = MagicMock()
        decomp.claims = []
        mock_read.return_value = [{}]

        with patch(
            "research_pipeline.cli.cmd_confidence_layers.ClaimDecomposition"
        ) as MockCD:
            MockCD.model_validate.return_value = decomp

            with pytest.raises(click.exceptions.Exit) as exc_info:
                run_confidence_layers(
                    config_path=None,
                    workspace=tmp_path,
                    run_id="run1",
                    l4_threshold=0.5,
                    damping=0.8,
                    calibrate=False,
                )
            # Exit code 0 for "no claims to score"
            assert exc_info.value.exit_code == 0


# ── 3. cmd_convert_fine ──────────────────────────────────────────────────
class TestCmdConvertFine:
    """Tests for cmd_convert_fine."""

    def test_empty_paper_ids_exits(self):
        from research_pipeline.cli.cmd_convert_fine import run_convert_fine

        with pytest.raises(click.exceptions.Exit):
            run_convert_fine(
                paper_ids=[],
                force=False,
                config_path=None,
                workspace=None,
                run_id=None,
                backend=None,
            )

    @patch("research_pipeline.cli.cmd_convert_fine.read_jsonl")
    @patch("research_pipeline.cli.cmd_convert_fine.init_run")
    @patch("research_pipeline.cli.cmd_convert_fine.load_config")
    def test_missing_download_manifest_exits(
        self, mock_cfg, mock_init, mock_read, tmp_path
    ):
        from research_pipeline.cli.cmd_convert_fine import run_convert_fine

        mock_cfg.return_value = _make_config(tmp_path)
        run_dir = tmp_path / "run1"
        mock_init.return_value = ("run1", run_dir)
        # "download_root" maps to "download" in STAGE_SUBDIRS — no manifest there
        (run_dir / "download").mkdir(parents=True)

        with pytest.raises(click.exceptions.Exit):
            run_convert_fine(
                paper_ids=["2301.00001"],
                force=False,
                config_path=None,
                workspace=tmp_path,
                run_id="run1",
                backend=None,
            )

    @patch("research_pipeline.cli.cmd_convert_fine.write_jsonl")
    @patch("research_pipeline.cli.cmd_convert_fine._create_converter")
    @patch("research_pipeline.cli.cmd_convert_fine.read_jsonl")
    @patch("research_pipeline.cli.cmd_convert_fine.init_run")
    @patch("research_pipeline.cli.cmd_convert_fine.load_config")
    def test_no_matching_papers_exits(
        self, mock_cfg, mock_init, mock_read, mock_conv, mock_write, tmp_path
    ):
        from research_pipeline.cli.cmd_convert_fine import run_convert_fine

        mock_cfg.return_value = _make_config(tmp_path)
        run_dir = tmp_path / "run1"
        mock_init.return_value = ("run1", run_dir)
        # "download_root" → "download" in STAGE_SUBDIRS
        dl_dir = run_dir / "download"
        dl_dir.mkdir(parents=True)
        (dl_dir / "download_manifest.jsonl").write_text("", encoding="utf-8")

        # Return entries that don't match requested IDs
        mock_read.return_value = [_dl_entry_dict(arxiv_id="9999.99999")]

        with pytest.raises(click.exceptions.Exit):
            run_convert_fine(
                paper_ids=["2301.00001"],
                force=False,
                config_path=None,
                workspace=tmp_path,
                run_id="run1",
                backend=None,
            )

    @patch("research_pipeline.cli.cmd_convert_fine.write_jsonl")
    @patch("research_pipeline.cli.cmd_convert_fine._create_converter")
    @patch("research_pipeline.cli.cmd_convert_fine.read_jsonl")
    @patch("research_pipeline.cli.cmd_convert_fine.init_run")
    @patch("research_pipeline.cli.cmd_convert_fine.load_config")
    def test_happy_path(
        self, mock_cfg, mock_init, mock_read, mock_create_conv, mock_write, tmp_path
    ):
        from research_pipeline.cli.cmd_convert_fine import run_convert_fine

        mock_cfg.return_value = _make_config(tmp_path)
        run_dir = tmp_path / "run1"
        mock_init.return_value = ("run1", run_dir)

        # "download_root" → "download" in STAGE_SUBDIRS
        dl_dir = run_dir / "download"
        dl_dir.mkdir(parents=True)
        (dl_dir / "download_manifest.jsonl").write_text("", encoding="utf-8")

        pdf_path = tmp_path / "paper.pdf"
        pdf_path.write_text("fake pdf", encoding="utf-8")
        mock_read.return_value = [
            _dl_entry_dict(local_path=str(pdf_path)),
        ]

        converter = MagicMock()
        converter.name = "test_backend"
        result = MagicMock()
        result.status = "converted"
        result.model_dump.return_value = {
            "status": "converted",
            "arxiv_id": "2301.00001",
        }
        converter.convert.return_value = result
        mock_create_conv.return_value = converter

        run_convert_fine(
            paper_ids=["2301.00001"],
            force=False,
            config_path=None,
            workspace=tmp_path,
            run_id="run1",
            backend="test_backend",
        )

        converter.convert.assert_called_once()
        mock_write.assert_called_once()


# ── 4. cmd_convert_rough ─────────────────────────────────────────────────
class TestCmdConvertRough:
    """Tests for cmd_convert_rough."""

    @patch("research_pipeline.cli.cmd_convert_rough.read_jsonl")
    @patch("research_pipeline.cli.cmd_convert_rough.init_run")
    @patch("research_pipeline.cli.cmd_convert_rough.load_config")
    def test_missing_download_manifest_exits(
        self, mock_cfg, mock_init, mock_read, tmp_path
    ):
        from research_pipeline.cli.cmd_convert_rough import run_convert_rough

        mock_cfg.return_value = _make_config(tmp_path)
        run_dir = tmp_path / "run1"
        mock_init.return_value = ("run1", run_dir)
        (run_dir / "download").mkdir(parents=True)

        with pytest.raises(click.exceptions.Exit):
            run_convert_rough(
                force=False,
                config_path=None,
                workspace=tmp_path,
                run_id="run1",
            )

    @patch("research_pipeline.cli.cmd_convert_rough.write_jsonl")
    @patch("research_pipeline.cli.cmd_convert_rough.get_backend")
    @patch("research_pipeline.cli.cmd_convert_rough._ensure_builtins_registered")
    @patch("research_pipeline.cli.cmd_convert_rough.read_jsonl")
    @patch("research_pipeline.cli.cmd_convert_rough.init_run")
    @patch("research_pipeline.cli.cmd_convert_rough.load_config")
    def test_happy_path(
        self,
        mock_cfg,
        mock_init,
        mock_read,
        mock_ensure,
        mock_get_backend,
        mock_write,
        tmp_path,
    ):
        from research_pipeline.cli.cmd_convert_rough import run_convert_rough

        mock_cfg.return_value = _make_config(tmp_path)
        run_dir = tmp_path / "run1"
        mock_init.return_value = ("run1", run_dir)

        # "download_root" → "download" in STAGE_SUBDIRS
        dl_dir = run_dir / "download"
        dl_dir.mkdir(parents=True)
        (dl_dir / "download_manifest.jsonl").write_text("", encoding="utf-8")

        pdf_path = tmp_path / "paper.pdf"
        pdf_path.write_text("fake pdf", encoding="utf-8")
        mock_read.return_value = [
            _dl_entry_dict(local_path=str(pdf_path)),
        ]

        converter = MagicMock()
        result = MagicMock()
        result.status = "converted"
        result.model_dump.return_value = {"status": "converted"}
        converter.convert.return_value = result
        mock_get_backend.return_value = converter

        run_convert_rough(
            force=False,
            config_path=None,
            workspace=tmp_path,
            run_id="run1",
        )

        converter.convert.assert_called_once()
        mock_write.assert_called_once()


# ── 5. cmd_download ──────────────────────────────────────────────────────
class TestCmdDownload:
    """Tests for cmd_download."""

    @patch("research_pipeline.cli.cmd_download.init_run")
    @patch("research_pipeline.cli.cmd_download.load_config")
    def test_missing_shortlist_exits(self, mock_cfg, mock_init, tmp_path):
        from research_pipeline.cli.cmd_download import run_download

        mock_cfg.return_value = _make_config(tmp_path)
        run_dir = tmp_path / "run1"
        mock_init.return_value = ("run1", run_dir)
        (run_dir / "screen").mkdir(parents=True)

        with pytest.raises(click.exceptions.Exit):
            run_download(
                force=False,
                config_path=None,
                workspace=tmp_path,
                run_id="run1",
            )

    @patch("research_pipeline.cli.cmd_download.write_jsonl")
    @patch("research_pipeline.cli.cmd_download.download_batch")
    @patch("research_pipeline.cli.cmd_download.create_session")
    @patch("research_pipeline.cli.cmd_download.ArxivRateLimiter")
    @patch("research_pipeline.cli.cmd_download.parse_shortlist_lenient")
    @patch("research_pipeline.cli.cmd_download.init_run")
    @patch("research_pipeline.cli.cmd_download.load_config")
    def test_happy_path(
        self,
        mock_cfg,
        mock_init,
        mock_parse,
        mock_limiter,
        mock_session,
        mock_batch,
        mock_write,
        tmp_path,
    ):
        from research_pipeline.cli.cmd_download import run_download

        mock_cfg.return_value = _make_config(tmp_path)
        run_dir = tmp_path / "run1"
        mock_init.return_value = ("run1", run_dir)

        screen_dir = run_dir / "screen"
        screen_dir.mkdir(parents=True)
        shortlist_data = [
            {
                "paper": {
                    "arxiv_id": "2301.00001",
                    "version": "v1",
                    "pdf_url": "http://x",
                },
                "download": True,
            }
        ]
        (screen_dir / "shortlist.json").write_text(
            json.dumps(shortlist_data), encoding="utf-8"
        )

        paper_mock = MagicMock()
        paper_mock.arxiv_id = "2301.00001"
        paper_mock.version = "v1"
        paper_mock.pdf_url = "http://x"
        decision_mock = MagicMock()
        decision_mock.paper = paper_mock
        decision_mock.download = True
        mock_parse.return_value = decision_mock

        dl_entry = MagicMock()
        dl_entry.status = "downloaded"
        dl_entry.model_dump.return_value = {"status": "downloaded"}
        mock_batch.return_value = [dl_entry]

        # get_stage_dir creates dirs; "download" → "download/pdf"
        # No need to pre-create — get_stage_dir does it

        run_download(
            force=False,
            config_path=None,
            workspace=tmp_path,
            run_id="run1",
        )

        mock_batch.assert_called_once()
        mock_write.assert_called_once()


# ── 6. cmd_enrich ────────────────────────────────────────────────────────
class TestCmdEnrich:
    """Tests for cmd_enrich."""

    @patch("research_pipeline.cli.cmd_enrich.load_config")
    def test_missing_run_dir_exits(self, mock_cfg, tmp_path):
        from research_pipeline.cli.cmd_enrich import enrich_command

        mock_cfg.return_value = _make_config(tmp_path)

        with pytest.raises(click.exceptions.Exit):
            enrich_command(
                run_id="nonexistent",
                stage="candidates",
                config_path=Path("config.toml"),
            )

    @patch("research_pipeline.cli.cmd_enrich.load_config")
    def test_missing_candidates_file_exits(self, mock_cfg, tmp_path):
        from research_pipeline.cli.cmd_enrich import enrich_command

        mock_cfg.return_value = _make_config(tmp_path)
        run_dir = tmp_path / "test-run"
        run_dir.mkdir()
        (run_dir / "search").mkdir(parents=True)

        with pytest.raises(click.exceptions.Exit):
            enrich_command(
                run_id="test-run",
                stage="candidates",
                config_path=Path("config.toml"),
            )

    @patch("research_pipeline.cli.cmd_enrich.write_jsonl")
    @patch("research_pipeline.cli.cmd_enrich.enrich_candidates")
    @patch("research_pipeline.cli.cmd_enrich.read_jsonl")
    @patch("research_pipeline.cli.cmd_enrich.load_config")
    def test_happy_path(self, mock_cfg, mock_read, mock_enrich, mock_write, tmp_path):
        from research_pipeline.cli.cmd_enrich import enrich_command

        mock_cfg.return_value = _make_config(tmp_path)
        run_dir = tmp_path / "test-run"
        search_dir = run_dir / "search"
        search_dir.mkdir(parents=True)
        (search_dir / "candidates.jsonl").write_text(
            json.dumps(_candidate_dict()) + "\n",
            encoding="utf-8",
        )
        mock_read.return_value = [_candidate_dict()]
        mock_enrich.return_value = 1

        enrich_command(
            run_id="test-run",
            stage="candidates",
            config_path=Path("config.toml"),
        )

        mock_enrich.assert_called_once()
        mock_write.assert_called_once()

    @patch("research_pipeline.cli.cmd_enrich.write_jsonl")
    @patch("research_pipeline.cli.cmd_enrich.enrich_candidates")
    @patch("research_pipeline.cli.cmd_enrich.read_jsonl")
    @patch("research_pipeline.cli.cmd_enrich.load_config")
    def test_screened_stage(
        self, mock_cfg, mock_read, mock_enrich, mock_write, tmp_path
    ):
        from research_pipeline.cli.cmd_enrich import enrich_command

        mock_cfg.return_value = _make_config(tmp_path)
        run_dir = tmp_path / "test-run"
        screen_dir = run_dir / "screen"
        screen_dir.mkdir(parents=True)
        (screen_dir / "screened.jsonl").write_text(
            json.dumps(_candidate_dict()) + "\n",
            encoding="utf-8",
        )
        mock_read.return_value = [_candidate_dict()]
        mock_enrich.return_value = 0

        enrich_command(
            run_id="test-run",
            stage="screened",
            config_path=Path("config.toml"),
        )

        mock_enrich.assert_called_once()


# ── 7. cmd_eval_log ──────────────────────────────────────────────────────
class TestCmdEvalLog:
    """Tests for cmd_eval_log."""

    @patch("research_pipeline.cli.cmd_eval_log.setup_logging")
    @patch("research_pipeline.cli.cmd_eval_log.load_config")
    def test_missing_run_dir_exits(self, mock_cfg, mock_setup, tmp_path):
        from research_pipeline.cli.cmd_eval_log import eval_log_cmd

        mock_cfg.return_value = _make_config(tmp_path)

        with pytest.raises(click.exceptions.Exit):
            eval_log_cmd(
                run_id="nonexistent",
                channel="all",
                stage="",
                limit=50,
                workspace=tmp_path,
            )

    @patch("research_pipeline.cli.cmd_eval_log.EvalLogger")
    @patch("research_pipeline.cli.cmd_eval_log.setup_logging")
    @patch("research_pipeline.cli.cmd_eval_log.load_config")
    def test_happy_path_all_channels(self, mock_cfg, mock_setup, mock_eval, tmp_path):
        from research_pipeline.cli.cmd_eval_log import eval_log_cmd

        mock_cfg.return_value = _make_config(tmp_path)
        run_dir = tmp_path / "test-run"
        run_dir.mkdir()

        eval_instance = MagicMock()
        eval_instance.tracer.read_traces.return_value = [
            {
                "timestamp": "2024-01-01T00:00:00Z",
                "event": "start",
                "stage": "plan",
                "level": "info",
            }
        ]
        eval_instance.audit.count.return_value = 0
        eval_instance.snapshots.list_snapshots.return_value = []
        mock_eval.return_value = eval_instance

        eval_log_cmd(
            run_id="test-run",
            channel="all",
            stage="",
            limit=50,
            workspace=tmp_path,
        )

        eval_instance.tracer.read_traces.assert_called_once()
        eval_instance.close.assert_called_once()

    @patch("research_pipeline.cli.cmd_eval_log.EvalLogger")
    @patch("research_pipeline.cli.cmd_eval_log.setup_logging")
    @patch("research_pipeline.cli.cmd_eval_log.load_config")
    def test_summary_channel(self, mock_cfg, mock_setup, mock_eval, tmp_path):
        from research_pipeline.cli.cmd_eval_log import eval_log_cmd

        mock_cfg.return_value = _make_config(tmp_path)
        run_dir = tmp_path / "test-run"
        run_dir.mkdir()

        eval_instance = MagicMock()
        eval_instance.summary.return_value = {"traces": 0, "audit": 0}
        mock_eval.return_value = eval_instance

        eval_log_cmd(
            run_id="test-run",
            channel="summary",
            stage="",
            limit=50,
            workspace=tmp_path,
        )

        eval_instance.summary.assert_called_once()

    @patch("research_pipeline.cli.cmd_eval_log.EvalLogger")
    @patch("research_pipeline.cli.cmd_eval_log.setup_logging")
    @patch("research_pipeline.cli.cmd_eval_log.load_config")
    def test_audit_channel_with_records(
        self, mock_cfg, mock_setup, mock_eval, tmp_path
    ):
        from research_pipeline.cli.cmd_eval_log import eval_log_cmd

        mock_cfg.return_value = _make_config(tmp_path)
        run_dir = tmp_path / "test-run"
        run_dir.mkdir()

        eval_instance = MagicMock()
        eval_instance.audit.count.return_value = 2
        eval_instance.audit.query.return_value = [
            {
                "timestamp": "2024-01-01T00:00:00Z",
                "stage": "plan",
                "action": "generate",
                "model": "gpt-4",
                "tokens_used": 100,
                "duration_ms": 500,
            }
        ]
        eval_instance.snapshots.list_snapshots.return_value = ["snap1"]
        eval_instance.snapshots.get_manifest.return_value = {
            "file_count": 3,
            "total_size": 2048,
        }
        eval_instance.tracer.read_traces.return_value = []
        mock_eval.return_value = eval_instance

        eval_log_cmd(
            run_id="test-run",
            channel="all",
            stage="",
            limit=50,
            workspace=tmp_path,
        )

        eval_instance.audit.query.assert_called_once()


# ── 8. cmd_expand ────────────────────────────────────────────────────────
class TestCmdExpand:
    """Tests for cmd_expand."""

    @patch("research_pipeline.cli.cmd_expand.load_config")
    def test_empty_paper_ids_returns(self, mock_cfg, tmp_path):
        from research_pipeline.cli.cmd_expand import run_expand

        mock_cfg.return_value = _make_config(tmp_path)
        # Should return early without error
        run_expand(
            paper_ids=[],
            direction="both",
            config_path=None,
            workspace=tmp_path,
            run_id="run1",
        )

    @patch("research_pipeline.cli.cmd_expand.load_config")
    def test_no_run_id_returns(self, mock_cfg, tmp_path):
        from research_pipeline.cli.cmd_expand import run_expand

        mock_cfg.return_value = _make_config(tmp_path)
        run_expand(
            paper_ids=["2301.00001"],
            direction="both",
            config_path=None,
            workspace=tmp_path,
            run_id=None,
        )

    @patch("research_pipeline.cli.cmd_expand.write_jsonl")
    @patch("research_pipeline.cli.cmd_expand.CitationGraphClient")
    @patch("research_pipeline.cli.cmd_expand.RateLimiter")
    @patch("research_pipeline.cli.cmd_expand.init_run")
    @patch("research_pipeline.cli.cmd_expand.load_config")
    def test_standard_expand_happy_path(
        self, mock_cfg, mock_init, mock_limiter, mock_client_cls, mock_write, tmp_path
    ):
        from research_pipeline.cli.cmd_expand import run_expand

        mock_cfg.return_value = _make_config(tmp_path)
        run_dir = tmp_path / "run1"
        mock_init.return_value = ("run1", run_dir)

        candidate = MagicMock()
        candidate.arxiv_id = "2301.00002"
        candidate.model_dump.return_value = {"arxiv_id": "2301.00002"}
        client = MagicMock()
        client.fetch_related.return_value = [candidate]
        mock_client_cls.return_value = client

        run_expand(
            paper_ids=["2301.00001"],
            direction="both",
            limit_per_paper=50,
            config_path=None,
            workspace=tmp_path,
            run_id="run1",
        )

        client.fetch_related.assert_called_once()
        mock_write.assert_called_once()

    @patch("research_pipeline.cli.cmd_expand.write_jsonl")
    @patch("research_pipeline.cli.cmd_expand.CitationGraphClient")
    @patch("research_pipeline.cli.cmd_expand.RateLimiter")
    @patch("research_pipeline.cli.cmd_expand.init_run")
    @patch("research_pipeline.cli.cmd_expand.load_config")
    def test_snowball_mode(
        self, mock_cfg, mock_init, mock_limiter, mock_client_cls, mock_write, tmp_path
    ):
        from research_pipeline.cli.cmd_expand import run_expand

        mock_cfg.return_value = _make_config(tmp_path)
        run_dir = tmp_path / "run1"
        mock_init.return_value = ("run1", run_dir)
        (run_dir / "expand").mkdir(parents=True)

        client = MagicMock()
        mock_client_cls.return_value = client

        candidate = MagicMock()
        candidate.model_dump.return_value = {"arxiv_id": "2301.00003"}
        result_mock = MagicMock()
        result_mock.model_dump.return_value = {"rounds": 1}

        with (
            patch("research_pipeline.sources.snowball.snowball_expand") as mock_snow,
            patch(
                "research_pipeline.sources.snowball.format_snowball_report"
            ) as mock_fmt,
        ):
            mock_snow.return_value = ([candidate], result_mock)
            mock_fmt.return_value = "# Snowball Report"

            run_expand(
                paper_ids=["2301.00001"],
                direction="both",
                snowball=True,
                snowball_max_rounds=2,
                snowball_max_papers=50,
                config_path=None,
                workspace=tmp_path,
                run_id="run1",
            )

        mock_write.assert_called_once()


# ── 9. cmd_extract ───────────────────────────────────────────────────────
class TestCmdExtract:
    """Tests for cmd_extract."""

    @patch("research_pipeline.cli.cmd_extract.read_jsonl")
    @patch("research_pipeline.cli.cmd_extract.init_run")
    @patch("research_pipeline.cli.cmd_extract.load_config")
    def test_missing_manifest_exits(self, mock_cfg, mock_init, mock_read, tmp_path):
        from research_pipeline.cli.cmd_extract import run_extract

        mock_cfg.return_value = _make_config(tmp_path)
        run_dir = tmp_path / "run1"
        mock_init.return_value = ("run1", run_dir)
        # Create all needed dirs without manifests
        # "convert_root" → "convert", "convert_rough" and "convert_fine" are same name
        (run_dir / "convert").mkdir(parents=True)
        (run_dir / "convert_rough").mkdir(parents=True)
        (run_dir / "convert_fine").mkdir(parents=True)

        with pytest.raises(click.exceptions.Exit):
            run_extract(
                config_path=None,
                workspace=tmp_path,
                run_id="run1",
                cross_encoder=None,
            )

    @patch("research_pipeline.cli.cmd_extract.extract_bibliography")
    @patch("research_pipeline.cli.cmd_extract.extract_from_markdown")
    @patch("research_pipeline.cli.cmd_extract.read_jsonl")
    @patch("research_pipeline.cli.cmd_extract.init_run")
    @patch("research_pipeline.cli.cmd_extract.load_config")
    def test_happy_path(
        self, mock_cfg, mock_init, mock_read, mock_extract, mock_bib, tmp_path
    ):
        from research_pipeline.cli.cmd_extract import run_extract

        mock_cfg.return_value = _make_config(tmp_path)
        run_dir = tmp_path / "run1"
        mock_init.return_value = ("run1", run_dir)

        # "convert_root" → "convert" in STAGE_SUBDIRS
        convert_root = run_dir / "convert"
        convert_root.mkdir(parents=True)

        md_path = tmp_path / "paper.md"
        md_path.write_text("# Paper\nSome content", encoding="utf-8")

        manifest_path = convert_root / "convert_manifest.jsonl"
        conv_data = _conv_entry_dict(markdown_path=str(md_path))
        manifest_path.write_text(
            json.dumps(conv_data),
            encoding="utf-8",
        )

        mock_read.return_value = [conv_data]

        extraction = MagicMock()
        extraction.model_dump_json.return_value = '{"chunks": []}'
        mock_extract.return_value = extraction

        bib_entry = MagicMock()
        bib_entry.raw_text = "ref"
        bib_entry.title = "T"
        bib_entry.authors = []
        bib_entry.year = "2023"
        bib_entry.arxiv_id = ""
        bib_entry.doi = ""
        mock_bib.return_value = [bib_entry]

        (run_dir / "extract").mkdir(parents=True)

        run_extract(
            config_path=None,
            workspace=tmp_path,
            run_id="run1",
            cross_encoder=None,
        )

        mock_extract.assert_called_once()
        mock_bib.assert_called_once()

    def test_discover_two_tier_merge(self, tmp_path):
        """Test _discover_convert_manifest with two-tier merge."""
        from research_pipeline.cli.cmd_extract import _discover_convert_manifest

        run_root = tmp_path / "run1"
        # "convert_root" → "convert" in STAGE_SUBDIRS
        (run_root / "convert").mkdir(parents=True)
        rough_dir = run_root / "convert_rough"
        rough_dir.mkdir(parents=True)

        rough_data = _conv_entry_dict()

        with (
            patch("research_pipeline.cli.cmd_extract.read_jsonl") as mock_read,
            patch("research_pipeline.cli.cmd_extract.write_jsonl"),
        ):
            mock_read.return_value = [rough_data]
            manifest = rough_dir / "convert_rough_manifest.jsonl"
            manifest.write_text("", encoding="utf-8")

            entries = _discover_convert_manifest(run_root)
            assert len(entries) == 1


# ── 10. cmd_feedback ─────────────────────────────────────────────────────
class TestCmdFeedback:
    """Tests for cmd_feedback."""

    @patch("research_pipeline.cli.cmd_feedback.FeedbackStore")
    @patch("research_pipeline.cli.cmd_feedback.setup_logging")
    @patch("research_pipeline.cli.cmd_feedback.load_config")
    def test_accept_and_reject(self, mock_cfg, mock_setup, mock_store_cls, tmp_path):
        from research_pipeline.cli.cmd_feedback import feedback_cmd

        mock_cfg.return_value = _make_config(tmp_path)
        run_dir = tmp_path / "test-run"
        screen_dir = run_dir / "screen"
        screen_dir.mkdir(parents=True)

        store = MagicMock()
        mock_store_cls.return_value = store

        feedback_cmd(
            run_id="test-run",
            accept=["paper1"],
            reject=["paper2"],
            reason="test reason",
            show=False,
            adjust=False,
            workspace=tmp_path,
        )

        assert store.record.call_count == 2
        store.close.assert_called_once()

    @patch("research_pipeline.cli.cmd_feedback.FeedbackStore")
    @patch("research_pipeline.cli.cmd_feedback.setup_logging")
    @patch("research_pipeline.cli.cmd_feedback.load_config")
    def test_show_and_adjust(self, mock_cfg, mock_setup, mock_store_cls, tmp_path):
        from research_pipeline.cli.cmd_feedback import feedback_cmd

        mock_cfg.return_value = _make_config(tmp_path)
        run_dir = tmp_path / "test-run"
        screen_dir = run_dir / "screen"
        screen_dir.mkdir(parents=True)

        store = MagicMock()
        store.count.return_value = {"accept": 5, "reject": 3, "total": 8}
        weights_mock = MagicMock()
        weights_mock.feedback_count = 8
        weights_mock.to_weight_dict.return_value = {"w1": 0.5}
        store.get_latest_weights.return_value = weights_mock
        store.compute_adjusted_weights.return_value = weights_mock
        mock_store_cls.return_value = store

        feedback_cmd(
            run_id="test-run",
            accept=None,
            reject=None,
            reason="",
            show=True,
            adjust=True,
            workspace=tmp_path,
        )

        assert store.count.call_count >= 1
        store.compute_adjusted_weights.assert_called_once()

    @patch("research_pipeline.cli.cmd_feedback.FeedbackStore")
    @patch("research_pipeline.cli.cmd_feedback.setup_logging")
    @patch("research_pipeline.cli.cmd_feedback.load_config")
    def test_shortlist_score_loading(
        self, mock_cfg, mock_setup, mock_store_cls, tmp_path
    ):
        from research_pipeline.cli.cmd_feedback import feedback_cmd

        mock_cfg.return_value = _make_config(tmp_path)
        run_dir = tmp_path / "test-run"
        screen_dir = run_dir / "screen"
        screen_dir.mkdir(parents=True)

        shortlist_data = [
            {"paper": {"arxiv_id": "p1"}, "final_score": 0.9},
        ]
        (screen_dir / "shortlist.json").write_text(
            json.dumps(shortlist_data), encoding="utf-8"
        )

        store = MagicMock()
        mock_store_cls.return_value = store

        feedback_cmd(
            run_id="test-run",
            accept=["p1"],
            reject=None,
            reason="",
            show=False,
            adjust=False,
            workspace=tmp_path,
        )

        store.record.assert_called_once()


# ── 11. cmd_kg_quality ───────────────────────────────────────────────────
class TestCmdKgQuality:
    """Tests for cmd_kg_quality (lazy imports)."""

    def test_missing_db_exits(self, tmp_path):
        from research_pipeline.cli.cmd_kg_quality import kg_quality_command

        with (
            patch(
                "research_pipeline.storage.knowledge_graph.DEFAULT_KG_PATH",
                tmp_path / "no_such.db",
            ),
            pytest.raises(click.exceptions.Exit),
        ):
            kg_quality_command(
                db_path=str(tmp_path / "no_such.db"),
                staleness_days=365.0,
                sample_size=0,
                output_json=False,
            )

    def test_happy_path_text_output(self, tmp_path):
        from research_pipeline.cli.cmd_kg_quality import kg_quality_command

        db_path = tmp_path / "kg.db"
        db_path.write_text("", encoding="utf-8")

        score_mock = MagicMock()
        score_mock.composite = 0.85
        score_mock.accuracy = 0.9
        score_mock.consistency = 0.8
        score_mock.completeness = 0.75
        score_mock.timeliness = 0.95
        score_mock.redundancy = 0.7
        score_mock.structural = MagicMock(
            num_entities=100,
            num_triples=500,
            icr=5.0,
            density=0.1,
            connected_components=2,
        )
        score_mock.consistency_detail = MagicMock(
            ic_score=0.9,
            ec_score=0.8,
            ic_contradiction_count=1,
            duplicate_triples=0,
        )
        score_mock.completeness_detail = MagicMock(
            entity_type_coverage=0.9,
            relation_type_coverage=0.8,
            orphan_entities=5,
        )

        with (
            patch(
                "research_pipeline.quality.kg_quality.evaluate_kg_quality",
                return_value=score_mock,
            ),
            patch(
                "research_pipeline.quality.kg_quality.sample_triples_twcs",
                return_value=[],
            ),
            patch(
                "research_pipeline.storage.knowledge_graph.DEFAULT_KG_PATH",
                db_path,
            ),
            patch("sqlite3.connect") as mock_conn,
        ):
            conn = MagicMock()
            mock_conn.return_value = conn

            kg_quality_command(
                db_path=str(db_path),
                staleness_days=365.0,
                sample_size=0,
                output_json=False,
            )

    def test_json_output_with_sampling(self, tmp_path):
        from research_pipeline.cli.cmd_kg_quality import kg_quality_command

        db_path = tmp_path / "kg.db"
        db_path.write_text("", encoding="utf-8")

        score_mock = MagicMock()
        score_mock.to_dict.return_value = {"composite": 0.85}

        sample = [{"subject_id": "s1", "relation": "cites", "object_id": "o1"}]

        with (
            patch(
                "research_pipeline.quality.kg_quality.evaluate_kg_quality",
                return_value=score_mock,
            ),
            patch(
                "research_pipeline.quality.kg_quality.sample_triples_twcs",
                return_value=sample,
            ),
            patch(
                "research_pipeline.storage.knowledge_graph.DEFAULT_KG_PATH",
                db_path,
            ),
            patch("sqlite3.connect") as mock_conn,
        ):
            conn = MagicMock()
            mock_conn.return_value = conn

            kg_quality_command(
                db_path=str(db_path),
                staleness_days=365.0,
                sample_size=5,
                output_json=True,
            )


# ── 12. cmd_quality ──────────────────────────────────────────────────────
class TestCmdQuality:
    """Tests for cmd_quality."""

    @patch("research_pipeline.cli.cmd_quality.load_config")
    def test_no_run_id_returns(self, mock_cfg, tmp_path):
        from research_pipeline.cli.cmd_quality import run_quality

        mock_cfg.return_value = _make_config(tmp_path)
        # Should return early without error
        run_quality(config_path=None, workspace=tmp_path, run_id=None)

    @patch("research_pipeline.cli.cmd_quality.write_jsonl")
    @patch("research_pipeline.cli.cmd_quality.compute_quality_score")
    @patch("research_pipeline.cli.cmd_quality.read_jsonl")
    @patch("research_pipeline.cli.cmd_quality.init_run")
    @patch("research_pipeline.cli.cmd_quality.load_config")
    def test_no_candidates_returns(
        self, mock_cfg, mock_init, mock_read, mock_score, mock_write, tmp_path
    ):
        from research_pipeline.cli.cmd_quality import run_quality

        mock_cfg.return_value = _make_config(tmp_path)
        run_dir = tmp_path / "run1"
        mock_init.return_value = ("run1", run_dir)
        (run_dir / "screen").mkdir(parents=True)
        (run_dir / "search").mkdir(parents=True)

        # No shortlist.json and no candidates.jsonl
        run_quality(config_path=None, workspace=tmp_path, run_id="run1")
        mock_score.assert_not_called()

    @patch("research_pipeline.cli.cmd_quality.write_jsonl")
    @patch("research_pipeline.cli.cmd_quality.compute_quality_score")
    @patch("research_pipeline.cli.cmd_quality.init_run")
    @patch("research_pipeline.cli.cmd_quality.load_config")
    def test_happy_path_from_candidates_jsonl(
        self, mock_cfg, mock_init, mock_score, mock_write, tmp_path
    ):
        from research_pipeline.cli.cmd_quality import run_quality

        mock_cfg.return_value = _make_config(tmp_path)
        run_dir = tmp_path / "run1"
        mock_init.return_value = ("run1", run_dir)

        screen_dir = run_dir / "screen"
        screen_dir.mkdir(parents=True)
        search_dir = run_dir / "search"
        search_dir.mkdir(parents=True)

        # Create candidates.jsonl
        candidate_data = _candidate_dict()
        (search_dir / "candidates.jsonl").write_text(
            json.dumps(candidate_data), encoding="utf-8"
        )

        with patch(
            "research_pipeline.cli.cmd_quality.read_jsonl",
            return_value=[candidate_data],
        ):
            qs = MagicMock()
            qs.model_dump.return_value = {"score": 0.8}
            mock_score.return_value = qs

            (run_dir / "quality").mkdir(parents=True)
            (run_dir / "summarize").mkdir(parents=True)

            run_quality(config_path=None, workspace=tmp_path, run_id="run1")

            mock_score.assert_called_once()

    @patch("research_pipeline.cli.cmd_quality.write_jsonl")
    @patch("research_pipeline.cli.cmd_quality.compute_quality_score")
    @patch("research_pipeline.cli.cmd_quality.init_run")
    @patch("research_pipeline.cli.cmd_quality.load_config")
    def test_shortlist_json_path(
        self, mock_cfg, mock_init, mock_score, mock_write, tmp_path
    ):
        from research_pipeline.cli.cmd_quality import run_quality

        mock_cfg.return_value = _make_config(tmp_path)
        run_dir = tmp_path / "run1"
        mock_init.return_value = ("run1", run_dir)

        screen_dir = run_dir / "screen"
        screen_dir.mkdir(parents=True)
        search_dir = run_dir / "search"
        search_dir.mkdir(parents=True)

        candidate_data = _candidate_dict()
        (screen_dir / "shortlist.json").write_text(
            json.dumps([candidate_data]), encoding="utf-8"
        )

        qs = MagicMock()
        qs.model_dump.return_value = {"score": 0.8}
        mock_score.return_value = qs

        (run_dir / "quality").mkdir(parents=True)
        (run_dir / "summarize").mkdir(parents=True)

        run_quality(config_path=None, workspace=tmp_path, run_id="run1")

        mock_score.assert_called_once()


# ── 13. cmd_score_claims ─────────────────────────────────────────────────
class TestCmdScoreClaims:
    """Tests for cmd_score_claims."""

    @patch("research_pipeline.cli.cmd_score_claims.read_jsonl")
    @patch("research_pipeline.cli.cmd_score_claims.init_run")
    @patch("research_pipeline.cli.cmd_score_claims.load_config")
    def test_missing_claims_exits(self, mock_cfg, mock_init, mock_read, tmp_path):
        from research_pipeline.cli.cmd_score_claims import run_score_claims

        mock_cfg.return_value = _make_config(tmp_path)
        run_dir = tmp_path / "run1"
        mock_init.return_value = ("run1", run_dir)
        (run_dir / "summarize" / "claims").mkdir(parents=True)

        with pytest.raises(click.exceptions.Exit):
            run_score_claims(
                config_path=None,
                workspace=tmp_path,
                run_id="run1",
            )

    @patch("research_pipeline.cli.cmd_score_claims.write_jsonl")
    @patch("research_pipeline.cli.cmd_score_claims.score_decomposition")
    @patch("research_pipeline.cli.cmd_score_claims.create_llm_provider")
    @patch("research_pipeline.cli.cmd_score_claims.read_jsonl")
    @patch("research_pipeline.cli.cmd_score_claims.init_run")
    @patch("research_pipeline.cli.cmd_score_claims.load_config")
    def test_happy_path(
        self,
        mock_cfg,
        mock_init,
        mock_read,
        mock_llm,
        mock_score,
        mock_write,
        tmp_path,
    ):
        from research_pipeline.cli.cmd_score_claims import run_score_claims

        mock_cfg.return_value = _make_config(tmp_path)
        run_dir = tmp_path / "run1"
        mock_init.return_value = ("run1", run_dir)

        claims_dir = run_dir / "summarize" / "claims"
        claims_dir.mkdir(parents=True)
        (claims_dir / "claim_decomposition.jsonl").write_text("", encoding="utf-8")

        mock_read.return_value = [{"paper_id": "p1", "claims": []}]

        claim_mock = MagicMock()
        claim_mock.confidence_score = 0.85
        scored_decomp = MagicMock()
        scored_decomp.claims = [claim_mock]
        scored_decomp.paper_id = "p1"
        scored_decomp.model_dump.return_value = {"paper_id": "p1"}

        with patch(
            "research_pipeline.cli.cmd_score_claims.ClaimDecomposition"
        ) as MockCD:
            MockCD.model_validate.return_value = MagicMock(
                paper_id="p1", claims=[claim_mock]
            )
            mock_score.return_value = scored_decomp
            mock_llm.return_value = None

            run_score_claims(
                config_path=None,
                workspace=tmp_path,
                run_id="run1",
            )

            mock_score.assert_called_once()
            mock_write.assert_called_once()


# ── 14. cmd_search ───────────────────────────────────────────────────────
class TestCmdSearch:
    """Tests for cmd_search."""

    def test_resolve_sources_all(self):
        from research_pipeline.cli.cmd_search import _resolve_sources

        result = _resolve_sources("all", [])
        assert "arxiv" in result
        assert "scholar" in result

    def test_resolve_sources_specific(self):
        from research_pipeline.cli.cmd_search import _resolve_sources

        result = _resolve_sources("arxiv,scholar", [])
        assert result == ["arxiv", "scholar"]

    def test_resolve_sources_from_config(self):
        from research_pipeline.cli.cmd_search import _resolve_sources

        result = _resolve_sources(None, ["arxiv"])
        assert result == ["arxiv"]

    @patch("research_pipeline.cli.cmd_search.init_run")
    @patch("research_pipeline.cli.cmd_search.load_config")
    def test_no_plan_no_topic_exits(self, mock_cfg, mock_init, tmp_path):
        from research_pipeline.cli.cmd_search import run_search

        mock_cfg.return_value = _make_config(tmp_path)
        run_dir = tmp_path / "run1"
        mock_init.return_value = ("run1", run_dir)
        (run_dir / "plan").mkdir(parents=True)

        with pytest.raises(click.exceptions.Exit):
            run_search(
                topic=None,
                resume=False,
                config_path=None,
                workspace=tmp_path,
                run_id="run1",
                source="arxiv",
            )

    @patch("research_pipeline.cli.cmd_search.write_jsonl")
    @patch("research_pipeline.cli.cmd_search.dedup_cross_source")
    @patch("research_pipeline.cli.cmd_search.init_run")
    @patch("research_pipeline.cli.cmd_search.load_config")
    def test_happy_path_with_topic(
        self, mock_cfg, mock_init, mock_dedup, mock_write, tmp_path
    ):
        from research_pipeline.cli.cmd_search import run_search

        mock_cfg.return_value = _make_config(tmp_path)
        run_dir = tmp_path / "run1"
        mock_init.return_value = ("run1", run_dir)
        (run_dir / "plan").mkdir(parents=True)
        (run_dir / "search").mkdir(parents=True)

        mock_dedup.return_value = []

        # Use an unknown source so it just logs a warning and skips
        run_search(
            topic="transformer architectures",
            resume=False,
            config_path=None,
            workspace=tmp_path,
            run_id="run1",
            source="nonexistent_source",
        )

        mock_dedup.assert_called_once()
        mock_write.assert_called_once()

    @patch("research_pipeline.cli.cmd_search.write_jsonl")
    @patch("research_pipeline.cli.cmd_search.dedup_cross_source")
    @patch("research_pipeline.cli.cmd_search._search_arxiv")
    @patch("research_pipeline.cli.cmd_search.init_run")
    @patch("research_pipeline.cli.cmd_search.load_config")
    def test_happy_path_with_existing_plan(
        self, mock_cfg, mock_init, mock_arxiv, mock_dedup, mock_write, tmp_path
    ):
        from research_pipeline.cli.cmd_search import run_search

        mock_cfg.return_value = _make_config(tmp_path)
        run_dir = tmp_path / "run1"
        mock_init.return_value = ("run1", run_dir)

        plan_dir = run_dir / "plan"
        plan_dir.mkdir(parents=True)

        plan_data = {
            "topic_raw": "transformers",
            "topic_normalized": "transformers",
            "must_terms": ["transformer"],
            "nice_terms": ["attention"],
        }
        (plan_dir / "query_plan.json").write_text(
            json.dumps(plan_data), encoding="utf-8"
        )
        (run_dir / "search").mkdir(parents=True)

        candidate = MagicMock()
        candidate.model_dump.return_value = {"arxiv_id": "2301.00001"}
        mock_arxiv.return_value = [candidate]
        mock_dedup.return_value = [candidate]

        run_search(
            topic=None,
            resume=False,
            config_path=None,
            workspace=tmp_path,
            run_id="run1",
            source="arxiv",
        )

        mock_arxiv.assert_called_once()
        mock_dedup.assert_called_once()

    @patch("research_pipeline.cli.cmd_search.write_jsonl")
    @patch("research_pipeline.cli.cmd_search.dedup_cross_source")
    @patch("research_pipeline.cli.cmd_search.init_run")
    @patch("research_pipeline.cli.cmd_search.load_config")
    def test_search_import_error_handled(
        self, mock_cfg, mock_init, mock_dedup, mock_write, tmp_path
    ):
        """Source that raises ImportError is handled gracefully."""
        from research_pipeline.cli.cmd_search import run_search

        mock_cfg.return_value = _make_config(tmp_path)
        run_dir = tmp_path / "run1"
        mock_init.return_value = ("run1", run_dir)

        plan_dir = run_dir / "plan"
        plan_dir.mkdir(parents=True)
        plan_data = {
            "topic_raw": "test",
            "topic_normalized": "test",
            "must_terms": ["test"],
            "nice_terms": [],
        }
        (plan_dir / "query_plan.json").write_text(
            json.dumps(plan_data), encoding="utf-8"
        )
        (run_dir / "search").mkdir(parents=True)

        mock_dedup.return_value = []

        with patch(
            "research_pipeline.cli.cmd_search._search_arxiv",
            side_effect=ImportError("missing dep"),
        ):
            run_search(
                topic=None,
                resume=False,
                config_path=None,
                workspace=tmp_path,
                run_id="run1",
                source="arxiv",
            )

        mock_dedup.assert_called_once()


# ── 15. cmd_summarize ────────────────────────────────────────────────────
class TestCmdSummarize:
    """Tests for cmd_summarize."""

    @patch("research_pipeline.cli.cmd_summarize.read_jsonl")
    @patch("research_pipeline.cli.cmd_summarize.init_run")
    @patch("research_pipeline.cli.cmd_summarize.load_config")
    def test_missing_convert_manifest_exits(
        self, mock_cfg, mock_init, mock_read, tmp_path
    ):
        from research_pipeline.cli.cmd_summarize import run_summarize

        mock_cfg.return_value = _make_config(tmp_path)
        run_dir = tmp_path / "run1"
        mock_init.return_value = ("run1", run_dir)
        (run_dir / "plan").mkdir(parents=True)
        (run_dir / "screen").mkdir(parents=True)
        # "convert_root" → "convert" in STAGE_SUBDIRS
        (run_dir / "convert").mkdir(parents=True)

        with pytest.raises(click.exceptions.Exit):
            run_summarize(
                config_path=None,
                workspace=tmp_path,
                run_id="run1",
                output_format="markdown",
            )

    @patch("research_pipeline.cli.cmd_summarize.synthesize")
    @patch("research_pipeline.cli.cmd_summarize.summarize_paper")
    @patch("research_pipeline.cli.cmd_summarize.read_jsonl")
    @patch("research_pipeline.cli.cmd_summarize.init_run")
    @patch("research_pipeline.cli.cmd_summarize.load_config")
    def test_happy_path(
        self, mock_cfg, mock_init, mock_read, mock_summarize, mock_synth, tmp_path
    ):
        from research_pipeline.cli.cmd_summarize import run_summarize

        mock_cfg.return_value = _make_config(tmp_path)
        run_dir = tmp_path / "run1"
        mock_init.return_value = ("run1", run_dir)

        (run_dir / "plan").mkdir(parents=True)
        (run_dir / "screen").mkdir(parents=True)

        # "convert_root" → "convert" in STAGE_SUBDIRS
        conv_root = run_dir / "convert"
        conv_root.mkdir(parents=True)

        md_path = tmp_path / "paper.md"
        md_path.write_text("# Content", encoding="utf-8")

        conv_manifest_path = conv_root / "convert_manifest.jsonl"
        conv_manifest_path.write_text("", encoding="utf-8")

        mock_read.return_value = [
            _conv_entry_dict(markdown_path=str(md_path)),
        ]

        summary_mock = MagicMock()
        summary_mock.model_dump_json.return_value = '{"summary": "test"}'
        mock_summarize.return_value = summary_mock

        report_mock = MagicMock()
        report_mock.agreements = []
        report_mock.disagreements = []
        report_mock.model_dump_json.return_value = '{"agreements": []}'
        mock_synth.return_value = report_mock

        sum_dir = run_dir / "summarize"
        sum_dir.mkdir(parents=True)

        run_summarize(
            config_path=None,
            workspace=tmp_path,
            run_id="run1",
            output_format="markdown",
        )

        mock_summarize.assert_called_once()
        mock_synth.assert_called_once()

    @patch("research_pipeline.cli.cmd_summarize.synthesize")
    @patch("research_pipeline.cli.cmd_summarize.summarize_paper")
    @patch("research_pipeline.cli.cmd_summarize.read_jsonl")
    @patch("research_pipeline.cli.cmd_summarize.init_run")
    @patch("research_pipeline.cli.cmd_summarize.load_config")
    def test_structured_json_export(
        self, mock_cfg, mock_init, mock_read, mock_summarize, mock_synth, tmp_path
    ):
        from research_pipeline.cli.cmd_summarize import run_summarize

        mock_cfg.return_value = _make_config(tmp_path)
        run_dir = tmp_path / "run1"
        mock_init.return_value = ("run1", run_dir)

        (run_dir / "plan").mkdir(parents=True)
        (run_dir / "screen").mkdir(parents=True)

        # "convert_root" → "convert" in STAGE_SUBDIRS
        conv_root = run_dir / "convert"
        conv_root.mkdir(parents=True)
        (conv_root / "convert_manifest.jsonl").write_text("", encoding="utf-8")

        mock_read.return_value = []

        report_mock = MagicMock()
        report_mock.agreements = []
        report_mock.disagreements = []
        report_mock.model_dump_json.return_value = '{"agreements": []}'
        mock_synth.return_value = report_mock

        sum_dir = run_dir / "summarize"
        sum_dir.mkdir(parents=True)

        with patch(
            "research_pipeline.summarization.structured_output.export_structured_json"
        ) as mock_export:
            run_summarize(
                config_path=None,
                workspace=tmp_path,
                run_id="run1",
                output_format="structured-json",
            )
            mock_export.assert_called_once()


# ── 16. scholar_source ───────────────────────────────────────────────────
class TestScholarSource:
    """Tests for ScholarlySource and SerpAPISource."""

    def test_extract_arxiv_id_found(self):
        from research_pipeline.sources.scholar_source import _extract_arxiv_id

        aid, ver = _extract_arxiv_id("https://arxiv.org/abs/2301.12345")
        assert aid == "2301.12345"
        assert ver == "v1"

    def test_extract_arxiv_id_not_found(self):
        from research_pipeline.sources.scholar_source import _extract_arxiv_id

        aid, ver = _extract_arxiv_id("https://example.com/paper")
        assert aid == ""
        assert ver == ""

    def test_scholarly_source_name(self):
        from research_pipeline.sources.scholar_source import ScholarlySource

        src = ScholarlySource(min_interval=0.0)
        assert src.name == "scholar"

    def test_scholarly_source_import_error(self):
        from research_pipeline.sources.scholar_source import ScholarlySource

        src = ScholarlySource(min_interval=0.0)
        with (
            patch.dict("sys.modules", {"scholarly": None}),
            patch(
                "research_pipeline.sources.scholar_source.ScholarlySource.search",
                wraps=src.search,
            ),
        ):
            # Force ImportError on scholarly import
            import builtins

            original_import = builtins.__import__

            def mock_import(name, *args, **kwargs):
                if name == "scholarly":
                    raise ImportError("scholarly not installed")
                return original_import(name, *args, **kwargs)

            with patch("builtins.__import__", side_effect=mock_import):
                result = src.search(
                    topic="test",
                    must_terms=["test"],
                    nice_terms=[],
                    max_results=5,
                )
                assert result == []

    def test_scholarly_search_with_mock(self):
        from research_pipeline.sources.scholar_source import ScholarlySource

        src = ScholarlySource(min_interval=0.0)

        mock_scholarly = MagicMock()
        mock_result = {
            "bib": {
                "title": "Test Paper",
                "abstract": "An abstract",
                "author": ["Author A", "Author B"],
                "pub_year": "2023",
            },
            "pub_url": "https://arxiv.org/abs/2301.12345",
            "eprint_url": "",
        }
        mock_scholarly.search_pubs.return_value = iter([mock_result])

        import sys

        scholarly_mod = MagicMock(scholarly=mock_scholarly)
        with patch.dict(sys.modules, {"scholarly": scholarly_mod}):
            import builtins

            original_import = builtins.__import__

            def mock_import(name, *args, **kwargs):
                if name == "scholarly":
                    mod = MagicMock()
                    mod.scholarly = mock_scholarly
                    return mod
                return original_import(name, *args, **kwargs)

            with patch("builtins.__import__", side_effect=mock_import):
                result = src.search(
                    topic="test",
                    must_terms=["transformer"],
                    nice_terms=["attention"],
                    max_results=5,
                )
                assert len(result) == 1
                assert result[0].title == "Test Paper"
                assert result[0].arxiv_id == "2301.12345"

    def test_scholarly_parse_result_no_arxiv_id(self):
        from research_pipeline.sources.scholar_source import ScholarlySource

        src = ScholarlySource(min_interval=0.0)
        result = {
            "bib": {
                "title": "Conference Paper",
                "abstract": "",
                "author": "Author A and Author B",
                "pub_year": "",
                "venue": "NeurIPS",
            },
            "pub_url": "https://example.com/paper",
            "eprint_url": "",
        }
        candidate = src._parse_result(result)
        assert candidate.title == "Conference Paper"
        assert candidate.arxiv_id.startswith("scholar-")
        assert candidate.categories == ["NeurIPS"]

    def test_serpapi_source_name(self):
        from research_pipeline.sources.scholar_source import SerpAPISource

        src = SerpAPISource(api_key="test_key", min_interval=0.0)
        assert src.name == "serpapi"

    def test_serpapi_no_key_returns_empty(self):
        from research_pipeline.sources.scholar_source import SerpAPISource

        src = SerpAPISource(api_key="", min_interval=0.0)
        result = src.search(
            topic="test",
            must_terms=["test"],
            nice_terms=[],
            max_results=5,
        )
        assert result == []

    def test_serpapi_search_with_mock(self):
        from research_pipeline.sources.scholar_source import SerpAPISource

        src = SerpAPISource(api_key="fake_key", min_interval=0.0)

        mock_search_cls = MagicMock()
        mock_search_instance = MagicMock()
        mock_search_instance.get_dict.return_value = {
            "organic_results": [
                {
                    "title": "SerpAPI Paper",
                    "snippet": "A snippet",
                    "link": "https://arxiv.org/abs/2301.54321",
                    "publication_info": {"summary": "Author A, Author B - arXiv, 2023"},
                    "resources": [{"link": "http://example.com/pdf.pdf"}],
                }
            ]
        }
        mock_search_cls.return_value = mock_search_instance

        import builtins

        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "serpapi":
                mod = MagicMock()
                mod.GoogleSearch = mock_search_cls
                return mod
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            result = src.search(
                topic="test",
                must_terms=["transformer"],
                nice_terms=[],
                max_results=5,
                date_from="2023-01-01",
                date_to="2024-01-01",
            )
            assert len(result) == 1
            assert result[0].arxiv_id == "2301.54321"

    def test_serpapi_parse_result_no_arxiv_id(self):
        from research_pipeline.sources.scholar_source import SerpAPISource

        src = SerpAPISource(api_key="key", min_interval=0.0)
        result = {
            "title": "Some Paper",
            "snippet": "Description",
            "link": "https://example.com",
            "publication_info": {"summary": "Author - Journal, 2022"},
            "resources": [],
        }
        candidate = src._parse_result(result)
        assert candidate.title == "Some Paper"
        assert candidate.arxiv_id.startswith("scholar-")

    def test_serpapi_import_error(self):
        from research_pipeline.sources.scholar_source import SerpAPISource

        src = SerpAPISource(api_key="key", min_interval=0.0)

        import builtins

        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "serpapi":
                raise ImportError("serpapi not installed")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            result = src.search(
                topic="test",
                must_terms=["test"],
                nice_terms=[],
                max_results=5,
            )
            assert result == []


# ── 17. __main__ ─────────────────────────────────────────────────────────
class TestMain:
    """Tests for __main__.py — just verify it imports and calls app()."""

    def test_main_imports_app(self):
        from research_pipeline.cli.app import app

        assert callable(app)

    def test_main_module_source_has_app_call(self):
        """Verify __main__.py contains app() call without executing it."""
        main_file = (
            Path(__file__).resolve().parents[2]
            / "src"
            / "research_pipeline"
            / "__main__.py"
        )
        source = main_file.read_text(encoding="utf-8")
        assert "app()" in source
        assert "from research_pipeline.cli.app import app" in source
