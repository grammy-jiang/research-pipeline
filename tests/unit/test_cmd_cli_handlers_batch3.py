"""Tests for CLI handlers — batch 3 (coverage boost).

Covers previously-untested modules and functions to push overall
project coverage toward 90%.
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
    cfg.conversion.timeout_seconds = 120
    cfg.conversion.fallback_backends = []
    cfg.conversion.marker.force_ocr = False
    cfg.conversion.marker.use_llm = False
    cfg.conversion.marker.llm_service = None
    cfg.conversion.marker.llm_api_key = None
    cfg.conversion.mathpix.accounts = []
    cfg.conversion.mathpix.app_id = ""
    cfg.conversion.mathpix.app_key = ""
    cfg.conversion.datalab.accounts = []
    cfg.conversion.datalab.api_key = ""
    cfg.conversion.datalab.mode = "default"
    cfg.conversion.llamaparse.accounts = []
    cfg.conversion.llamaparse.api_key = ""
    cfg.conversion.llamaparse.tier = "free"
    cfg.conversion.mistral_ocr.accounts = []
    cfg.conversion.mistral_ocr.api_key = ""
    cfg.conversion.mistral_ocr.model = "default"
    cfg.conversion.openai_vision.accounts = []
    cfg.conversion.openai_vision.api_key = ""
    cfg.conversion.openai_vision.model = "gpt-4o"
    cfg.conversion.mineru.parse_method = "auto"
    cfg.conversion.mineru.timeout_seconds = 300
    cfg.search.max_query_variants = 5
    cfg.search.primary_months = 6
    cfg.search.fallback_months = 12
    cfg.llm = MagicMock()
    return cfg


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
        "abstract": "Test abstract about transformers",
        "abs_url": "http://example.com/abs",
        "pdf_url": "http://example.com/pdf",
    }
    base.update(overrides)
    return base


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


# ---------------------------------------------------------------------------
# cmd_plan — run_plan
# ---------------------------------------------------------------------------
class TestCmdPlan:
    """Tests for the plan CLI handler."""

    @patch("research_pipeline.cli.cmd_plan.augment_query_plan")
    @patch("research_pipeline.cli.cmd_plan.clean_query_terms")
    @patch("research_pipeline.cli.cmd_plan.load_config")
    def test_run_plan_basic(
        self,
        mock_config: MagicMock,
        mock_clean: MagicMock,
        mock_augment: MagicMock,
        tmp_path: Path,
    ) -> None:
        from research_pipeline.cli.cmd_plan import run_plan

        mock_config.return_value = _make_config(tmp_path)
        mock_clean.side_effect = lambda terms, **kw: terms
        mock_augment.return_value = ["transformers attention", "attention mechanisms"]

        run_plan(
            topic="transformer architectures for time series",
            workspace=tmp_path,
            run_id="test-plan-001",
        )

        plan_path = tmp_path / "test-plan-001" / "plan" / "query_plan.json"
        assert plan_path.exists()
        plan_data = json.loads(plan_path.read_text())
        assert plan_data["topic_raw"] == "transformer architectures for time series"
        assert len(plan_data["must_terms"]) > 0

    @patch("research_pipeline.cli.cmd_plan.augment_query_plan")
    @patch("research_pipeline.cli.cmd_plan.clean_query_terms")
    @patch("research_pipeline.cli.cmd_plan.load_config")
    def test_run_plan_stop_words_removed(
        self,
        mock_config: MagicMock,
        mock_clean: MagicMock,
        mock_augment: MagicMock,
        tmp_path: Path,
    ) -> None:
        from research_pipeline.cli.cmd_plan import (
            _split_topic_terms,
        )

        must, nice = _split_topic_terms("the best approach for deep learning")
        assert "the" not in must
        assert "for" not in must
        assert "best" in must or "best" in nice

    def test_generate_query_variants(self) -> None:
        from research_pipeline.cli.cmd_plan import _generate_query_variants

        variants = _generate_query_variants(
            must_terms=["transformers", "attention"],
            nice_terms=["time", "series"],
            max_variants=5,
        )
        assert len(variants) <= 5
        assert len(variants) >= 1
        # First variant should include all terms
        assert "transformers" in variants[0]

    def test_filter_stop_words(self) -> None:
        from research_pipeline.cli.cmd_plan import _filter_stop_words

        result = _filter_stop_words(["the", "deep", "learning", "for", "nlp"])
        assert result == ["deep", "learning", "nlp"]


# ---------------------------------------------------------------------------
# cmd_search — _resolve_sources and run_search
# ---------------------------------------------------------------------------
class TestCmdSearch:
    """Tests for the search CLI handler."""

    def test_resolve_sources_all(self) -> None:
        from research_pipeline.cli.cmd_search import _resolve_sources

        result = _resolve_sources("all", ["arxiv"])
        assert "arxiv" in result
        assert "scholar" in result
        assert "semantic_scholar" in result

    def test_resolve_sources_specific(self) -> None:
        from research_pipeline.cli.cmd_search import _resolve_sources

        result = _resolve_sources("arxiv,scholar", ["arxiv"])
        assert result == ["arxiv", "scholar"]

    def test_resolve_sources_default(self) -> None:
        from research_pipeline.cli.cmd_search import _resolve_sources

        result = _resolve_sources(None, ["arxiv", "dblp"])
        assert result == ["arxiv", "dblp"]

    @patch("research_pipeline.cli.cmd_search.dedup_cross_source")
    @patch("research_pipeline.cli.cmd_search._search_arxiv")
    @patch("research_pipeline.cli.cmd_search.load_config")
    def test_run_search_with_topic(
        self,
        mock_config: MagicMock,
        mock_search_arxiv: MagicMock,
        mock_dedup: MagicMock,
        tmp_path: Path,
    ) -> None:
        from research_pipeline.cli.cmd_search import run_search
        from research_pipeline.models.candidate import CandidateRecord

        cfg = _make_config(tmp_path)
        mock_config.return_value = cfg

        cand = CandidateRecord.model_validate(_candidate_dict())
        mock_search_arxiv.return_value = [cand]
        mock_dedup.return_value = [cand]

        run_search(
            topic="deep learning",
            workspace=tmp_path,
            run_id="test-search-001",
            source="arxiv",
        )

        candidates_path = tmp_path / "test-search-001" / "search" / "candidates.jsonl"
        assert candidates_path.exists()

    @patch("research_pipeline.cli.cmd_search.load_config")
    def test_run_search_no_topic_no_plan_exits(
        self,
        mock_config: MagicMock,
        tmp_path: Path,
    ) -> None:
        from research_pipeline.cli.cmd_search import run_search

        mock_config.return_value = _make_config(tmp_path)

        with pytest.raises(click.exceptions.Exit):
            run_search(
                topic=None,
                workspace=tmp_path,
                run_id="test-search-no-plan",
            )


# ---------------------------------------------------------------------------
# cmd_convert — _backend_kwargs_list + _create_converter + run_convert
# ---------------------------------------------------------------------------
class TestCmdConvert:
    """Tests for the convert CLI handler."""

    def test_backend_kwargs_list_docling(self) -> None:
        from research_pipeline.cli.cmd_convert import _backend_kwargs_list

        cfg = _make_config(Path("/fake"))
        result = _backend_kwargs_list("docling", cfg)
        assert len(result) == 1
        assert "timeout_seconds" in result[0]

    def test_backend_kwargs_list_marker(self) -> None:
        from research_pipeline.cli.cmd_convert import _backend_kwargs_list

        cfg = _make_config(Path("/fake"))
        result = _backend_kwargs_list("marker", cfg)
        assert len(result) == 1
        assert "force_ocr" in result[0]

    def test_backend_kwargs_list_pymupdf4llm(self) -> None:
        from research_pipeline.cli.cmd_convert import _backend_kwargs_list

        cfg = _make_config(Path("/fake"))
        result = _backend_kwargs_list("pymupdf4llm", cfg)
        assert result == [{}]

    def test_backend_kwargs_list_mathpix_default(self) -> None:
        from research_pipeline.cli.cmd_convert import _backend_kwargs_list

        cfg = _make_config(Path("/fake"))
        result = _backend_kwargs_list("mathpix", cfg)
        assert len(result) == 1
        assert "app_id" in result[0]

    def test_backend_kwargs_list_datalab_default(self) -> None:
        from research_pipeline.cli.cmd_convert import _backend_kwargs_list

        cfg = _make_config(Path("/fake"))
        result = _backend_kwargs_list("datalab", cfg)
        assert len(result) == 1
        assert "api_key" in result[0]

    def test_backend_kwargs_list_llamaparse_default(self) -> None:
        from research_pipeline.cli.cmd_convert import _backend_kwargs_list

        cfg = _make_config(Path("/fake"))
        result = _backend_kwargs_list("llamaparse", cfg)
        assert len(result) == 1
        assert "api_key" in result[0]

    def test_backend_kwargs_list_mistral_ocr_default(self) -> None:
        from research_pipeline.cli.cmd_convert import _backend_kwargs_list

        cfg = _make_config(Path("/fake"))
        result = _backend_kwargs_list("mistral_ocr", cfg)
        assert len(result) == 1
        assert "api_key" in result[0]

    def test_backend_kwargs_list_openai_vision_default(self) -> None:
        from research_pipeline.cli.cmd_convert import _backend_kwargs_list

        cfg = _make_config(Path("/fake"))
        result = _backend_kwargs_list("openai_vision", cfg)
        assert len(result) == 1
        assert "api_key" in result[0]

    def test_backend_kwargs_list_mineru(self) -> None:
        from research_pipeline.cli.cmd_convert import _backend_kwargs_list

        cfg = _make_config(Path("/fake"))
        result = _backend_kwargs_list("mineru", cfg)
        assert len(result) == 1
        assert "parse_method" in result[0]

    def test_backend_kwargs_list_mathpix_multi_account(self) -> None:
        from research_pipeline.cli.cmd_convert import _backend_kwargs_list

        cfg = _make_config(Path("/fake"))
        acct1 = MagicMock(app_id="id1", app_key="key1")
        acct2 = MagicMock(app_id="id2", app_key="key2")
        cfg.conversion.mathpix.accounts = [acct1, acct2]
        result = _backend_kwargs_list("mathpix", cfg)
        assert len(result) == 2
        assert result[0]["app_id"] == "id1"
        assert result[1]["app_id"] == "id2"

    def test_backend_kwargs_list_datalab_multi_account(self) -> None:
        from research_pipeline.cli.cmd_convert import _backend_kwargs_list

        cfg = _make_config(Path("/fake"))
        acct = MagicMock(api_key="k1", mode="fast")
        cfg.conversion.datalab.accounts = [acct]
        result = _backend_kwargs_list("datalab", cfg)
        assert len(result) == 1
        assert result[0]["api_key"] == "k1"

    def test_backend_kwargs_list_llamaparse_multi_account(self) -> None:
        from research_pipeline.cli.cmd_convert import _backend_kwargs_list

        cfg = _make_config(Path("/fake"))
        acct = MagicMock(api_key="k1", tier="premium")
        cfg.conversion.llamaparse.accounts = [acct]
        result = _backend_kwargs_list("llamaparse", cfg)
        assert result[0]["api_key"] == "k1"

    def test_backend_kwargs_list_mistral_multi_account(self) -> None:
        from research_pipeline.cli.cmd_convert import _backend_kwargs_list

        cfg = _make_config(Path("/fake"))
        acct = MagicMock(api_key="k1", model="m1")
        cfg.conversion.mistral_ocr.accounts = [acct]
        result = _backend_kwargs_list("mistral_ocr", cfg)
        assert result[0]["api_key"] == "k1"

    def test_backend_kwargs_list_openai_multi_account(self) -> None:
        from research_pipeline.cli.cmd_convert import _backend_kwargs_list

        cfg = _make_config(Path("/fake"))
        acct = MagicMock(api_key="k1", model="gpt-4o")
        cfg.conversion.openai_vision.accounts = [acct]
        result = _backend_kwargs_list("openai_vision", cfg)
        assert result[0]["api_key"] == "k1"

    def test_backend_kwargs_marker_with_llm(self) -> None:
        from research_pipeline.cli.cmd_convert import _backend_kwargs_list

        cfg = _make_config(Path("/fake"))
        cfg.conversion.marker.use_llm = True
        cfg.conversion.marker.llm_service = "openai"
        cfg.conversion.marker.llm_api_key = "sk-test"
        result = _backend_kwargs_list("marker", cfg)
        assert result[0]["use_llm"] is True
        assert result[0]["llm_service"] == "openai"
        assert result[0]["llm_api_key"] == "sk-test"

    @patch("research_pipeline.cli.cmd_convert.get_backend")
    @patch("research_pipeline.cli.cmd_convert._ensure_builtins_registered")
    @patch("research_pipeline.cli.cmd_convert.load_config")
    def test_run_convert_no_manifest_exits(
        self,
        mock_config: MagicMock,
        mock_ensure: MagicMock,
        mock_get_backend: MagicMock,
        tmp_path: Path,
    ) -> None:
        from research_pipeline.cli.cmd_convert import run_convert

        mock_config.return_value = _make_config(tmp_path)

        with pytest.raises(click.exceptions.Exit):
            run_convert(workspace=tmp_path, run_id="test-convert-001")

    @patch("research_pipeline.cli.cmd_convert.get_backend")
    @patch("research_pipeline.cli.cmd_convert._ensure_builtins_registered")
    @patch("research_pipeline.cli.cmd_convert.load_config")
    def test_run_convert_with_entries(
        self,
        mock_config: MagicMock,
        mock_ensure: MagicMock,
        mock_get_backend: MagicMock,
        tmp_path: Path,
    ) -> None:
        from research_pipeline.cli.cmd_convert import run_convert

        cfg = _make_config(tmp_path)
        mock_config.return_value = cfg

        # Set up run directory structure
        run_root = tmp_path / "test-convert-002"
        run_root.mkdir(parents=True)
        dl_root = run_root / "download"
        dl_root.mkdir(parents=True)

        # Create a fake PDF
        pdf_path = dl_root / "2301.00001v1.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 fake content")

        # Write download manifest
        dl_manifest_dir = run_root / "download"
        dl_manifest_path = dl_manifest_dir / "download_manifest.jsonl"
        entry = _dl_entry_dict(local_path=str(pdf_path))
        dl_manifest_path.write_text(json.dumps(entry) + "\n")

        # Mock converter
        mock_converter = MagicMock()
        mock_result = MagicMock(
            status="converted",
            model_dump=lambda mode="json": {
                "arxiv_id": "2301.00001",
                "status": "converted",
            },
        )
        mock_converter.convert.return_value = mock_result
        mock_get_backend.return_value = mock_converter

        run_convert(workspace=tmp_path, run_id="test-convert-002")

        mock_converter.convert.assert_called_once()


# ---------------------------------------------------------------------------
# cmd_analyze — run_analyze (prepare + collect)
# ---------------------------------------------------------------------------
class TestCmdAnalyze:
    """Tests for the analyze CLI handler."""

    @patch("research_pipeline.cli.cmd_analyze.load_config")
    def test_run_analyze_no_run_id(
        self,
        mock_config: MagicMock,
        tmp_path: Path,
    ) -> None:
        from research_pipeline.cli.cmd_analyze import run_analyze

        mock_config.return_value = _make_config(tmp_path)
        # No run_id → logs error and returns
        run_analyze(workspace=tmp_path, run_id=None)

    @patch("research_pipeline.cli.cmd_analyze.load_config")
    def test_run_analyze_prepare_no_papers(
        self,
        mock_config: MagicMock,
        tmp_path: Path,
    ) -> None:
        from research_pipeline.cli.cmd_analyze import run_analyze

        mock_config.return_value = _make_config(tmp_path)
        run_analyze(workspace=tmp_path, run_id="test-analyze-001")
        # Should complete without error (no papers found)

    @patch("research_pipeline.cli.cmd_analyze.load_config")
    def test_run_analyze_prepare_with_papers(
        self,
        mock_config: MagicMock,
        tmp_path: Path,
    ) -> None:
        from research_pipeline.cli.cmd_analyze import run_analyze

        mock_config.return_value = _make_config(tmp_path)

        # Create fake converted markdown
        run_root = tmp_path / "test-analyze-002"
        convert_dir = run_root / "convert"
        convert_dir.mkdir(parents=True)
        (convert_dir / "2301.00001.md").write_text("# Paper content")
        (convert_dir / "2301.00002.md").write_text("# Another paper")

        # Create query plan
        plan_dir = run_root / "plan"
        plan_dir.mkdir(parents=True)
        (plan_dir / "query_plan.json").write_text(json.dumps({"topic": "transformers"}))

        run_analyze(workspace=tmp_path, run_id="test-analyze-002")

        analysis_dir = run_root / "analysis"
        tasks_file = analysis_dir / "analysis_tasks.json"
        assert tasks_file.exists()
        tasks = json.loads(tasks_file.read_text())
        assert len(tasks) == 2

    @patch("research_pipeline.cli.cmd_analyze.load_config")
    def test_run_analyze_prepare_with_paper_id_filter(
        self,
        mock_config: MagicMock,
        tmp_path: Path,
    ) -> None:
        from research_pipeline.cli.cmd_analyze import run_analyze

        mock_config.return_value = _make_config(tmp_path)

        run_root = tmp_path / "test-analyze-003"
        convert_dir = run_root / "convert"
        convert_dir.mkdir(parents=True)
        (convert_dir / "2301.00001.md").write_text("# Paper 1")
        (convert_dir / "2301.00002.md").write_text("# Paper 2")

        plan_dir = run_root / "plan"
        plan_dir.mkdir(parents=True)
        (plan_dir / "query_plan.json").write_text(json.dumps({"topic": "AI"}))

        run_analyze(
            workspace=tmp_path,
            run_id="test-analyze-003",
            paper_ids=["2301.00001"],
        )

        analysis_dir = run_root / "analysis"
        tasks = json.loads((analysis_dir / "analysis_tasks.json").read_text())
        assert len(tasks) == 1
        assert tasks[0]["arxiv_id"] == "2301.00001"

    @patch("research_pipeline.cli.cmd_analyze.load_config")
    def test_run_analyze_collect_no_files(
        self,
        mock_config: MagicMock,
        tmp_path: Path,
    ) -> None:
        from research_pipeline.cli.cmd_analyze import run_analyze

        mock_config.return_value = _make_config(tmp_path)
        # Should log error but not crash
        run_analyze(workspace=tmp_path, run_id="test-analyze-collect", collect=True)

    @patch("research_pipeline.cli.cmd_analyze.load_config")
    def test_run_analyze_collect_valid_json(
        self,
        mock_config: MagicMock,
        tmp_path: Path,
    ) -> None:
        from research_pipeline.cli.cmd_analyze import run_analyze

        mock_config.return_value = _make_config(tmp_path)

        run_root = tmp_path / "test-analyze-cv"
        analysis_dir = run_root / "analysis"
        analysis_dir.mkdir(parents=True)

        analysis_data = {
            "arxiv_id": "2301.00001",
            "title": "Test Paper",
            "ratings": {
                "methodology": {
                    "score": 4,
                    "justification": "Very solid methodology with good controls",
                },
                "experimental_rigor": {
                    "score": 3,
                    "justification": "Adequate experimental setup and baselines",
                },
                "novelty": {
                    "score": 5,
                    "justification": "Highly novel approach to the problem",
                },
                "practical_value": {
                    "score": 4,
                    "justification": "Strong practical applications shown in paper",
                },
                "overall": {
                    "score": 4,
                    "justification": "Good paper overall with strong contributions",
                },
            },
            "methodology_assessment": "Good",
            "key_findings": [
                {"finding": "Transformers work well", "confidence": "high"}
            ],
            "strengths": ["Clear writing"],
            "weaknesses": ["Limited baselines"],
            "limitations": ["Small dataset"],
            "evidence_quotes": ["Quote from paper"],
            "key_contributions": ["New architecture"],
            "reproducibility": "High",
            "relevance_to_topic": "Very relevant",
        }
        (analysis_dir / "2301.00001_analysis.json").write_text(
            json.dumps(analysis_data)
        )

        run_analyze(workspace=tmp_path, run_id="test-analyze-cv", collect=True)

        report = analysis_dir / "validation_report.json"
        assert report.exists()
        result = json.loads(report.read_text())
        assert result["valid"] == 1

    def test_validate_analysis_json_missing_fields(self, tmp_path: Path) -> None:
        from research_pipeline.cli.cmd_analyze import _validate_analysis_json

        f = tmp_path / "test.json"
        f.write_text(json.dumps({"arxiv_id": "123"}))
        errors = _validate_analysis_json(f)
        assert len(errors) > 0
        assert any("Missing required fields" in e for e in errors)

    def test_validate_analysis_json_bad_rating(self, tmp_path: Path) -> None:
        from research_pipeline.cli.cmd_analyze import _validate_analysis_json

        data = {
            "arxiv_id": "123",
            "title": "T",
            "ratings": {
                "methodology": {"score": 10, "justification": "bad"},
                "experimental_rigor": {
                    "score": 1,
                    "justification": "ok stuff here now",
                },
                "novelty": {"score": 1, "justification": "ok stuff here now"},
                "practical_value": {"score": 1, "justification": "ok stuff here now"},
                "overall": {"score": 1, "justification": "ok stuff here now"},
            },
            "methodology_assessment": "ok",
            "key_findings": [{"finding": "x", "confidence": "invalid"}],
            "strengths": [],
            "weaknesses": [],
            "limitations": [],
            "evidence_quotes": [],
            "key_contributions": [],
            "reproducibility": "ok",
            "relevance_to_topic": "ok",
        }
        f = tmp_path / "test.json"
        f.write_text(json.dumps(data))
        errors = _validate_analysis_json(f)
        assert any("score must be int 1-5" in e for e in errors)
        assert any("confidence must be high/medium/low" in e for e in errors)

    def test_validate_analysis_json_invalid_json(self, tmp_path: Path) -> None:
        from research_pipeline.cli.cmd_analyze import _validate_analysis_json

        f = tmp_path / "bad.json"
        f.write_text("not json")
        errors = _validate_analysis_json(f)
        assert len(errors) == 1
        assert "Cannot read" in errors[0]

    def test_validate_analysis_json_ratings_not_dict(self, tmp_path: Path) -> None:
        from research_pipeline.cli.cmd_analyze import _validate_analysis_json

        data = {
            "arxiv_id": "123",
            "title": "T",
            "ratings": "not a dict",
            "methodology_assessment": "ok",
            "key_findings": [],
            "strengths": [],
            "weaknesses": [],
            "limitations": [],
            "evidence_quotes": [],
            "key_contributions": [],
            "reproducibility": "ok",
            "relevance_to_topic": "ok",
        }
        f = tmp_path / "test.json"
        f.write_text(json.dumps(data))
        errors = _validate_analysis_json(f)
        assert any("must be a dict" in e for e in errors)


# ---------------------------------------------------------------------------
# cmd_validate — validate_report + helper functions
# ---------------------------------------------------------------------------
class TestCmdValidate:
    """Tests for the validate CLI handler and helper functions."""

    def test_extract_headings(self) -> None:
        from research_pipeline.cli.cmd_validate import _extract_headings

        text = "# Executive Summary\n## Research Question\nSome text\n### Methodology\n"
        headings = _extract_headings(text)
        assert "executive summary" in headings
        assert "research question" in headings
        assert "methodology" in headings

    def test_check_sections_all_present(self) -> None:
        from research_pipeline.cli.cmd_validate import _check_sections

        text = "\n".join(
            f"# {s.title()}"
            for s in [
                "executive summary",
                "research question",
                "methodology",
                "papers reviewed",
                "research landscape",
                "research gaps",
                "practical recommendations",
                "references",
                "appendix",
            ]
        )
        present, missing, cond, opt_missing = _check_sections(text)
        assert len(missing) == 0
        assert len(present) == 9

    def test_check_sections_some_missing(self) -> None:
        from research_pipeline.cli.cmd_validate import _check_sections

        text = "# Executive Summary\n# Research Question\n"
        present, missing, _, _ = _check_sections(text)
        assert "executive summary" in present
        assert len(missing) > 0

    def test_check_confidence_levels(self) -> None:
        from research_pipeline.cli.cmd_validate import _check_confidence_levels

        text = "🟢 high result\n🟡 medium result\n🔴 low result\nconfidence: high\n"
        counts = _check_confidence_levels(text)
        assert counts["high"] >= 2
        assert counts["medium"] >= 1
        assert counts["low"] >= 1

    def test_check_evidence_citations(self) -> None:
        from research_pipeline.cli.cmd_validate import _check_evidence_citations

        text = "As shown in [2301.00001] and [Author, 2023], the results [xyz]."
        count = _check_evidence_citations(text)
        assert count == 3

    def test_check_gap_classification(self) -> None:
        from research_pipeline.cli.cmd_validate import _check_gap_classification

        text = "This is an ACADEMIC gap. Another ENGINEERING gap. ACADEMIC again."
        gaps = _check_gap_classification(text)
        assert gaps["academic_gaps"] == 2
        assert gaps["engineering_gaps"] == 1

    def test_check_tables(self) -> None:
        from research_pipeline.cli.cmd_validate import _check_tables

        text = (
            "| Col1 | Col2 |\n| --- | --- |\n| val | val |\n"
            "\nText\n\n| A | B |\n| - | - |\n"
        )
        count = _check_tables(text)
        assert count == 2

    def test_check_mermaid(self) -> None:
        from research_pipeline.cli.cmd_validate import _check_mermaid

        text = (
            "```mermaid\ngraph TD\n```\nsome text\n```mermaid\nsequenceDiagram\n```\n"
        )
        assert _check_mermaid(text) == 2

    def test_check_latex(self) -> None:
        from research_pipeline.cli.cmd_validate import _check_latex

        text = "Inline $x^2$ and display $$E = mc^2$$"
        count = _check_latex(text)
        assert count >= 2

    @patch("research_pipeline.cli.cmd_validate.compute_race_score")
    @patch("research_pipeline.cli.cmd_validate.compute_fact_score")
    def test_validate_report_pass(
        self,
        mock_fact: MagicMock,
        mock_race: MagicMock,
        tmp_path: Path,
    ) -> None:
        from research_pipeline.cli.cmd_validate import validate_report

        mock_race.return_value = MagicMock(model_dump=lambda: {"overall": 0.8})

        report = tmp_path / "report.md"
        sections = [
            "# Executive Summary",
            "Good summary here.",
            "# Research Question",
            "What is the question?",
            "# Methodology",
            "```mermaid\ngraph TD\n```",
            "# Papers Reviewed",
            "| Paper | Score |\n| --- | --- |\n| A | 0.9 |",
            "# Research Landscape",
            "| Aspect | Detail |\n| --- | --- |\n| X | Y |",
            "# Research Gaps",
            "ACADEMIC gap found. ENGINEERING gap too.",
            "# Practical Recommendations",
            "🟢 high confidence result [2301.00001] [2301.00002] [2301.00003]",
            "# References",
            "List of refs",
            "# Appendix",
            "Extra data",
        ]
        report.write_text("\n".join(sections))

        result = validate_report(report)
        assert result["verdict"] == "PASS"
        assert result["overall_score"] >= 0.7

    @patch("research_pipeline.cli.cmd_validate.compute_race_score")
    def test_validate_report_fail_missing_sections(
        self,
        mock_race: MagicMock,
        tmp_path: Path,
    ) -> None:
        from research_pipeline.cli.cmd_validate import validate_report

        mock_race.return_value = MagicMock(model_dump=lambda: {"overall": 0.3})

        report = tmp_path / "report.md"
        report.write_text("# Executive Summary\nSome text")

        result = validate_report(report)
        assert result["verdict"] == "FAIL"
        assert len(result["sections"]["missing_required"]) > 0

    @patch("research_pipeline.cli.cmd_validate.compute_race_score")
    @patch("research_pipeline.cli.cmd_validate.compute_fact_score")
    def test_validate_report_with_fact_score(
        self,
        mock_fact: MagicMock,
        mock_race: MagicMock,
        tmp_path: Path,
    ) -> None:
        from research_pipeline.cli.cmd_validate import validate_report

        mock_race.return_value = MagicMock(model_dump=lambda: {"overall": 0.8})
        mock_fact.return_value = MagicMock(
            model_dump=lambda: {
                "citation_accuracy": 0.9,
                "effective_citation_ratio": 0.8,
                "verified_citations": 5,
                "total_citations": 6,
            },
            citation_accuracy=0.9,
            effective_citation_ratio=0.8,
            unsupported_citations=[],
            uncited_papers=[],
        )

        report = tmp_path / "report.md"
        report.write_text("# Executive Summary\n[2301.00001] [2301.00002] [2301.00003]")
        result = validate_report(report, paper_ids=["2301.00001", "2301.00002"])
        assert "fact_score" in result

    @patch("research_pipeline.cli.cmd_validate.compute_race_score")
    @patch("research_pipeline.cli.cmd_validate.compute_fact_score")
    def test_validate_report_low_fact_score_issues(
        self,
        mock_fact: MagicMock,
        mock_race: MagicMock,
        tmp_path: Path,
    ) -> None:
        from research_pipeline.cli.cmd_validate import validate_report

        mock_race.return_value = MagicMock(model_dump=lambda: {"overall": 0.5})
        mock_fact.return_value = MagicMock(
            model_dump=lambda: {
                "citation_accuracy": 0.3,
                "effective_citation_ratio": 0.1,
            },
            citation_accuracy=0.3,
            effective_citation_ratio=0.1,
            unsupported_citations=["fake1", "fake2"],
            uncited_papers=["p1", "p2", "p3"],
        )

        report = tmp_path / "report.md"
        report.write_text("# Summary\n[ref1]")
        result = validate_report(report, paper_ids=["p1", "p2"])
        assert any("Low citation accuracy" in i for i in result["issues"])
        assert any("Low citation coverage" in i for i in result["issues"])

    def test_run_validate_no_report_no_run_id(self) -> None:
        from research_pipeline.cli.cmd_validate import run_validate

        # Should not crash — logs error and returns
        run_validate(report=None, workspace=None, run_id=None)

    @patch("research_pipeline.cli.cmd_validate.compute_race_score")
    def test_run_validate_with_report_file(
        self,
        mock_race: MagicMock,
        tmp_path: Path,
    ) -> None:
        from research_pipeline.cli.cmd_validate import run_validate

        mock_race.return_value = MagicMock(model_dump=lambda: {"overall": 0.5})

        report = tmp_path / "test_report.md"
        report.write_text("# Executive Summary\nSome content")

        run_validate(report=report)

        result_path = tmp_path / "validation_result.json"
        assert result_path.exists()

    @patch("research_pipeline.cli.cmd_validate.compute_race_score")
    def test_run_validate_with_output_path(
        self,
        mock_race: MagicMock,
        tmp_path: Path,
    ) -> None:
        from research_pipeline.cli.cmd_validate import run_validate

        mock_race.return_value = MagicMock(model_dump=lambda: {"overall": 0.5})

        report = tmp_path / "report.md"
        report.write_text("# Executive Summary\nContent")
        output = tmp_path / "custom_output.json"

        run_validate(report=report, output=output)
        assert output.exists()


# ---------------------------------------------------------------------------
# cmd_kg — kg-stats, kg-query, kg-ingest
# ---------------------------------------------------------------------------
class TestCmdKg:
    """Tests for the knowledge graph CLI handlers."""

    @patch("research_pipeline.cli.cmd_kg.KnowledgeGraph")
    def test_run_kg_stats(self, mock_kg_class: MagicMock) -> None:
        from research_pipeline.cli.cmd_kg import run_kg_stats

        mock_kg = MagicMock()
        mock_kg.stats.return_value = {
            "total_entities": 100,
            "total_triples": 250,
            "entities": {"paper": 80, "author": 20},
            "triples": {"authored_by": 150, "cites": 100},
        }
        mock_kg_class.return_value = mock_kg

        run_kg_stats()
        mock_kg.stats.assert_called_once()
        mock_kg.close.assert_called_once()

    @patch("research_pipeline.cli.cmd_kg.KnowledgeGraph")
    def test_run_kg_query_found(self, mock_kg_class: MagicMock) -> None:
        from research_pipeline.cli.cmd_kg import run_kg_query

        mock_kg = MagicMock()
        mock_entity = MagicMock()
        mock_entity.name = "Test Paper"
        mock_entity.entity_type.value = "paper"
        mock_entity.properties = {"year": 2023}
        mock_kg.get_entity.return_value = mock_entity

        mock_triple = MagicMock()
        mock_triple.subject_id = "test-id"
        mock_triple.object_id = "other-id"
        mock_triple.relation.value = "cites"
        mock_triple.confidence = 0.95
        mock_kg.get_neighbors.return_value = [mock_triple]

        mock_kg_class.return_value = mock_kg

        run_kg_query("test-id")
        mock_kg.get_entity.assert_called_once_with("test-id")
        mock_kg.close.assert_called_once()

    @patch("research_pipeline.cli.cmd_kg.KnowledgeGraph")
    def test_run_kg_query_not_found(self, mock_kg_class: MagicMock) -> None:
        from research_pipeline.cli.cmd_kg import run_kg_query

        mock_kg = MagicMock()
        mock_kg.get_entity.return_value = None
        mock_kg_class.return_value = mock_kg

        with pytest.raises(click.exceptions.Exit):
            run_kg_query("nonexistent")
        mock_kg.close.assert_called_once()

    @patch("research_pipeline.cli.cmd_kg.KnowledgeGraph")
    def test_run_kg_query_no_neighbors(self, mock_kg_class: MagicMock) -> None:
        from research_pipeline.cli.cmd_kg import run_kg_query

        mock_kg = MagicMock()
        mock_entity = MagicMock()
        mock_entity.name = "Paper"
        mock_entity.entity_type.value = "paper"
        mock_entity.properties = {}
        mock_kg.get_entity.return_value = mock_entity
        mock_kg.get_neighbors.return_value = []
        mock_kg_class.return_value = mock_kg

        run_kg_query("test-id")
        mock_kg.close.assert_called_once()


# ---------------------------------------------------------------------------
# cmd_setup — run_setup, _find_skill_source, _install_directory
# ---------------------------------------------------------------------------
class TestCmdSetup:
    """Tests for the setup CLI handler."""

    def test_run_setup_skip_both(self, tmp_path: Path) -> None:
        from research_pipeline.cli.cmd_setup import run_setup

        # Both skipped — should just warn and return
        run_setup(
            skill_target=tmp_path / "skill",
            agents_target=tmp_path / "agents",
            skip_skill=True,
            skip_agents=True,
        )

    @patch("research_pipeline.cli.cmd_setup._find_skill_source")
    def test_run_setup_skill_source_not_found(
        self, mock_find: MagicMock, tmp_path: Path
    ) -> None:
        from research_pipeline.cli.cmd_setup import run_setup

        mock_find.return_value = None
        with pytest.raises(SystemExit):
            run_setup(
                skill_target=tmp_path / "skill",
                agents_target=tmp_path / "agents",
                skip_agents=True,
            )

    @patch("research_pipeline.cli.cmd_setup._find_agent_source")
    def test_run_setup_agent_source_not_found(
        self, mock_find: MagicMock, tmp_path: Path
    ) -> None:
        from research_pipeline.cli.cmd_setup import run_setup

        mock_find.return_value = None
        with pytest.raises(SystemExit):
            run_setup(
                skill_target=tmp_path / "skill",
                agents_target=tmp_path / "agents",
                skip_skill=True,
            )

    def test_install_directory_copy(self, tmp_path: Path) -> None:
        from research_pipeline.cli.cmd_setup import _install_directory

        source = tmp_path / "src_dir"
        source.mkdir()
        (source / "file.txt").write_text("content")

        target = tmp_path / "dest_dir"
        _install_directory(source, target, symlink=False, force=False, label="Test")
        assert target.exists()
        assert (target / "file.txt").read_text() == "content"

    def test_install_directory_symlink(self, tmp_path: Path) -> None:
        from research_pipeline.cli.cmd_setup import _install_directory

        source = tmp_path / "src_dir"
        source.mkdir()
        (source / "file.txt").write_text("content")

        target = tmp_path / "link_dir"
        _install_directory(source, target, symlink=True, force=False, label="Test")
        assert target.is_symlink()

    def test_install_directory_force_overwrite(self, tmp_path: Path) -> None:
        from research_pipeline.cli.cmd_setup import _install_directory

        source = tmp_path / "src_dir"
        source.mkdir()
        (source / "file.txt").write_text("new content")

        target = tmp_path / "dest_dir"
        target.mkdir()
        (target / "old.txt").write_text("old")

        _install_directory(source, target, symlink=False, force=True, label="Test")
        assert (target / "file.txt").exists()
        assert not (target / "old.txt").exists()

    def test_install_directory_no_force_raises(self, tmp_path: Path) -> None:
        from research_pipeline.cli.cmd_setup import _install_directory

        source = tmp_path / "src_dir"
        source.mkdir()

        target = tmp_path / "dest_dir"
        target.mkdir()

        with pytest.raises(SystemExit):
            _install_directory(source, target, symlink=False, force=False, label="Test")

    def test_install_directory_force_overwrite_symlink(self, tmp_path: Path) -> None:
        from research_pipeline.cli.cmd_setup import _install_directory

        source = tmp_path / "src_dir"
        source.mkdir()
        (source / "f.txt").write_text("data")

        target = tmp_path / "link_dir"
        # Create existing symlink to remove
        dummy = tmp_path / "dummy"
        dummy.mkdir()
        target.symlink_to(dummy)

        _install_directory(source, target, symlink=False, force=True, label="Test")
        assert not target.is_symlink()
        assert target.is_dir()

    def test_install_directory_force_overwrite_file(self, tmp_path: Path) -> None:
        from research_pipeline.cli.cmd_setup import _install_directory

        source = tmp_path / "src_dir"
        source.mkdir()
        (source / "f.txt").write_text("data")

        target = tmp_path / "target_file"
        target.write_text("I am a file")

        _install_directory(source, target, symlink=False, force=True, label="Test")
        assert target.is_dir()

    def test_install_agent_files_copy(self, tmp_path: Path) -> None:
        from research_pipeline.cli.cmd_setup import _install_agent_files

        source = tmp_path / "agents_src"
        source.mkdir()
        (source / "agent1.md").write_text("# Agent 1")
        (source / "agent2.md").write_text("# Agent 2")

        target = tmp_path / "agents_dest"
        count = _install_agent_files(source, target, symlink=False, force=False)
        assert count == 2
        assert (target / "agent1.md").exists()

    def test_install_agent_files_symlink(self, tmp_path: Path) -> None:
        from research_pipeline.cli.cmd_setup import _install_agent_files

        source = tmp_path / "agents_src"
        source.mkdir()
        (source / "agent1.md").write_text("# Agent 1")

        target = tmp_path / "agents_dest"
        count = _install_agent_files(source, target, symlink=True, force=False)
        assert count == 1
        assert (target / "agent1.md").is_symlink()

    def test_install_agent_files_no_force_skips(self, tmp_path: Path) -> None:
        from research_pipeline.cli.cmd_setup import _install_agent_files

        source = tmp_path / "agents_src"
        source.mkdir()
        (source / "agent1.md").write_text("# Agent 1")

        target = tmp_path / "agents_dest"
        target.mkdir()
        (target / "agent1.md").write_text("existing")

        count = _install_agent_files(source, target, symlink=False, force=False)
        assert count == 0

    def test_install_agent_files_force_overwrites(self, tmp_path: Path) -> None:
        from research_pipeline.cli.cmd_setup import _install_agent_files

        source = tmp_path / "agents_src"
        source.mkdir()
        (source / "agent1.md").write_text("# New Agent 1")

        target = tmp_path / "agents_dest"
        target.mkdir()
        (target / "agent1.md").write_text("old")

        count = _install_agent_files(source, target, symlink=False, force=True)
        assert count == 1
        assert (target / "agent1.md").read_text() == "# New Agent 1"

    def test_install_agent_files_force_overwrite_symlink(self, tmp_path: Path) -> None:
        from research_pipeline.cli.cmd_setup import _install_agent_files

        source = tmp_path / "agents_src"
        source.mkdir()
        (source / "agent1.md").write_text("# Agent 1")

        target = tmp_path / "agents_dest"
        target.mkdir()
        dummy = tmp_path / "dummy.md"
        dummy.write_text("dummy")
        (target / "agent1.md").symlink_to(dummy)

        count = _install_agent_files(source, target, symlink=False, force=True)
        assert count == 1
        assert not (target / "agent1.md").is_symlink()


# ---------------------------------------------------------------------------
# conversion/pymupdf4llm_backend — PyMuPDF4LLMBackend
# ---------------------------------------------------------------------------
class TestPyMuPDF4LLMBackend:
    """Tests for the pymupdf4llm converter backend."""

    def test_fingerprint(self) -> None:
        from research_pipeline.conversion.pymupdf4llm_backend import (
            PyMuPDF4LLMBackend,
        )

        backend = PyMuPDF4LLMBackend(page_chunks=False)
        fp = backend.fingerprint()
        assert fp.startswith("pymupdf4llm/")

    def test_version_property(self) -> None:
        from research_pipeline.conversion.pymupdf4llm_backend import (
            PyMuPDF4LLMBackend,
        )

        backend = PyMuPDF4LLMBackend()
        v = backend.version
        assert isinstance(v, str)
        # Call again to exercise cache
        assert backend.version == v

    def test_convert_skip_existing(self, tmp_path: Path) -> None:
        from research_pipeline.conversion.pymupdf4llm_backend import (
            PyMuPDF4LLMBackend,
        )

        backend = PyMuPDF4LLMBackend()
        pdf_path = tmp_path / "2301.00001v1.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 fake")
        md_path = tmp_path / "output" / "2301.00001v1.md"
        md_path.parent.mkdir(parents=True)
        md_path.write_text("# Existing")

        result = backend.convert(pdf_path, tmp_path / "output")
        assert result.status == "skipped_exists"

    def test_convert_force_removes_existing(self, tmp_path: Path) -> None:
        from research_pipeline.conversion.pymupdf4llm_backend import (
            PyMuPDF4LLMBackend,
        )

        backend = PyMuPDF4LLMBackend()
        pdf_path = tmp_path / "2301.00001v1.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 fake")
        md_path = tmp_path / "output" / "2301.00001v1.md"
        md_path.parent.mkdir(parents=True)
        md_path.write_text("# Old content")

        with patch(
            "research_pipeline.conversion.pymupdf4llm_backend.pymupdf4llm",
            create=True,
        ) as mock_lib:
            mock_lib.to_markdown.return_value = "# New content"
            # The import happens inside the convert method, so we need to
            # patch at the right level
            import sys

            sys.modules["pymupdf4llm"] = mock_lib
            try:
                result = backend.convert(pdf_path, tmp_path / "output", force=True)
                assert result.status == "converted"
            finally:
                del sys.modules["pymupdf4llm"]

    @patch.dict("sys.modules", {"pymupdf4llm": None})
    def test_convert_import_error(self, tmp_path: Path) -> None:
        from research_pipeline.conversion.pymupdf4llm_backend import (
            PyMuPDF4LLMBackend,
        )

        backend = PyMuPDF4LLMBackend()
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 fake")

        result = backend.convert(pdf_path, tmp_path / "output")
        assert result.status == "failed"
        assert "not installed" in result.error

    def test_convert_exception_handling(self, tmp_path: Path) -> None:
        from research_pipeline.conversion.pymupdf4llm_backend import (
            PyMuPDF4LLMBackend,
        )

        backend = PyMuPDF4LLMBackend()
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 fake")

        mock_lib = MagicMock()
        mock_lib.to_markdown.side_effect = RuntimeError("conversion error")

        import sys

        sys.modules["pymupdf4llm"] = mock_lib
        try:
            result = backend.convert(pdf_path, tmp_path / "output")
            assert result.status == "failed"
            assert "conversion error" in result.error
        finally:
            del sys.modules["pymupdf4llm"]

    def test_convert_arxiv_id_extraction(self, tmp_path: Path) -> None:
        from research_pipeline.conversion.pymupdf4llm_backend import (
            PyMuPDF4LLMBackend,
        )

        backend = PyMuPDF4LLMBackend()
        pdf_path = tmp_path / "2301.00001v2.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 fake")
        md_path = tmp_path / "output" / "2301.00001v2.md"
        md_path.parent.mkdir(parents=True)
        md_path.write_text("# Content")

        result = backend.convert(pdf_path, tmp_path / "output")
        assert result.arxiv_id == "2301.00001"
        assert result.version == "v2"


# ---------------------------------------------------------------------------
# conversion/marker_backend — MarkerBackend
# ---------------------------------------------------------------------------
class TestMarkerBackend:
    """Tests for the marker converter backend."""

    def test_fingerprint(self) -> None:
        from research_pipeline.conversion.marker_backend import MarkerBackend

        backend = MarkerBackend(force_ocr=True, use_llm=False)
        fp = backend.fingerprint()
        assert fp.startswith("marker/")

    def test_version_property(self) -> None:
        from research_pipeline.conversion.marker_backend import MarkerBackend

        backend = MarkerBackend()
        v = backend.version
        assert isinstance(v, str)
        assert backend.version == v

    def test_convert_skip_existing(self, tmp_path: Path) -> None:
        from research_pipeline.conversion.marker_backend import MarkerBackend

        backend = MarkerBackend()
        pdf_path = tmp_path / "2301.00001v1.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 fake")
        md_path = tmp_path / "output" / "2301.00001v1.md"
        md_path.parent.mkdir(parents=True)
        md_path.write_text("# Existing")

        result = backend.convert(pdf_path, tmp_path / "output")
        assert result.status == "skipped_exists"

    @patch.dict(
        "sys.modules",
        {
            "marker": None,
            "marker.converters": None,
            "marker.converters.pdf": None,
            "marker.models": None,
        },
    )
    def test_convert_import_error(self, tmp_path: Path) -> None:
        from research_pipeline.conversion.marker_backend import MarkerBackend

        backend = MarkerBackend()
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 fake")

        result = backend.convert(pdf_path, tmp_path / "output")
        assert result.status == "failed"
        assert "not installed" in result.error

    def test_convert_exception_handling(self, tmp_path: Path) -> None:
        from research_pipeline.conversion.marker_backend import MarkerBackend

        backend = MarkerBackend()
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 fake")

        mock_marker_pdf = MagicMock()
        mock_marker_models = MagicMock()
        mock_marker_pdf.PdfConverter.side_effect = RuntimeError("marker error")
        mock_marker_models.create_model_dict.return_value = {}

        import sys

        sys.modules["marker"] = MagicMock()
        sys.modules["marker.converters"] = MagicMock()
        sys.modules["marker.converters.pdf"] = mock_marker_pdf
        sys.modules["marker.models"] = mock_marker_models
        try:
            result = backend.convert(pdf_path, tmp_path / "output")
            assert result.status == "failed"
            assert "marker error" in result.error
        finally:
            for mod in [
                "marker",
                "marker.converters",
                "marker.converters.pdf",
                "marker.models",
            ]:
                sys.modules.pop(mod, None)

    def test_convert_force_removes_existing(self, tmp_path: Path) -> None:
        from research_pipeline.conversion.marker_backend import MarkerBackend

        backend = MarkerBackend()
        pdf_path = tmp_path / "2301.00001v1.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 fake")
        md_path = tmp_path / "output" / "2301.00001v1.md"
        md_path.parent.mkdir(parents=True)
        md_path.write_text("# Old")

        mock_marker_pdf = MagicMock()
        mock_marker_models = MagicMock()
        mock_marker_models.create_model_dict.return_value = {}
        mock_rendered = MagicMock()
        mock_rendered.markdown = "# New converted content"
        mock_converter_instance = MagicMock(return_value=mock_rendered)
        mock_marker_pdf.PdfConverter.return_value = mock_converter_instance

        import sys

        sys.modules["marker"] = MagicMock()
        sys.modules["marker.converters"] = MagicMock()
        sys.modules["marker.converters.pdf"] = mock_marker_pdf
        sys.modules["marker.models"] = mock_marker_models
        try:
            result = backend.convert(pdf_path, tmp_path / "output", force=True)
            assert result.status == "converted"
        finally:
            for mod in [
                "marker",
                "marker.converters",
                "marker.converters.pdf",
                "marker.models",
            ]:
                sys.modules.pop(mod, None)


# ---------------------------------------------------------------------------
# download/pdf — download_pdf + download_batch
# ---------------------------------------------------------------------------
class TestDownloadPdf:
    """Tests for the PDF download module."""

    @patch("research_pipeline.download.pdf.sha256_file")
    def test_download_pdf_skip_existing(
        self, mock_hash: MagicMock, tmp_path: Path
    ) -> None:
        from research_pipeline.download.pdf import download_pdf

        mock_hash.return_value = "abc123"
        pdf_path = tmp_path / "2301.00001v1.pdf"
        pdf_path.write_bytes(b"%PDF-1.4")

        mock_session = MagicMock()
        mock_rl = MagicMock()

        result = download_pdf(
            arxiv_id="2301.00001",
            version="v1",
            pdf_url="http://example.com/pdf",
            output_dir=tmp_path,
            session=mock_session,
            rate_limiter=mock_rl,
        )
        assert result.status == "skipped_exists"
        mock_session.get.assert_not_called()

    @patch("research_pipeline.download.pdf.sha256_file")
    @patch("research_pipeline.download.pdf._fetch_pdf_bytes")
    def test_download_pdf_success(
        self,
        mock_fetch: MagicMock,
        mock_hash: MagicMock,
        tmp_path: Path,
    ) -> None:
        from research_pipeline.download.pdf import download_pdf

        mock_hash.return_value = "hash123"
        mock_response = MagicMock()
        mock_response.iter_content.return_value = [b"PDF content here"]
        mock_fetch.return_value = mock_response

        mock_session = MagicMock()
        mock_rl = MagicMock()

        result = download_pdf(
            arxiv_id="2301.00002",
            version="v1",
            pdf_url="http://example.com/pdf",
            output_dir=tmp_path,
            session=mock_session,
            rate_limiter=mock_rl,
        )
        assert result.status == "downloaded"
        mock_rl.wait.assert_called_once()

    @patch("research_pipeline.download.pdf._fetch_pdf_bytes")
    def test_download_pdf_failure(
        self,
        mock_fetch: MagicMock,
        tmp_path: Path,
    ) -> None:
        from research_pipeline.download.pdf import download_pdf

        mock_fetch.side_effect = RuntimeError("network error")

        mock_session = MagicMock()
        mock_rl = MagicMock()

        result = download_pdf(
            arxiv_id="2301.00003",
            version="v1",
            pdf_url="http://example.com/pdf",
            output_dir=tmp_path,
            session=mock_session,
            rate_limiter=mock_rl,
        )
        assert result.status == "failed"
        assert "network error" in result.error

    @patch("research_pipeline.download.pdf.download_pdf")
    def test_download_batch_respects_max(
        self,
        mock_dl: MagicMock,
        tmp_path: Path,
    ) -> None:
        from research_pipeline.download.pdf import download_batch

        entry_downloaded = MagicMock(status="downloaded")
        mock_dl.return_value = entry_downloaded

        papers = [
            {
                "arxiv_id": f"230{i}",
                "version": "v1",
                "pdf_url": f"http://example.com/{i}",
            }
            for i in range(10)
        ]

        result = download_batch(
            papers=papers,
            output_dir=tmp_path,
            session=MagicMock(),
            rate_limiter=MagicMock(),
            max_downloads=3,
        )
        assert len(result) == 3

    @patch("research_pipeline.download.pdf.download_pdf")
    def test_download_batch_skips_dont_count(
        self,
        mock_dl: MagicMock,
        tmp_path: Path,
    ) -> None:
        from research_pipeline.download.pdf import download_batch

        call_count = 0

        def side_effect(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                return MagicMock(status="skipped_exists")
            return MagicMock(status="downloaded")

        mock_dl.side_effect = side_effect

        papers = [
            {"arxiv_id": f"230{i}", "version": "v1", "pdf_url": f"http://x.com/{i}"}
            for i in range(5)
        ]

        result = download_batch(
            papers=papers,
            output_dir=tmp_path,
            session=MagicMock(),
            rate_limiter=MagicMock(),
            max_downloads=2,
        )
        # 2 skipped + 2 downloaded = 4 entries, stopped at max_downloads=2
        assert len(result) == 4


# ---------------------------------------------------------------------------
# arxiv/client — ArxivClient
# ---------------------------------------------------------------------------
class TestArxivClient:
    """Tests for the ArxivClient class."""

    def test_init_defaults(self) -> None:
        from research_pipeline.arxiv.client import ArxivClient

        client = ArxivClient()
        assert client.base_url == "https://export.arxiv.org/api/query"
        assert client.max_retries == 4

    @patch("research_pipeline.arxiv.client.parse_total_results")
    @patch("research_pipeline.arxiv.client.parse_atom_response")
    @patch("research_pipeline.arxiv.client.build_api_url")
    @patch("research_pipeline.arxiv.client.canonical_cache_key")
    def test_search_single_page(
        self,
        mock_cache_key: MagicMock,
        mock_build_url: MagicMock,
        mock_parse: MagicMock,
        mock_total: MagicMock,
    ) -> None:
        from research_pipeline.arxiv.client import ArxivClient
        from research_pipeline.models.candidate import CandidateRecord

        mock_build_url.return_value = "http://example.com/api?q=test"
        mock_cache_key.return_value = "cache-key-1"
        cand = CandidateRecord.model_validate(_candidate_dict())
        mock_parse.return_value = [cand]
        mock_total.return_value = 1

        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "<xml>fake</xml>"
        mock_session.get.return_value = mock_response

        mock_rl = MagicMock()

        client = ArxivClient(
            rate_limiter=mock_rl,
            session=mock_session,
            max_retries=0,
        )

        results, raw = client.search("test query", max_results=10)
        assert len(results) == 1
        assert results[0].arxiv_id == "2301.00001"

    @patch("research_pipeline.arxiv.client.build_api_url")
    @patch("research_pipeline.arxiv.client.canonical_cache_key")
    def test_fetch_page_cache_hit(
        self,
        mock_cache_key: MagicMock,
        mock_build_url: MagicMock,
    ) -> None:
        from research_pipeline.arxiv.client import ArxivClient

        mock_cache = MagicMock()
        mock_cache.get.return_value = "<xml>cached</xml>"

        client = ArxivClient(
            cache=mock_cache,
            rate_limiter=MagicMock(),
        )

        result = client._fetch_page("http://example.com", "key1")
        assert result == "<xml>cached</xml>"
        mock_cache.get.assert_called_once_with("key1")

    @patch("research_pipeline.arxiv.client.build_api_url")
    @patch("research_pipeline.arxiv.client.canonical_cache_key")
    @patch("research_pipeline.arxiv.client.time.sleep")
    def test_fetch_page_429_retry(
        self,
        mock_sleep: MagicMock,
        mock_cache_key: MagicMock,
        mock_build_url: MagicMock,
    ) -> None:
        from research_pipeline.arxiv.client import ArxivClient

        mock_session = MagicMock()

        response_429 = MagicMock()
        response_429.status_code = 429
        response_429.headers = {"Retry-After": "2"}

        response_ok = MagicMock()
        response_ok.status_code = 200
        response_ok.text = "<xml>ok</xml>"
        response_ok.raise_for_status = MagicMock()

        mock_session.get.side_effect = [response_429, response_ok]

        client = ArxivClient(
            session=mock_session,
            rate_limiter=MagicMock(),
            max_retries=2,
            backoff_base=1,
        )

        result = client._fetch_page("http://example.com", "key1")
        assert result == "<xml>ok</xml>"
        mock_sleep.assert_called_once_with(2)

    @patch("research_pipeline.arxiv.client.time.sleep")
    def test_fetch_page_timeout_retry(self, mock_sleep: MagicMock) -> None:
        import requests

        from research_pipeline.arxiv.client import ArxivClient

        mock_session = MagicMock()
        mock_session.get.side_effect = requests.ReadTimeout("timeout")

        client = ArxivClient(
            session=mock_session,
            rate_limiter=MagicMock(),
            max_retries=1,
            backoff_base=1,
        )

        with pytest.raises(requests.ReadTimeout):
            client._fetch_page("http://example.com", "key1")

    @patch("research_pipeline.arxiv.client.time.sleep")
    def test_fetch_page_429_exhausted(self, mock_sleep: MagicMock) -> None:
        import requests

        from research_pipeline.arxiv.client import ArxivClient

        mock_session = MagicMock()

        response_429 = MagicMock()
        response_429.status_code = 429
        response_429.headers = {}
        mock_session.get.return_value = response_429

        client = ArxivClient(
            session=mock_session,
            rate_limiter=MagicMock(),
            max_retries=1,
            backoff_base=1,
        )

        with pytest.raises(requests.HTTPError, match="429"):
            client._fetch_page("http://example.com", "key1")

    @patch("research_pipeline.arxiv.client.parse_total_results")
    @patch("research_pipeline.arxiv.client.parse_atom_response")
    @patch("research_pipeline.arxiv.client.build_api_url")
    @patch("research_pipeline.arxiv.client.canonical_cache_key")
    def test_search_saves_raw(
        self,
        mock_cache_key: MagicMock,
        mock_build_url: MagicMock,
        mock_parse: MagicMock,
        mock_total: MagicMock,
        tmp_path: Path,
    ) -> None:
        from research_pipeline.arxiv.client import ArxivClient

        mock_build_url.return_value = "http://example.com/api"
        mock_cache_key.return_value = "key"
        mock_parse.return_value = []
        mock_total.return_value = 0

        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "<xml>data</xml>"
        mock_session.get.return_value = mock_response

        client = ArxivClient(
            session=mock_session,
            rate_limiter=MagicMock(),
            max_retries=0,
        )

        raw_dir = tmp_path / "raw"
        results, raw_paths = client.search("test", max_results=10, save_raw_dir=raw_dir)
        assert len(raw_paths) == 1
        assert Path(raw_paths[0]).exists()


# ---------------------------------------------------------------------------
# extraction/retrieval — retrieve_relevant_chunks
# ---------------------------------------------------------------------------
class TestExtractionRetrieval:
    """Tests for chunk retrieval functions."""

    def test_bm25_rank_basic(self) -> None:
        from research_pipeline.extraction.retrieval import _bm25_rank
        from research_pipeline.models.extraction import ChunkMetadata

        chunks = [
            (
                ChunkMetadata(
                    chunk_id="c1", section="intro", start_line=0, end_line=10
                ),
                "deep learning transformers",
            ),
            (
                ChunkMetadata(
                    chunk_id="c2", section="methods", start_line=11, end_line=20
                ),
                "random forest classification",
            ),
            (
                ChunkMetadata(
                    chunk_id="c3", section="results", start_line=21, end_line=30
                ),
                "transformer attention mechanism",
            ),
        ]
        ranked = _bm25_rank(chunks, ["transformer", "attention"])
        # Chunk c3 should rank highest
        assert ranked[0][0] == 2  # index of c3

    def test_reciprocal_rank_fusion(self) -> None:
        from research_pipeline.extraction.retrieval import _reciprocal_rank_fusion

        ranking_a = [(0, 1.0), (1, 0.8), (2, 0.5)]
        ranking_b = [(2, 1.0), (0, 0.8), (1, 0.5)]
        fused = _reciprocal_rank_fusion([ranking_a, ranking_b], k=60)
        # Both 0 and 2 appear at rank 0 and 1 across rankings
        assert len(fused) == 3

    def test_retrieve_relevant_chunks_empty(self) -> None:
        from research_pipeline.extraction.retrieval import retrieve_relevant_chunks

        result = retrieve_relevant_chunks([], ["query"])
        assert result == []

    def test_retrieve_relevant_chunks_no_query(self) -> None:
        from research_pipeline.extraction.retrieval import retrieve_relevant_chunks
        from research_pipeline.models.extraction import ChunkMetadata

        chunks = [
            (
                ChunkMetadata(
                    chunk_id="c1", section="intro", start_line=0, end_line=10
                ),
                "text",
            ),
        ]
        result = retrieve_relevant_chunks(chunks, [])
        assert result == []

    @patch("research_pipeline.extraction.retrieval._is_embedding_available")
    @patch("research_pipeline.extraction.retrieval._is_cross_encoder_available")
    def test_retrieve_relevant_chunks_bm25_only(
        self,
        mock_cross: MagicMock,
        mock_emb: MagicMock,
    ) -> None:
        from research_pipeline.extraction.retrieval import retrieve_relevant_chunks
        from research_pipeline.models.extraction import ChunkMetadata

        mock_emb.return_value = False
        mock_cross.return_value = False

        chunks = [
            (
                ChunkMetadata(
                    chunk_id="c1", section="intro", start_line=0, end_line=10
                ),
                "deep learning models",
            ),
            (
                ChunkMetadata(
                    chunk_id="c2", section="methods", start_line=11, end_line=20
                ),
                "transformer attention layers",
            ),
        ]
        result = retrieve_relevant_chunks(
            chunks,
            ["transformer", "attention"],
            top_k=2,
            use_embeddings=False,
            use_cross_encoder=False,
        )
        assert len(result) <= 2
        assert all(len(r) == 3 for r in result)

    def test_tokenize(self) -> None:
        from research_pipeline.extraction.retrieval import _tokenize

        result = _tokenize("Hello World Test")
        assert result == ["hello", "world", "test"]


# ---------------------------------------------------------------------------
# cmd_report — report_cmd
# ---------------------------------------------------------------------------
class TestCmdReport:
    """Tests for the report CLI command (extended)."""

    @patch("research_pipeline.cli.cmd_report.render_report_to_file")
    @patch("research_pipeline.cli.cmd_report.load_config")
    def test_report_cmd_success(
        self,
        mock_config: MagicMock,
        mock_render: MagicMock,
        tmp_path: Path,
    ) -> None:
        from research_pipeline.cli.cmd_report import report_cmd

        cfg = _make_config(tmp_path)
        mock_config.return_value = cfg

        # Set up run directory
        run_root = tmp_path / "test-report-001"
        summ_dir = run_root / "summarize"
        summ_dir.mkdir(parents=True)

        synth = {
            "topic": "test",
            "paper_count": 2,
            "common_themes": [],
            "key_findings": [],
            "contradictions": [],
            "gaps": [],
            "methodology_comparison": "",
            "synthesis_narrative": "narrative",
        }
        (summ_dir / "synthesis_report.json").write_text(json.dumps(synth))

        with patch(
            "research_pipeline.cli.cmd_report.list_templates",
            return_value=["survey"],
        ):
            report_cmd(
                run_id="test-report-001",
                template="survey",
                custom_template="",
                output="",
            )
        mock_render.assert_called_once()

    @patch("research_pipeline.cli.cmd_report.load_config")
    def test_report_cmd_unknown_template(
        self,
        mock_config: MagicMock,
        tmp_path: Path,
    ) -> None:
        from research_pipeline.cli.cmd_report import report_cmd

        mock_config.return_value = _make_config(tmp_path)

        with (
            patch(
                "research_pipeline.cli.cmd_report.list_templates",
                return_value=["survey"],
            ),
            pytest.raises(click.exceptions.Exit),
        ):
            report_cmd(
                run_id="test-report-002",
                template="nonexistent",
                custom_template="",
                output="",
            )

    @patch("research_pipeline.cli.cmd_report.load_config")
    def test_report_cmd_no_synthesis_json(
        self,
        mock_config: MagicMock,
        tmp_path: Path,
    ) -> None:
        from research_pipeline.cli.cmd_report import report_cmd

        mock_config.return_value = _make_config(tmp_path)

        with (
            patch(
                "research_pipeline.cli.cmd_report.list_templates",
                return_value=["survey"],
            ),
            pytest.raises(click.exceptions.Exit),
        ):
            report_cmd(
                run_id="test-report-003",
                template="survey",
                custom_template="",
                output="",
            )


# ---------------------------------------------------------------------------
# cmd_aggregate — aggregate_cmd
# ---------------------------------------------------------------------------
class TestCmdAggregate:
    """Tests for the aggregate CLI command (extended)."""

    @patch("research_pipeline.cli.cmd_aggregate.format_aggregation_text")
    @patch("research_pipeline.cli.cmd_aggregate.aggregate_evidence")
    @patch("research_pipeline.cli.cmd_aggregate.load_config")
    def test_aggregate_cmd_success(
        self,
        mock_config: MagicMock,
        mock_aggregate: MagicMock,
        mock_format: MagicMock,
        tmp_path: Path,
    ) -> None:
        from research_pipeline.cli.cmd_aggregate import aggregate_cmd

        cfg = _make_config(tmp_path)
        mock_config.return_value = cfg

        # Set up run directory with synthesis
        run_root = tmp_path / "test-agg-001"
        summ_dir = run_root / "summarize"
        summ_dir.mkdir(parents=True)

        synth = {
            "topic": "test",
            "paper_count": 2,
            "common_themes": [],
            "key_findings": [],
            "contradictions": [],
            "gaps": [],
            "methodology_comparison": "",
            "synthesis_narrative": "narrative",
        }
        (summ_dir / "synthesis.json").write_text(json.dumps(synth))

        mock_result = MagicMock()
        mock_result.model_dump_json.return_value = '{"stats": {}}'
        mock_result.stats.input_statements = 10
        mock_result.stats.output_statements = 5
        mock_aggregate.return_value = mock_result
        mock_format.return_value = "Formatted output"

        aggregate_cmd(
            run_id="test-agg-001",
            config_path=str(tmp_path / "config.toml"),
        )
        mock_aggregate.assert_called_once()

    @patch("research_pipeline.cli.cmd_aggregate.load_config")
    def test_aggregate_cmd_no_synthesis(
        self,
        mock_config: MagicMock,
        tmp_path: Path,
    ) -> None:
        from research_pipeline.cli.cmd_aggregate import aggregate_cmd

        cfg = _make_config(tmp_path)
        mock_config.return_value = cfg

        # No synthesis file → should exit
        with pytest.raises(click.exceptions.Exit):
            aggregate_cmd(
                run_id="test-agg-missing",
                config_path=str(tmp_path / "config.toml"),
            )

    @patch("research_pipeline.cli.cmd_aggregate.format_aggregation_text")
    @patch("research_pipeline.cli.cmd_aggregate.aggregate_evidence")
    @patch("research_pipeline.cli.cmd_aggregate.load_config")
    def test_aggregate_cmd_json_output(
        self,
        mock_config: MagicMock,
        mock_aggregate: MagicMock,
        mock_format: MagicMock,
        tmp_path: Path,
    ) -> None:
        from research_pipeline.cli.cmd_aggregate import aggregate_cmd

        cfg = _make_config(tmp_path)
        mock_config.return_value = cfg

        run_root = tmp_path / "test-agg-json"
        summ_dir = run_root / "summarize"
        summ_dir.mkdir(parents=True)

        synth = {
            "topic": "test",
            "paper_count": 1,
            "common_themes": [],
            "key_findings": [],
            "contradictions": [],
            "gaps": [],
            "methodology_comparison": "",
            "synthesis_narrative": "n",
        }
        (summ_dir / "synthesis.json").write_text(json.dumps(synth))

        mock_result = MagicMock()
        mock_result.model_dump_json.return_value = '{"data": "json"}'
        mock_result.stats.input_statements = 5
        mock_result.stats.output_statements = 3
        mock_aggregate.return_value = mock_result

        aggregate_cmd(
            run_id="test-agg-json",
            output_format="json",
            config_path=str(tmp_path / "config.toml"),
        )


# ---------------------------------------------------------------------------
# cmd_download — run_download
# ---------------------------------------------------------------------------
class TestCmdDownload:
    """Tests for the download CLI handler (extended)."""

    @patch("research_pipeline.cli.cmd_download.load_config")
    def test_run_download_no_shortlist_exits(
        self,
        mock_config: MagicMock,
        tmp_path: Path,
    ) -> None:
        from research_pipeline.cli.cmd_download import run_download

        mock_config.return_value = _make_config(tmp_path)

        with pytest.raises(click.exceptions.Exit):
            run_download(workspace=tmp_path, run_id="test-dl-no-shortlist")

    @patch("research_pipeline.cli.cmd_download.download_batch")
    @patch("research_pipeline.cli.cmd_download.create_session")
    @patch("research_pipeline.cli.cmd_download.load_config")
    def test_run_download_with_shortlist(
        self,
        mock_config: MagicMock,
        mock_session: MagicMock,
        mock_batch: MagicMock,
        tmp_path: Path,
    ) -> None:
        from research_pipeline.cli.cmd_download import run_download

        cfg = _make_config(tmp_path)
        mock_config.return_value = cfg
        mock_session.return_value = MagicMock()

        # Create shortlist
        run_root = tmp_path / "test-dl-001"
        screen_dir = run_root / "screen"
        screen_dir.mkdir(parents=True)

        shortlist_data = [
            {
                "paper": _candidate_dict(),
                "download": True,
                "decision": "accept",
                "score": 0.9,
                "cheap_score": {"blended_score": 0.9},
            }
        ]
        (screen_dir / "shortlist.json").write_text(json.dumps(shortlist_data))

        dl_entry = MagicMock(
            status="downloaded",
            model_dump=lambda mode="json": _dl_entry_dict(),
        )
        mock_batch.return_value = [dl_entry]

        run_download(workspace=tmp_path, run_id="test-dl-001")
        mock_batch.assert_called_once()


# ---------------------------------------------------------------------------
# app.py — _version_callback, _common_options, plan/search/screen/etc wrappers
# ---------------------------------------------------------------------------
class TestAppWrappers:
    """Tests for app.py thin wrappers that dispatch to handler modules."""

    def test_version_callback_true(self) -> None:
        import typer

        from research_pipeline.cli.app import _version_callback

        with pytest.raises(typer.Exit):
            _version_callback(True)

    def test_version_callback_false(self) -> None:
        from research_pipeline.cli.app import _version_callback

        # Should not raise
        _version_callback(False)

    def test_common_options(self, tmp_path: Path) -> None:
        from research_pipeline.cli.app import _common_options

        opts = _common_options(
            verbose=True,
            config=tmp_path / "config.toml",
            workspace=tmp_path,
            run_id="test-run",
        )
        assert opts["run_id"] == "test-run"
        assert opts["workspace"] == tmp_path
        assert opts["config_path"] == tmp_path / "config.toml"

    def test_common_options_defaults(self) -> None:
        from research_pipeline.cli.app import _common_options

        opts = _common_options(verbose=False)
        assert opts["run_id"] is None
        assert opts["workspace"] is None
        assert opts["config_path"] is None

    @patch("research_pipeline.cli.cmd_plan.run_plan")
    def test_app_plan_dispatches(self, mock_run_plan: MagicMock) -> None:
        from research_pipeline.cli.app import plan

        plan(
            topic="test topic",
            verbose=False,
            config=None,
            workspace=None,
            run_id="test-001",
        )
        mock_run_plan.assert_called_once()

    @patch("research_pipeline.cli.cmd_search.run_search")
    def test_app_search_dispatches(self, mock_run_search: MagicMock) -> None:
        from research_pipeline.cli.app import search

        search(
            topic="test",
            verbose=False,
            config=None,
            workspace=None,
            run_id="test-001",
            resume=False,
            source=None,
        )
        mock_run_search.assert_called_once()

    @patch("research_pipeline.cli.cmd_screen.run_screen")
    def test_app_screen_dispatches(self, mock_run_screen: MagicMock) -> None:
        from research_pipeline.cli.app import screen

        screen(
            verbose=False,
            config=None,
            workspace=None,
            run_id="test-001",
            resume=False,
            diversity=None,
            diversity_lambda=None,
        )
        mock_run_screen.assert_called_once()

    @patch("research_pipeline.cli.cmd_download.run_download")
    def test_app_download_dispatches(self, mock_run: MagicMock) -> None:
        from research_pipeline.cli.app import download

        download(
            verbose=False,
            config=None,
            workspace=None,
            run_id="test-001",
            force=False,
        )
        mock_run.assert_called_once()

    @patch("research_pipeline.cli.cmd_convert.run_convert")
    def test_app_convert_dispatches(self, mock_run: MagicMock) -> None:
        from research_pipeline.cli.app import convert

        convert(
            verbose=False,
            config=None,
            workspace=None,
            run_id="test-001",
            force=False,
            backend=None,
        )
        mock_run.assert_called_once()

    @patch("research_pipeline.cli.cmd_extract.run_extract")
    def test_app_extract_dispatches(self, mock_run: MagicMock) -> None:
        from research_pipeline.cli.app import extract

        extract(
            verbose=False,
            config=None,
            workspace=None,
            run_id="test-001",
            cross_encoder=None,
        )
        mock_run.assert_called_once()

    @patch("research_pipeline.cli.cmd_summarize.run_summarize")
    def test_app_summarize_dispatches(self, mock_run: MagicMock) -> None:
        from research_pipeline.cli.app import summarize

        summarize(
            verbose=False,
            config=None,
            workspace=None,
            run_id="test-001",
            output_format="markdown",
        )
        mock_run.assert_called_once()

    @patch("research_pipeline.cli.cmd_run.run_full")
    def test_app_run_dispatches(self, mock_run: MagicMock) -> None:
        from research_pipeline.cli.app import run

        run(
            topic="test topic",
            verbose=False,
            config=None,
            workspace=None,
            run_id=None,
            resume=False,
            source=None,
            profile="standard",
            ter_iterations=3,
            auto_approve=True,
        )
        mock_run.assert_called_once()

    @patch("research_pipeline.cli.cmd_inspect.run_inspect")
    def test_app_inspect_dispatches(self, mock_run: MagicMock) -> None:
        from research_pipeline.cli.app import inspect

        inspect(
            verbose=False,
            workspace=None,
            run_id=None,
        )
        mock_run.assert_called_once()

    @patch("research_pipeline.cli.cmd_quality.run_quality")
    def test_app_quality_dispatches(self, mock_run: MagicMock) -> None:
        from research_pipeline.cli.app import quality

        quality(
            verbose=False,
            config=None,
            workspace=None,
            run_id="test-001",
        )
        mock_run.assert_called_once()

    @patch("research_pipeline.cli.cmd_convert_file.run_convert_file")
    def test_app_convert_file_dispatches(
        self, mock_run: MagicMock, tmp_path: Path
    ) -> None:
        from research_pipeline.cli.app import convert_file

        convert_file(
            pdf_path=tmp_path / "test.pdf",
            output_dir=None,
            backend=None,
            config=None,
        )
        mock_run.assert_called_once()

    @patch("research_pipeline.cli.cmd_expand.run_expand")
    def test_app_expand_dispatches(self, mock_run: MagicMock) -> None:
        from research_pipeline.cli.app import expand

        expand(
            verbose=False,
            config=None,
            workspace=None,
            run_id="test-001",
            paper_ids="2301.00001,2301.00002",
            direction="both",
            limit=50,
            reference_boost=1.0,
            bfs_depth=0,
            bfs_top_k=10,
            bfs_query="",
            snowball=False,
            snowball_max_rounds=5,
            snowball_max_papers=200,
            snowball_decay_threshold=0.10,
            snowball_decay_patience=2,
            bfs_budget=0,
            bfs_min_new=0,
        )
        mock_run.assert_called_once()

    @patch("research_pipeline.cli.cmd_convert_rough.run_convert_rough")
    def test_app_convert_rough_dispatches(self, mock_run: MagicMock) -> None:
        from research_pipeline.cli.app import convert_rough

        convert_rough(
            verbose=False,
            config=None,
            workspace=None,
            run_id="test-001",
            force=False,
        )
        mock_run.assert_called_once()

    @patch("research_pipeline.cli.cmd_convert_fine.run_convert_fine")
    def test_app_convert_fine_dispatches(self, mock_run: MagicMock) -> None:
        from research_pipeline.cli.app import convert_fine

        convert_fine(
            verbose=False,
            config=None,
            workspace=None,
            run_id="test-001",
            paper_ids="2301.00001",
            force=False,
            backend=None,
        )
        mock_run.assert_called_once()

    @patch("research_pipeline.cli.cmd_index.run_index")
    def test_app_index_dispatches(self, mock_run: MagicMock) -> None:
        from research_pipeline.cli.app import index

        index(
            verbose=False,
            list_papers=True,
            gc=False,
            search=None,
            search_limit=50,
            db_path=None,
        )
        mock_run.assert_called_once()

    @patch("research_pipeline.cli.cmd_analyze.run_analyze")
    def test_app_analyze_dispatches(self, mock_run: MagicMock) -> None:
        from research_pipeline.cli.app import analyze

        analyze(
            verbose=False,
            config=None,
            workspace=None,
            run_id="test-001",
            collect=False,
            paper_ids=None,
        )
        mock_run.assert_called_once()

    @patch("research_pipeline.cli.cmd_validate.run_validate")
    def test_app_validate_dispatches(self, mock_run: MagicMock) -> None:
        from research_pipeline.cli.app import validate

        validate(
            report=None,
            verbose=False,
            config=None,
            workspace=None,
            run_id="test-001",
            output=None,
        )
        mock_run.assert_called_once()

    @patch("research_pipeline.cli.cmd_compare.run_compare")
    def test_app_compare_dispatches(self, mock_run: MagicMock) -> None:
        from research_pipeline.cli.app import compare

        compare(
            run_a="run1",
            run_b="run2",
            verbose=False,
            config=None,
            workspace=None,
            output=None,
        )
        mock_run.assert_called_once()
