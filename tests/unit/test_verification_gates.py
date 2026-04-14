"""Tests for verification gates (pipeline/orchestrator.py)."""

import json
from pathlib import Path

import pytest

from research_pipeline.pipeline.orchestrator import (
    _verify_convert,
    _verify_download,
    _verify_extract,
    _verify_plan,
    _verify_screen,
    _verify_search,
    _verify_summarize,
    verify_stage,
)


@pytest.fixture()
def run_root(tmp_path: Path) -> Path:
    """Create a minimal run root."""
    return tmp_path


class TestVerifyPlan:
    def test_valid_plan(self, run_root: Path) -> None:
        plan_dir = run_root / "plan"
        plan_dir.mkdir()
        (plan_dir / "query_plan.json").write_text(
            json.dumps(
                {
                    "must_terms": ["memory", "AI"],
                    "query_variants": ["v1", "v2", "v3"],
                }
            )
        )
        assert _verify_plan(run_root) == []

    def test_missing_plan_file(self, run_root: Path) -> None:
        (run_root / "plan").mkdir()
        errors = _verify_plan(run_root)
        assert len(errors) == 1
        assert "not found" in errors[0]

    def test_no_must_terms(self, run_root: Path) -> None:
        plan_dir = run_root / "plan"
        plan_dir.mkdir()
        (plan_dir / "query_plan.json").write_text(
            json.dumps({"must_terms": [], "query_variants": ["v1", "v2"]})
        )
        errors = _verify_plan(run_root)
        assert any("must_terms" in e for e in errors)

    def test_too_few_variants(self, run_root: Path) -> None:
        plan_dir = run_root / "plan"
        plan_dir.mkdir()
        (plan_dir / "query_plan.json").write_text(
            json.dumps({"must_terms": ["AI"], "query_variants": ["v1"]})
        )
        errors = _verify_plan(run_root)
        assert any("variants" in e for e in errors)


class TestVerifySearch:
    def test_valid_search(self, run_root: Path) -> None:
        search_dir = run_root / "search"
        search_dir.mkdir()
        lines = [
            json.dumps({"arxiv_id": "2401.00001", "title": "Paper A"}),
            json.dumps({"arxiv_id": "2401.00002", "title": "Paper B"}),
        ]
        (search_dir / "candidates.jsonl").write_text("\n".join(lines))
        assert _verify_search(run_root) == []

    def test_missing_candidates(self, run_root: Path) -> None:
        (run_root / "search").mkdir()
        errors = _verify_search(run_root)
        assert any("not found" in e for e in errors)

    def test_empty_candidates(self, run_root: Path) -> None:
        search_dir = run_root / "search"
        search_dir.mkdir()
        (search_dir / "candidates.jsonl").write_text("")
        errors = _verify_search(run_root)
        assert any("empty" in e for e in errors)


class TestVerifyScreen:
    def test_valid_screen(self, run_root: Path) -> None:
        screen_dir = run_root / "screen"
        screen_dir.mkdir()
        (screen_dir / "shortlist.json").write_text(
            json.dumps([{"paper": {"arxiv_id": "2401.00001"}}])
        )
        assert _verify_screen(run_root) == []

    def test_empty_shortlist(self, run_root: Path) -> None:
        screen_dir = run_root / "screen"
        screen_dir.mkdir()
        (screen_dir / "shortlist.json").write_text("[]")
        errors = _verify_screen(run_root)
        assert any("empty" in e for e in errors)


class TestVerifyDownload:
    def test_valid_download(self, run_root: Path) -> None:
        pdf_dir = run_root / "download" / "pdf"
        pdf_dir.mkdir(parents=True)
        (pdf_dir / "paper.pdf").write_bytes(b"x" * 20_000)
        assert _verify_download(run_root) == []

    def test_no_pdfs(self, run_root: Path) -> None:
        pdf_dir = run_root / "download" / "pdf"
        pdf_dir.mkdir(parents=True)
        errors = _verify_download(run_root)
        assert any("No PDF" in e for e in errors)

    def test_small_pdf_warning(self, run_root: Path) -> None:
        pdf_dir = run_root / "download" / "pdf"
        pdf_dir.mkdir(parents=True)
        (pdf_dir / "good.pdf").write_bytes(b"x" * 20_000)
        (pdf_dir / "bad.pdf").write_bytes(b"x" * 100)
        errors = _verify_download(run_root)
        assert any("smaller than 10KB" in e for e in errors)


class TestVerifyConvert:
    def test_valid_convert(self, run_root: Path) -> None:
        md_dir = run_root / "convert" / "markdown"
        md_dir.mkdir(parents=True)
        (md_dir / "paper.md").write_text("x" * 1000)
        assert _verify_convert(run_root) == []

    def test_no_markdown(self, run_root: Path) -> None:
        md_dir = run_root / "convert" / "markdown"
        md_dir.mkdir(parents=True)
        errors = _verify_convert(run_root)
        assert any("No Markdown" in e for e in errors)

    def test_empty_markdown(self, run_root: Path) -> None:
        md_dir = run_root / "convert" / "markdown"
        md_dir.mkdir(parents=True)
        (md_dir / "good.md").write_text("x" * 1000)
        (md_dir / "bad.md").write_text("tiny")
        errors = _verify_convert(run_root)
        assert any("under 500 bytes" in e for e in errors)


class TestVerifyExtract:
    def test_valid_extract(self, run_root: Path) -> None:
        extract_dir = run_root / "extract"
        extract_dir.mkdir()
        (extract_dir / "paper.extract.json").write_text("{}")
        assert _verify_extract(run_root) == []

    def test_no_extractions(self, run_root: Path) -> None:
        (run_root / "extract").mkdir()
        errors = _verify_extract(run_root)
        assert any("No extraction" in e for e in errors)


class TestVerifySummarize:
    def test_valid_summarize(self, run_root: Path) -> None:
        sum_dir = run_root / "summarize"
        sum_dir.mkdir()
        (sum_dir / "synthesis.json").write_text("{}")
        assert _verify_summarize(run_root) == []

    def test_no_synthesis(self, run_root: Path) -> None:
        (run_root / "summarize").mkdir()
        errors = _verify_summarize(run_root)
        assert any("No synthesis" in e for e in errors)


class TestVerifyStageDispatch:
    def test_dispatches_to_correct_verifier(self, run_root: Path) -> None:
        (run_root / "plan").mkdir()
        errors = verify_stage(run_root, "plan")
        # Should have errors since no query_plan.json
        assert len(errors) > 0

    def test_unknown_stage_returns_empty(self, run_root: Path) -> None:
        assert verify_stage(run_root, "nonexistent_stage") == []
