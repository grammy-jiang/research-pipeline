"""Tests for structural verification of pipeline stage outputs."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from mcp_server.workflow.verification import (
    VerificationResult,
    verify_analyze,
    verify_convert,
    verify_download,
    verify_extract,
    verify_plan,
    verify_screen,
    verify_search,
    verify_synthesize,
)


class TestVerificationResult:
    """VerificationResult model tests."""

    def test_pass(self) -> None:
        vr = VerificationResult(True, "All good", {"check1": True})
        assert vr.passed
        assert "PASS" in repr(vr)

    def test_fail(self) -> None:
        vr = VerificationResult(False, "Missing file", {"check1": False})
        assert not vr.passed
        assert "FAIL" in repr(vr)


class TestVerifyPlan:
    """Plan stage verification tests."""

    def test_valid_plan(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            plan_dir = Path(tmpdir) / "test-run" / "plan"
            plan_dir.mkdir(parents=True)
            (plan_dir / "query_plan.json").write_text(
                json.dumps(
                    {
                        "must_terms": ["harness", "engineering"],
                        "nice_terms": ["AI", "agents"],
                        "query_variants": ["v1", "v2", "v3"],
                    }
                )
            )
            vr = verify_plan(tmpdir, "test-run")
            assert vr.passed
            assert vr.checks["has_must_terms"]
            assert vr.checks["has_query_variants"]

    def test_missing_plan(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            vr = verify_plan(tmpdir, "no-run")
            assert not vr.passed

    def test_plan_no_must_terms(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            plan_dir = Path(tmpdir) / "test-run" / "plan"
            plan_dir.mkdir(parents=True)
            (plan_dir / "query_plan.json").write_text(
                json.dumps(
                    {
                        "must_terms": [],
                        "query_variants": ["v1", "v2"],
                    }
                )
            )
            vr = verify_plan(tmpdir, "test-run")
            assert not vr.passed
            assert not vr.checks["has_must_terms"]

    def test_plan_insufficient_variants(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            plan_dir = Path(tmpdir) / "test-run" / "plan"
            plan_dir.mkdir(parents=True)
            (plan_dir / "query_plan.json").write_text(
                json.dumps(
                    {
                        "must_terms": ["test"],
                        "query_variants": ["v1"],
                    }
                )
            )
            vr = verify_plan(tmpdir, "test-run")
            assert not vr.passed
            assert not vr.checks["has_query_variants"]


class TestVerifySearch:
    """Search stage verification tests."""

    def test_valid_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            search_dir = Path(tmpdir) / "test-run" / "search"
            search_dir.mkdir(parents=True)
            candidates = [
                {"paper_id": "2301.00001", "title": "Paper A", "abstract": "..."},
                {"paper_id": "2301.00002", "title": "Paper B", "abstract": "..."},
            ]
            (search_dir / "candidates.jsonl").write_text(
                "\n".join(json.dumps(c) for c in candidates)
            )
            vr = verify_search(tmpdir, "test-run")
            assert vr.passed

    def test_missing_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            vr = verify_search(tmpdir, "no-run")
            assert not vr.passed

    def test_candidates_missing_paper_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            search_dir = Path(tmpdir) / "test-run" / "search"
            search_dir.mkdir(parents=True)
            (search_dir / "candidates.jsonl").write_text(json.dumps({"title": "No ID"}))
            vr = verify_search(tmpdir, "test-run")
            assert not vr.passed
            assert not vr.checks["all_have_paper_id"]


class TestVerifyScreen:
    """Screen stage verification tests."""

    def test_valid_shortlist(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            screen_dir = Path(tmpdir) / "test-run" / "screen"
            screen_dir.mkdir(parents=True)
            (screen_dir / "shortlist.json").write_text(
                json.dumps([{"paper_id": "2301.00001", "score": 0.9}])
            )
            vr = verify_screen(tmpdir, "test-run")
            assert vr.passed

    def test_empty_shortlist(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            screen_dir = Path(tmpdir) / "test-run" / "screen"
            screen_dir.mkdir(parents=True)
            (screen_dir / "shortlist.json").write_text(json.dumps([]))
            vr = verify_screen(tmpdir, "test-run")
            assert not vr.passed


class TestVerifyDownload:
    """Download stage verification tests."""

    def test_valid_pdfs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            pdf_dir = Path(tmpdir) / "test-run" / "download" / "pdf"
            pdf_dir.mkdir(parents=True)
            # Create a 10KB+ file
            (pdf_dir / "paper1.pdf").write_bytes(b"x" * 15_000)
            vr = verify_download(tmpdir, "test-run")
            assert vr.passed

    def test_trivial_pdf(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            pdf_dir = Path(tmpdir) / "test-run" / "download" / "pdf"
            pdf_dir.mkdir(parents=True)
            (pdf_dir / "tiny.pdf").write_bytes(b"x" * 100)
            vr = verify_download(tmpdir, "test-run")
            assert not vr.passed
            assert not vr.checks["all_non_trivial"]


class TestVerifyConvert:
    """Convert stage verification tests."""

    def test_valid_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            md_dir = Path(tmpdir) / "test-run" / "convert" / "markdown"
            md_dir.mkdir(parents=True)
            (md_dir / "paper1.md").write_text("# Paper\n" + "x" * 600)
            vr = verify_convert(tmpdir, "test-run")
            assert vr.passed

    def test_no_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            vr = verify_convert(tmpdir, "test-run")
            assert not vr.passed


class TestVerifyExtract:
    """Extract stage verification tests."""

    def test_valid_extraction(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            ext_dir = Path(tmpdir) / "test-run" / "extract"
            ext_dir.mkdir(parents=True)
            (ext_dir / "chunks.json").write_text("[]")
            vr = verify_extract(tmpdir, "test-run")
            assert vr.passed

    def test_missing_extraction(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            vr = verify_extract(tmpdir, "no-run")
            assert not vr.passed


class TestVerifyAnalyze:
    """In-memory analysis verification tests."""

    def test_valid_analyses(self) -> None:
        analyses = [
            {"rating": 4, "findings": ["Good methodology"], "methodology": "RCT"},
            {"rating": 3, "findings": ["Interesting results"], "methodology": "survey"},
        ]
        vr = verify_analyze(analyses, expected_count=2)
        assert vr.passed

    def test_empty_analyses(self) -> None:
        vr = verify_analyze([], expected_count=5)
        assert not vr.passed

    def test_invalid_rating(self) -> None:
        analyses = [{"rating": 10, "findings": ["x"], "methodology": "y"}]
        vr = verify_analyze(analyses, expected_count=1)
        assert not vr.passed
        assert not vr.checks["all_have_rating"]

    def test_missing_findings(self) -> None:
        analyses = [{"rating": 3, "findings": [], "methodology": "y"}]
        vr = verify_analyze(analyses, expected_count=1)
        assert not vr.passed


class TestVerifySynthesize:
    """Synthesis verification tests."""

    def test_valid_synthesis(self) -> None:
        synthesis = {
            "themes": ["Theme A", "Theme B"],
            "readiness_verdict": "HAS_GAPS",
            "gaps": [{"description": "Gap 1", "type": "ACADEMIC"}],
        }
        vr = verify_synthesize(synthesis)
        assert vr.passed

    def test_no_themes(self) -> None:
        synthesis = {"themes": [], "readiness_verdict": "INSUFFICIENT"}
        vr = verify_synthesize(synthesis)
        assert not vr.passed

    def test_no_verdict(self) -> None:
        synthesis = {"themes": ["A"], "readiness_verdict": ""}
        vr = verify_synthesize(synthesis)
        assert not vr.passed
