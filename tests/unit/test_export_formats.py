"""Tests for synthesis report export formats (JSON and BibTeX)."""

import json
from pathlib import Path

import pytest

from research_pipeline.models.summary import (
    PaperSummary,
    SummaryEvidence,
    SynthesisAgreement,
    SynthesisDisagreement,
    SynthesisReport,
)
from research_pipeline.summarization.export import (
    _extract_year_from_arxiv_id,
    _sanitize_bibtex,
    export_bibtex,
    export_json,
    export_report,
)


def _make_summary(
    arxiv_id: str = "2301.12345",
    version: str = "v1",
    title: str = "Test Paper",
) -> PaperSummary:
    return PaperSummary(
        arxiv_id=arxiv_id,
        version=version,
        title=title,
        objective="Test objective",
        methodology="Test methodology",
        findings=["finding1"],
        limitations=["limitation1"],
        evidence=[
            SummaryEvidence(chunk_id="c1", line_range="1-5", quote="quote1"),
        ],
        uncertainties=["uncertainty1"],
    )


def _make_report(
    paper_summaries: list[PaperSummary] | None = None,
) -> SynthesisReport:
    summaries = paper_summaries or [_make_summary()]
    return SynthesisReport(
        topic="Test Topic",
        paper_count=len(summaries),
        agreements=[
            SynthesisAgreement(
                claim="Claim A",
                supporting_papers=["2301.12345"],
                evidence=[
                    SummaryEvidence(chunk_id="c1", line_range="1-5", quote="q"),
                ],
            ),
        ],
        disagreements=[
            SynthesisDisagreement(
                topic="Topic X",
                positions={"2301.12345": "position A"},
                evidence=[],
            ),
        ],
        open_questions=["OQ1"],
        paper_summaries=summaries,
    )


def _make_empty_report() -> SynthesisReport:
    return SynthesisReport(
        topic="Empty Topic",
        paper_count=0,
    )


# --- JSON export tests ---


class TestExportJson:
    def test_produces_valid_json(self, tmp_path: Path) -> None:
        out = tmp_path / "report.json"
        export_json(_make_report(), out)
        data = json.loads(out.read_text(encoding="utf-8"))
        assert isinstance(data, dict)

    def test_includes_metadata_fields(self, tmp_path: Path) -> None:
        out = tmp_path / "report.json"
        export_json(_make_report(), out)
        data = json.loads(out.read_text(encoding="utf-8"))
        assert "metadata" in data
        assert "export_timestamp" in data["metadata"]
        assert "pipeline_version" in data["metadata"]

    def test_includes_report_body(self, tmp_path: Path) -> None:
        report = _make_report()
        out = tmp_path / "report.json"
        export_json(report, out)
        data = json.loads(out.read_text(encoding="utf-8"))
        assert data["report"]["topic"] == "Test Topic"
        assert data["report"]["paper_count"] == 1

    def test_includes_evidence_map(self, tmp_path: Path) -> None:
        out = tmp_path / "report.json"
        export_json(_make_report(), out)
        data = json.loads(out.read_text(encoding="utf-8"))
        assert "evidence_map" in data
        emap = data["evidence_map"]
        assert len(emap) >= 1
        assert emap[0]["paper_id"] == "2301.12345"

    def test_evidence_map_contains_agreement_and_disagreement(
        self, tmp_path: Path
    ) -> None:
        out = tmp_path / "report.json"
        export_json(_make_report(), out)
        data = json.loads(out.read_text(encoding="utf-8"))
        types = {e["type"] for e in data["evidence_map"]}
        assert "agreement" in types
        assert "disagreement" in types

    def test_empty_report_json(self, tmp_path: Path) -> None:
        out = tmp_path / "empty.json"
        export_json(_make_empty_report(), out)
        data = json.loads(out.read_text(encoding="utf-8"))
        assert data["report"]["paper_count"] == 0
        assert data["evidence_map"] == []

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        out = tmp_path / "nested" / "dir" / "report.json"
        export_json(_make_report(), out)
        assert out.exists()

    def test_roundtrip_fields(self, tmp_path: Path) -> None:
        """SynthesisReport → JSON → load → verify all top-level fields."""
        report = _make_report()
        out = tmp_path / "roundtrip.json"
        export_json(report, out)
        data = json.loads(out.read_text(encoding="utf-8"))
        r = data["report"]
        assert r["topic"] == report.topic
        assert r["paper_count"] == report.paper_count
        assert len(r["agreements"]) == len(report.agreements)
        assert len(r["disagreements"]) == len(report.disagreements)
        assert r["open_questions"] == report.open_questions
        assert len(r["paper_summaries"]) == len(report.paper_summaries)
        ps = r["paper_summaries"][0]
        assert ps["arxiv_id"] == "2301.12345"
        assert ps["title"] == "Test Paper"


# --- BibTeX export tests ---


class TestExportBibtex:
    def test_produces_article_for_arxiv_id(self, tmp_path: Path) -> None:
        out = tmp_path / "refs.bib"
        export_bibtex(_make_report(), out)
        content = out.read_text(encoding="utf-8")
        assert "@article{2301.12345," in content

    def test_includes_required_fields(self, tmp_path: Path) -> None:
        out = tmp_path / "refs.bib"
        export_bibtex(_make_report(), out)
        content = out.read_text(encoding="utf-8")
        assert "title = {Test Paper}" in content
        assert "year = {2023}" in content
        assert "eprint = {2301.12345}" in content
        assert "archivePrefix = {arXiv}" in content

    def test_non_arxiv_id_uses_misc(self, tmp_path: Path) -> None:
        summary = _make_summary(arxiv_id="DOI:10.1234/test", title="Non-arXiv")
        report = _make_report(paper_summaries=[summary])
        out = tmp_path / "refs.bib"
        export_bibtex(report, out)
        content = out.read_text(encoding="utf-8")
        assert "@misc{" in content
        assert "archivePrefix" not in content

    def test_empty_report_bibtex(self, tmp_path: Path) -> None:
        out = tmp_path / "empty.bib"
        export_bibtex(_make_empty_report(), out)
        content = out.read_text(encoding="utf-8")
        assert content.strip() == ""

    def test_multiple_papers(self, tmp_path: Path) -> None:
        summaries = [
            _make_summary(arxiv_id="2301.12345", title="Paper A"),
            _make_summary(arxiv_id="2312.54321", title="Paper B"),
        ]
        report = _make_report(paper_summaries=summaries)
        out = tmp_path / "refs.bib"
        export_bibtex(report, out)
        content = out.read_text(encoding="utf-8")
        assert "@article{2301.12345," in content
        assert "@article{2312.54321," in content

    def test_special_chars_escaped(self, tmp_path: Path) -> None:
        summary = _make_summary(title="Results & Analysis: 100% Accurate")
        report = _make_report(paper_summaries=[summary])
        out = tmp_path / "refs.bib"
        export_bibtex(report, out)
        content = out.read_text(encoding="utf-8")
        assert r"Results \& Analysis: 100\% Accurate" in content


# --- Dispatcher tests ---


class TestExportReport:
    def test_dispatch_json(self, tmp_path: Path) -> None:
        out = tmp_path / "report.json"
        export_report(_make_report(), out, fmt="json")
        data = json.loads(out.read_text(encoding="utf-8"))
        assert "report" in data

    def test_dispatch_bibtex(self, tmp_path: Path) -> None:
        out = tmp_path / "refs.bib"
        export_report(_make_report(), out, fmt="bibtex")
        content = out.read_text(encoding="utf-8")
        assert "@article{" in content

    def test_unknown_format_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="Unknown export format"):
            export_report(_make_report(), tmp_path / "out.xml", fmt="xml")


# --- Helper tests ---


class TestHelpers:
    @pytest.mark.parametrize(
        ("arxiv_id", "expected"),
        [
            ("2301.12345", "2023"),
            ("9901.00001", "1999"),
            ("0704.0001", "2007"),
            ("not-an-id", "unknown"),
            ("", "unknown"),
        ],
    )
    def test_extract_year(self, arxiv_id: str, expected: str) -> None:
        assert _extract_year_from_arxiv_id(arxiv_id) == expected

    def test_sanitize_bibtex_escapes(self) -> None:
        assert _sanitize_bibtex("A & B") == r"A \& B"
        assert _sanitize_bibtex("100%") == r"100\%"
        assert _sanitize_bibtex("no specials") == "no specials"
