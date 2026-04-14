"""Tests for the validate command (cmd_validate)."""

from pathlib import Path

from research_pipeline.cli.cmd_validate import (
    REQUIRED_SECTIONS,
    _check_confidence_levels,
    _check_evidence_citations,
    _check_gap_classification,
    _check_latex,
    _check_mermaid,
    _check_sections,
    _check_tables,
    _extract_headings,
    validate_report,
)


def _make_full_report() -> str:
    """Create a report that passes all validation checks."""
    sections = [
        "# Research Report",
        "",
        "## Executive Summary",
        "This is a study on AI memory systems. Verdict: IMPLEMENTATION_READY",
        "",
        "## Research Question",
        "How do AI agents use memory?",
        "",
        "## Methodology",
        "We surveyed 10 papers.",
        "",
        "```mermaid",
        "graph TD",
        "  A --> B",
        "```",
        "",
        "## Papers Reviewed",
        "| Paper | Year | Method |",
        "| --- | --- | --- |",
        "| [2401.12345] | 2024 | Retrieval |",
        "| [2401.67890] | 2024 | Generation |",
        "",
        "## Research Landscape",
        "The field is active.",
        "",
        "## Methodology Comparison",
        "| Approach | Pros | Cons |",
        "| --- | --- | --- |",
        "| RAG | Fast | Less accurate |",
        "| Fine-tuning | Accurate | Slow |",
        "",
        "## Confidence-Graded Findings",
        "🟢 **High confidence**: Memory retrieval improves performance [2401.12345]",
        "🟡 **Medium confidence**: Hybrid approaches show promise [2401.67890]",
        "🔴 **Low confidence**: Long-term storage remains unsolved",
        "",
        "## Trade-Off Analysis",
        "Speed vs accuracy is the main trade-off.",
        "The formula is: $E = mc^2$",
        "",
        "## Points of Agreement",
        "All papers agree on the importance of retrieval.",
        "",
        "## Points of Contradiction",
        "Papers disagree on optimal chunk size.",
        "",
        "## Research Gaps",
        "- ACADEMIC: Long-term memory persistence (HIGH severity)",
        "- ENGINEERING: Integration with existing frameworks",
        "",
        "## Reproducibility Notes",
        "Most papers provide code. $$\\text{Score} = \\frac{a}{b}$$",
        "",
        "## Practical Recommendations",
        "Use RAG for most applications.",
        "",
        "## References",
        "- [2401.12345] Paper A",
        "- [2401.67890] Paper B",
        "- [2401.11111] Paper C",
    ]
    return "\n".join(sections)


class TestExtractHeadings:
    def test_extracts_markdown_headings(self) -> None:
        text = "# Title\n## Section A\n### Sub B\nNot a heading"
        headings = _extract_headings(text)
        assert headings == ["title", "section a", "sub b"]

    def test_empty_text(self) -> None:
        assert _extract_headings("") == []


class TestCheckSections:
    def test_full_report_has_all_sections(self) -> None:
        text = _make_full_report()
        present, missing_req, missing_opt = _check_sections(text)
        assert len(missing_req) == 0
        assert len(present) == len(REQUIRED_SECTIONS)

    def test_missing_sections_detected(self) -> None:
        text = "# Report\n## Executive Summary\nHello"
        present, missing_req, _ = _check_sections(text)
        assert "executive summary" in present
        assert len(missing_req) > 0


class TestCheckConfidenceLevels:
    def test_detects_emoji_confidence(self) -> None:
        text = "🟢 High confidence finding\n🟡 Medium\n🔴 Low"
        counts = _check_confidence_levels(text)
        assert counts["high"] >= 1
        assert counts["medium"] >= 1
        assert counts["low"] >= 1

    def test_no_confidence_levels(self) -> None:
        text = "Just a plain report without any confidence markers."
        counts = _check_confidence_levels(text)
        assert counts["high"] + counts["medium"] + counts["low"] == 0


class TestCheckEvidenceCitations:
    def test_counts_citations(self) -> None:
        text = "Finding [2401.12345] supports [2401.67890] and [Author2024]"
        assert _check_evidence_citations(text) == 3

    def test_no_citations(self) -> None:
        assert _check_evidence_citations("No citations here") == 0


class TestCheckGapClassification:
    def test_detects_gap_types(self) -> None:
        text = "ACADEMIC gap: X\nENGINEERING gap: Y\nACADEMIC gap: Z"
        result = _check_gap_classification(text)
        assert result["academic_gaps"] == 2
        assert result["engineering_gaps"] == 1


class TestCheckTables:
    def test_counts_tables(self) -> None:
        text = "| A | B |\n| --- | --- |\n| 1 | 2 |\n\nText\n\n| C |\n| --- |\n| 3 |"
        assert _check_tables(text) == 2

    def test_no_tables(self) -> None:
        assert _check_tables("No tables here") == 0


class TestCheckMermaid:
    def test_counts_mermaid(self) -> None:
        text = "```mermaid\ngraph TD\n```\n\n```mermaid\nsequenceDiagram\n```"
        assert _check_mermaid(text) == 2

    def test_no_mermaid(self) -> None:
        assert _check_mermaid("```python\ncode\n```") == 0


class TestCheckLatex:
    def test_counts_latex(self) -> None:
        text = "The formula $E=mc^2$ and display $$x = y$$"
        assert _check_latex(text) >= 2

    def test_no_latex(self) -> None:
        assert _check_latex("No formulas here") == 0


class TestValidateReport:
    def test_full_report_passes(self, tmp_path: Path) -> None:
        report = tmp_path / "report.md"
        report.write_text(_make_full_report())
        result = validate_report(report)
        assert result["verdict"] == "PASS"
        assert result["overall_score"] >= 0.7
        assert len(result["sections"]["missing_required"]) == 0

    def test_empty_report_fails(self, tmp_path: Path) -> None:
        report = tmp_path / "report.md"
        report.write_text("# Empty Report\nNothing here.")
        result = validate_report(report)
        assert result["verdict"] == "FAIL"
        assert len(result["sections"]["missing_required"]) > 0
        assert len(result["issues"]) > 0

    def test_partial_report_scoring(self, tmp_path: Path) -> None:
        text = (
            "# Report\n## Executive Summary\nSummary\n"
            "## Research Question\nQuestion\n"
            "## Methodology\nMethods\n"
            "## References\n- [2401.12345] A\n- [2401.67890] B\n- [2401.11111] C\n"
        )
        report = tmp_path / "report.md"
        report.write_text(text)
        result = validate_report(report)
        # Has some sections but not all
        assert 0 < result["section_score"] < 1.0
        assert result["overall_score"] < 0.7  # Shouldn't pass
