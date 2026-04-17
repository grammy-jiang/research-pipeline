"""Tests for summarization.report_templates — Jinja2 report rendering."""

from __future__ import annotations

from pathlib import Path

import pytest

from research_pipeline.models.summary import (
    PaperSummary,
    SummaryEvidence,
    SynthesisAgreement,
    SynthesisDisagreement,
    SynthesisReport,
)
from research_pipeline.summarization.report_templates import (
    TEMPLATES,
    list_templates,
    render_report,
    render_report_to_file,
)


def _make_report() -> SynthesisReport:
    """Create a test synthesis report."""
    return SynthesisReport(
        topic="Large Language Models",
        paper_count=2,
        agreements=[
            SynthesisAgreement(
                claim="Transformers are effective for NLP",
                supporting_papers=["2401.00001", "2401.00002"],
                evidence=[
                    SummaryEvidence(
                        chunk_id="chunk-1",
                        quote="experimental results show...",
                    )
                ],
            ),
        ],
        disagreements=[
            SynthesisDisagreement(
                topic="Scaling laws",
                positions={
                    "2401.00001": "Linear scaling is sufficient",
                    "2401.00002": "Power-law scaling is needed",
                },
            ),
        ],
        open_questions=["How to reduce hallucination?"],
        paper_summaries=[
            PaperSummary(
                arxiv_id="2401.00001",
                version="v1",
                title="Paper Alpha",
                objective="Study LLM scaling",
                methodology="Empirical benchmarks",
                findings=["Scaling improves performance"],
                limitations=["Only tested on English"],
            ),
            PaperSummary(
                arxiv_id="2401.00002",
                version="v1",
                title="Paper Beta",
                objective="Analyze inference costs",
                methodology="Cost modeling",
                findings=["Costs grow sublinearly"],
                limitations=["Ignores training costs"],
            ),
        ],
    )


class TestListTemplates:
    """Tests for list_templates()."""

    def test_returns_sorted_names(self) -> None:
        names = list_templates()
        assert names == sorted(names)

    def test_includes_all_builtin(self) -> None:
        names = list_templates()
        assert "survey" in names
        assert "gap_analysis" in names
        assert "lit_review" in names
        assert "executive" in names

    def test_four_templates(self) -> None:
        assert len(list_templates()) == 4


class TestRenderReport:
    """Tests for render_report()."""

    def test_survey_template(self) -> None:
        report = _make_report()
        rendered = render_report(report, "survey")
        assert "# Survey: Large Language Models" in rendered
        assert "Paper Alpha" in rendered
        assert "Paper Beta" in rendered
        assert "Transformers are effective" in rendered
        assert "Scaling laws" in rendered

    def test_gap_analysis_template(self) -> None:
        report = _make_report()
        rendered = render_report(report, "gap_analysis")
        assert "# Gap Analysis:" in rendered
        assert "How to reduce hallucination?" in rendered
        assert "Coverage Matrix" in rendered

    def test_lit_review_template(self) -> None:
        report = _make_report()
        rendered = render_report(report, "lit_review")
        assert "# Literature Review:" in rendered
        assert "Summary Table" in rendered
        assert "2401.00001" in rendered

    def test_executive_template(self) -> None:
        report = _make_report()
        rendered = render_report(report, "executive")
        assert "# Executive Summary:" in rendered
        assert "Research Gaps" in rendered

    def test_unknown_template_raises(self) -> None:
        report = _make_report()
        with pytest.raises(ValueError, match="Unknown template"):
            render_report(report, "nonexistent")

    def test_custom_template(self) -> None:
        report = _make_report()
        custom = "Topic: {{ report.topic }}, Papers: {{ report.paper_count }}"
        rendered = render_report(report, custom_template=custom)
        assert "Topic: Large Language Models" in rendered
        assert "Papers: 2" in rendered

    def test_custom_overrides_name(self) -> None:
        report = _make_report()
        custom = "CUSTOM: {{ report.topic }}"
        rendered = render_report(report, "survey", custom_template=custom)
        assert "CUSTOM:" in rendered
        assert "# Survey:" not in rendered

    def test_includes_version(self) -> None:
        from research_pipeline import __version__

        report = _make_report()
        rendered = render_report(report, "survey")
        assert __version__ in rendered

    def test_includes_timestamp(self) -> None:
        report = _make_report()
        rendered = render_report(report, "survey")
        assert "UTC" in rendered

    def test_ends_with_newline(self) -> None:
        report = _make_report()
        rendered = render_report(report, "survey")
        assert rendered.endswith("\n")


class TestRenderReportToFile:
    """Tests for render_report_to_file()."""

    def test_writes_file(self, tmp_path: Path) -> None:
        report = _make_report()
        out = tmp_path / "report.md"
        render_report_to_file(report, out, "survey")
        assert out.exists()
        content = out.read_text()
        assert "# Survey:" in content

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        report = _make_report()
        out = tmp_path / "sub" / "dir" / "report.md"
        render_report_to_file(report, out, "executive")
        assert out.exists()

    def test_all_templates_render_without_error(self) -> None:
        report = _make_report()
        for name in list_templates():
            rendered = render_report(report, name)
            assert len(rendered) > 50, f"Template {name} produced too-short output"


class TestEmptyReport:
    """Test rendering with empty/minimal reports."""

    def test_empty_agreements(self) -> None:
        report = SynthesisReport(
            topic="Empty Topic",
            paper_count=0,
        )
        rendered = render_report(report, "survey")
        assert "No cross-paper agreements" in rendered

    def test_empty_disagreements(self) -> None:
        report = SynthesisReport(
            topic="Empty Topic",
            paper_count=0,
        )
        rendered = render_report(report, "executive")
        assert "No active debates" in rendered

    def test_empty_open_questions(self) -> None:
        report = SynthesisReport(
            topic="Empty Topic",
            paper_count=0,
        )
        rendered = render_report(report, "executive")
        assert "No gaps identified" in rendered
