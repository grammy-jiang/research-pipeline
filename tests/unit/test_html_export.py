"""Unit tests for HTML report export (A5)."""

from __future__ import annotations

from pathlib import Path

import pytest

from research_pipeline.models.summary import (
    PaperSummary,
    SynthesisAgreement,
    SynthesisDisagreement,
    SynthesisReport,
)
from research_pipeline.summarization.html_export import (
    CONFIDENCE_COLORS,
    CONFIDENCE_LABELS,
    _detect_confidence,
    _escape,
    _inline_format,
    _linkify_arxiv_ids,
    _markdown_to_html,
    render_html_from_markdown,
    render_html_report,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_report() -> SynthesisReport:
    """Create a sample SynthesisReport for testing."""
    return SynthesisReport(
        topic="Transformer architectures",
        paper_count=3,
        agreements=[
            SynthesisAgreement(
                claim="Transformers outperform RNNs with strong evidence",
                supporting_papers=["2301.12345", "2302.67890"],
            ),
            SynthesisAgreement(
                claim="Some evidence for attention efficiency",
                supporting_papers=["2301.12345"],
            ),
        ],
        disagreements=[
            SynthesisDisagreement(
                topic="Optimal attention head count",
                positions={
                    "2301.12345": "More heads better",
                    "2302.67890": "Fewer heads sufficient",
                },
            ),
        ],
        open_questions=[
            "How does scaling affect long-context performance?",
            "What is the optimal layer depth?",
        ],
        paper_summaries=[
            PaperSummary(
                arxiv_id="2301.12345",
                version="v1",
                title="Efficient Transformers",
                objective="Study attention mechanisms",
                methodology="Empirical evaluation",
                findings=["Linear attention scales better"],
                limitations=["Small dataset"],
            ),
            PaperSummary(
                arxiv_id="2302.67890",
                version="v1",
                title="Attention Is All You Need (Revisited)",
                objective="Revisit original transformer",
                methodology="Ablation study",
                findings=["Multi-head is key", "Dropout helps"],
                limitations=[],
            ),
        ],
    )


@pytest.fixture()
def minimal_report() -> SynthesisReport:
    """A minimal report with only topic and count."""
    return SynthesisReport(
        topic="Minimal topic",
        paper_count=0,
        agreements=[],
        disagreements=[],
        open_questions=[],
        paper_summaries=[],
    )


@pytest.fixture()
def sample_markdown(tmp_path: Path) -> Path:
    """Create a sample Markdown file for testing."""
    md = tmp_path / "report.md"
    md.write_text(
        "# Test Report\n\n"
        "## Introduction\n\n"
        "This is a test with arXiv 2301.12345 and **bold** text.\n\n"
        "- Item one\n"
        "- Item two\n\n"
        "1. First\n"
        "2. Second\n\n"
        "> A blockquote\n\n"
        "```python\nprint('hello')\n```\n\n"
        "---\n\n"
        "| Header | Value |\n"
        "| --- | --- |\n"
        "| A | B |\n",
        encoding="utf-8",
    )
    return md


# ---------------------------------------------------------------------------
# _detect_confidence
# ---------------------------------------------------------------------------


class TestDetectConfidence:
    """Tests for confidence level detection."""

    def test_high_strong_evidence(self) -> None:
        assert _detect_confidence("strong evidence shows this") == "high"

    def test_high_well_established(self) -> None:
        assert _detect_confidence("This is well-established") == "high"

    def test_high_robust(self) -> None:
        assert _detect_confidence("Robust findings confirm") == "high"

    def test_medium_some_evidence(self) -> None:
        assert _detect_confidence("some evidence suggests") == "medium"

    def test_medium_moderate(self) -> None:
        assert _detect_confidence("moderate support for") == "medium"

    def test_medium_preliminary(self) -> None:
        assert _detect_confidence("preliminary results indicate") == "medium"

    def test_low_limited_evidence(self) -> None:
        assert _detect_confidence("limited evidence available") == "low"

    def test_low_unclear(self) -> None:
        assert _detect_confidence("unclear whether this holds") == "low"

    def test_low_speculative(self) -> None:
        assert _detect_confidence("speculative claim about") == "low"

    def test_unknown_neutral(self) -> None:
        assert _detect_confidence("transformers are neural networks") == "unknown"

    def test_empty_string(self) -> None:
        assert _detect_confidence("") == "unknown"

    def test_case_insensitive(self) -> None:
        assert _detect_confidence("STRONG EVIDENCE") == "high"


# ---------------------------------------------------------------------------
# _linkify_arxiv_ids
# ---------------------------------------------------------------------------


class TestLinkifyArxivIds:
    """Tests for arXiv ID linkification."""

    def test_single_id(self) -> None:
        result = _linkify_arxiv_ids("See 2301.12345 for details")
        assert "arxiv.org/abs/2301.12345" in result
        assert 'target="_blank"' in result

    def test_multiple_ids(self) -> None:
        result = _linkify_arxiv_ids("Papers 2301.12345 and 2302.67890")
        assert "2301.12345" in result
        assert "2302.67890" in result
        assert result.count("arxiv.org") == 2

    def test_versioned_id(self) -> None:
        result = _linkify_arxiv_ids("Paper 2301.12345v2")
        assert "arxiv.org/abs/2301.12345v2" in result

    def test_no_ids(self) -> None:
        text = "No arXiv IDs here"
        assert _linkify_arxiv_ids(text) == text

    def test_five_digit_id(self) -> None:
        result = _linkify_arxiv_ids("ID 2301.12345")
        assert "arxiv.org/abs/2301.12345" in result


# ---------------------------------------------------------------------------
# _escape
# ---------------------------------------------------------------------------


class TestEscape:
    """Tests for HTML escaping with linkification."""

    def test_html_entities_escaped(self) -> None:
        result = _escape("<script>alert('xss')</script>")
        assert "&lt;script&gt;" in result
        assert "<script>" not in result

    def test_preserves_arxiv_links(self) -> None:
        result = _escape("See 2301.12345")
        assert "arxiv.org/abs/2301.12345" in result

    def test_ampersand_escaped(self) -> None:
        result = _escape("A & B")
        assert "&amp;" in result


# ---------------------------------------------------------------------------
# _inline_format
# ---------------------------------------------------------------------------


class TestInlineFormat:
    """Tests for inline Markdown formatting."""

    def test_bold(self) -> None:
        assert "<strong>bold</strong>" in _inline_format("**bold**")

    def test_italic(self) -> None:
        assert "<em>italic</em>" in _inline_format("*italic*")

    def test_bold_italic(self) -> None:
        result = _inline_format("***both***")
        assert "<strong><em>both</em></strong>" in result

    def test_inline_code(self) -> None:
        assert "<code>code</code>" in _inline_format("`code`")

    def test_link(self) -> None:
        result = _inline_format("[text](https://example.com)")
        assert 'href="https://example.com"' in result
        assert ">text</a>" in result

    def test_html_escape(self) -> None:
        result = _inline_format("<div>test</div>")
        assert "&lt;div&gt;" in result

    def test_plain_text(self) -> None:
        assert _inline_format("plain text") == "plain text"


# ---------------------------------------------------------------------------
# _markdown_to_html
# ---------------------------------------------------------------------------


class TestMarkdownToHtml:
    """Tests for Markdown to HTML conversion."""

    def test_heading_h1(self) -> None:
        result = _markdown_to_html("# Title")
        assert "<h1" in result
        assert "Title</h1>" in result

    def test_heading_h2(self) -> None:
        result = _markdown_to_html("## Subtitle")
        assert "<h2" in result

    def test_heading_h3(self) -> None:
        result = _markdown_to_html("### Section")
        assert "<h3" in result

    def test_heading_id_slug(self) -> None:
        result = _markdown_to_html("## My Section Title")
        assert 'id="my-section-title"' in result

    def test_paragraph(self) -> None:
        result = _markdown_to_html("Simple paragraph text")
        assert "<p>Simple paragraph text</p>" in result

    def test_unordered_list(self) -> None:
        result = _markdown_to_html("- Item 1\n- Item 2")
        assert "<ul>" in result
        assert "<li>Item 1</li>" in result
        assert "<li>Item 2</li>" in result
        assert "</ul>" in result

    def test_ordered_list(self) -> None:
        result = _markdown_to_html("1. First\n2. Second")
        assert "<ol>" in result
        assert "<li>First</li>" in result
        assert "</ol>" in result

    def test_code_block(self) -> None:
        result = _markdown_to_html("```python\nprint('hi')\n```")
        assert "<pre><code" in result
        assert "print(&#x27;hi&#x27;)" in result or "print('hi')" in result
        assert "</code></pre>" in result

    def test_code_block_no_language(self) -> None:
        result = _markdown_to_html("```\ncode\n```")
        assert "<pre><code>" in result

    def test_blockquote(self) -> None:
        result = _markdown_to_html("> Quote text")
        assert "<blockquote>" in result
        assert "Quote text" in result

    def test_horizontal_rule(self) -> None:
        assert "<hr>" in _markdown_to_html("---")
        assert "<hr>" in _markdown_to_html("***")
        assert "<hr>" in _markdown_to_html("___")

    def test_table(self) -> None:
        md = "| A | B |\n| --- | --- |\n| 1 | 2 |"
        result = _markdown_to_html(md)
        assert "<table>" in result
        assert "<th>" in result
        assert "<td>" in result
        assert "</table>" in result

    def test_blank_lines(self) -> None:
        result = _markdown_to_html("Line 1\n\nLine 2")
        assert "<p>Line 1</p>" in result
        assert "<p>Line 2</p>" in result

    def test_list_closes_properly(self) -> None:
        result = _markdown_to_html("- Item\n\nParagraph")
        assert "</ul>" in result
        assert "<p>Paragraph</p>" in result

    def test_mixed_content(self) -> None:
        md = "# Title\n\n- Item\n\n> Quote\n\nParagraph"
        result = _markdown_to_html(md)
        assert "<h1" in result
        assert "<ul>" in result
        assert "<blockquote>" in result
        assert "<p>Paragraph</p>" in result

    def test_inline_formatting_in_list(self) -> None:
        result = _markdown_to_html("- **Bold** item")
        assert "<strong>Bold</strong>" in result


# ---------------------------------------------------------------------------
# render_html_report
# ---------------------------------------------------------------------------


class TestRenderHtmlReport:
    """Tests for structured SynthesisReport rendering."""

    def test_returns_html_string(self, sample_report: SynthesisReport) -> None:
        result = render_html_report(sample_report)
        assert isinstance(result, str)
        assert "<!DOCTYPE html>" in result

    def test_contains_title(self, sample_report: SynthesisReport) -> None:
        result = render_html_report(sample_report)
        assert "Transformer architectures" in result

    def test_contains_paper_count(self, sample_report: SynthesisReport) -> None:
        result = render_html_report(sample_report)
        assert "3 papers" in result

    def test_contains_agreements(self, sample_report: SynthesisReport) -> None:
        result = render_html_report(sample_report)
        assert "Agreements" in result
        assert "Transformers outperform RNNs" in result

    def test_contains_disagreements(self, sample_report: SynthesisReport) -> None:
        result = render_html_report(sample_report)
        assert "Disagreements" in result
        assert "Optimal attention head count" in result

    def test_contains_open_questions(self, sample_report: SynthesisReport) -> None:
        result = render_html_report(sample_report)
        assert "Open Questions" in result
        assert "long-context performance" in result

    def test_contains_paper_summaries(self, sample_report: SynthesisReport) -> None:
        result = render_html_report(sample_report)
        assert "Paper Summaries" in result
        assert "Efficient Transformers" in result
        assert "Linear attention scales better" in result

    def test_arxiv_links_present(self, sample_report: SynthesisReport) -> None:
        result = render_html_report(sample_report)
        assert "arxiv.org/abs/2301.12345" in result
        assert "arxiv.org/abs/2302.67890" in result

    def test_confidence_badges(self, sample_report: SynthesisReport) -> None:
        result = render_html_report(sample_report)
        # "strong evidence" → high confidence badge
        assert CONFIDENCE_COLORS["high"] in result
        # "Some evidence" → medium confidence
        assert CONFIDENCE_COLORS["medium"] in result

    def test_responsive_meta(self, sample_report: SynthesisReport) -> None:
        result = render_html_report(sample_report)
        assert "viewport" in result
        assert "width=device-width" in result

    def test_dark_mode_support(self, sample_report: SynthesisReport) -> None:
        result = render_html_report(sample_report)
        assert "prefers-color-scheme: dark" in result

    def test_navigation_toc(self, sample_report: SynthesisReport) -> None:
        result = render_html_report(sample_report)
        assert 'href="#agreements"' in result
        assert 'href="#papers"' in result

    def test_writes_to_file(
        self, sample_report: SynthesisReport, tmp_path: Path
    ) -> None:
        out = tmp_path / "report.html"
        render_html_report(sample_report, out)
        assert out.exists()
        content = out.read_text(encoding="utf-8")
        assert "<!DOCTYPE html>" in content

    def test_creates_parent_dirs(
        self, sample_report: SynthesisReport, tmp_path: Path
    ) -> None:
        out = tmp_path / "sub" / "dir" / "report.html"
        render_html_report(sample_report, out)
        assert out.exists()

    def test_none_output_no_file(self, sample_report: SynthesisReport) -> None:
        result = render_html_report(sample_report, None)
        assert "<!DOCTYPE html>" in result

    def test_minimal_report(self, minimal_report: SynthesisReport) -> None:
        result = render_html_report(minimal_report)
        assert "<!DOCTYPE html>" in result
        assert "0 papers" in result
        # No sections should appear
        assert "Agreements" not in result or "Agreements (0)" not in result

    def test_html_escaping_in_claims(self) -> None:
        """Verify XSS-safe output."""
        report = SynthesisReport(
            topic="<script>alert('xss')</script>",
            paper_count=1,
            agreements=[
                SynthesisAgreement(
                    claim="<img onerror=alert(1)>",
                    supporting_papers=["2301.12345"],
                ),
            ],
            disagreements=[],
            open_questions=["<b>bold?</b>"],
            paper_summaries=[],
        )
        result = render_html_report(report)
        assert "<script>" not in result
        assert "<img onerror" not in result

    def test_findings_rendered(self, sample_report: SynthesisReport) -> None:
        result = render_html_report(sample_report)
        assert "Linear attention scales better" in result

    def test_limitations_rendered(self, sample_report: SynthesisReport) -> None:
        result = render_html_report(sample_report)
        assert "Small dataset" in result

    def test_multiple_findings(self, sample_report: SynthesisReport) -> None:
        result = render_html_report(sample_report)
        assert "Multi-head is key" in result
        assert "Dropout helps" in result

    def test_disagreement_positions(self, sample_report: SynthesisReport) -> None:
        result = render_html_report(sample_report)
        assert "More heads better" in result
        assert "Fewer heads sufficient" in result


# ---------------------------------------------------------------------------
# render_html_from_markdown
# ---------------------------------------------------------------------------


class TestRenderHtmlFromMarkdown:
    """Tests for Markdown → HTML conversion."""

    def test_returns_html(self, sample_markdown: Path) -> None:
        result = render_html_from_markdown(sample_markdown)
        assert "<!DOCTYPE html>" in result

    def test_contains_heading(self, sample_markdown: Path) -> None:
        result = render_html_from_markdown(sample_markdown)
        assert "Test Report" in result

    def test_arxiv_linked(self, sample_markdown: Path) -> None:
        result = render_html_from_markdown(sample_markdown)
        assert "arxiv.org/abs/2301.12345" in result

    def test_custom_title(self, sample_markdown: Path) -> None:
        result = render_html_from_markdown(sample_markdown, title="Custom Title")
        assert "Custom Title" in result

    def test_writes_to_file(self, sample_markdown: Path, tmp_path: Path) -> None:
        out = tmp_path / "output.html"
        render_html_from_markdown(sample_markdown, out)
        assert out.exists()

    def test_creates_parent_dirs(self, sample_markdown: Path, tmp_path: Path) -> None:
        out = tmp_path / "deep" / "path" / "output.html"
        render_html_from_markdown(sample_markdown, out)
        assert out.exists()

    def test_default_title(self, sample_markdown: Path) -> None:
        result = render_html_from_markdown(sample_markdown)
        assert "Research Report" in result

    def test_responsive_styling(self, sample_markdown: Path) -> None:
        result = render_html_from_markdown(sample_markdown)
        assert "viewport" in result

    def test_dark_mode_styling(self, sample_markdown: Path) -> None:
        result = render_html_from_markdown(sample_markdown)
        assert "prefers-color-scheme: dark" in result

    def test_bold_text(self, sample_markdown: Path) -> None:
        result = render_html_from_markdown(sample_markdown)
        assert "<strong>bold</strong>" in result

    def test_list_items(self, sample_markdown: Path) -> None:
        result = render_html_from_markdown(sample_markdown)
        assert "Item one" in result
        assert "Item two" in result

    def test_ordered_list(self, sample_markdown: Path) -> None:
        result = render_html_from_markdown(sample_markdown)
        assert "<ol>" in result

    def test_blockquote(self, sample_markdown: Path) -> None:
        result = render_html_from_markdown(sample_markdown)
        assert "<blockquote>" in result

    def test_code_block(self, sample_markdown: Path) -> None:
        result = render_html_from_markdown(sample_markdown)
        assert "<pre><code" in result

    def test_horizontal_rule(self, sample_markdown: Path) -> None:
        result = render_html_from_markdown(sample_markdown)
        assert "<hr>" in result

    def test_table(self, sample_markdown: Path) -> None:
        result = render_html_from_markdown(sample_markdown)
        assert "<table>" in result


# ---------------------------------------------------------------------------
# Constants / edge cases
# ---------------------------------------------------------------------------


class TestConstants:
    """Tests for module-level constants."""

    def test_confidence_colors_keys(self) -> None:
        assert set(CONFIDENCE_COLORS.keys()) == {
            "high",
            "medium",
            "low",
            "unknown",
        }

    def test_confidence_labels_keys(self) -> None:
        assert set(CONFIDENCE_LABELS.keys()) == {
            "high",
            "medium",
            "low",
            "unknown",
        }

    def test_all_colors_are_hex(self) -> None:
        for color in CONFIDENCE_COLORS.values():
            assert color.startswith("#")

    def test_all_labels_are_strings(self) -> None:
        for label in CONFIDENCE_LABELS.values():
            assert isinstance(label, str)
            assert len(label) > 0


class TestEdgeCases:
    """Edge case and integration tests."""

    def test_empty_markdown_file(self, tmp_path: Path) -> None:
        md = tmp_path / "empty.md"
        md.write_text("", encoding="utf-8")
        result = render_html_from_markdown(md)
        assert "<!DOCTYPE html>" in result

    def test_unicode_content(self, tmp_path: Path) -> None:
        md = tmp_path / "unicode.md"
        md.write_text(
            "# 研究报告\n\nTransformer 架构 — análisis\n",
            encoding="utf-8",
        )
        result = render_html_from_markdown(md)
        assert "研究报告" in result
        assert "análisis" in result

    def test_large_report(self) -> None:
        """Render a report with many papers."""
        summaries = [
            PaperSummary(
                arxiv_id=f"2301.{10000 + i}",
                version="v1",
                title=f"Paper {i}",
                objective=f"Objective {i}",
                methodology=f"Method {i}",
                findings=[f"Finding {i}"],
                limitations=[],
            )
            for i in range(50)
        ]
        report = SynthesisReport(
            topic="Large-scale test",
            paper_count=50,
            agreements=[],
            disagreements=[],
            open_questions=[],
            paper_summaries=summaries,
        )
        result = render_html_report(report)
        assert "50 papers" in result
        assert "Paper 49" in result

    def test_special_chars_in_topic(self) -> None:
        report = SynthesisReport(
            topic='Topic with "quotes" & <brackets>',
            paper_count=0,
            agreements=[],
            disagreements=[],
            open_questions=[],
            paper_summaries=[],
        )
        result = render_html_report(report)
        assert "&amp;" in result or "& " in result
        assert "<brackets>" not in result  # Must be escaped
