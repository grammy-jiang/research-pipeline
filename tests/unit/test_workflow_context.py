"""Tests for context engineering: token budgets and paper compaction."""

from __future__ import annotations

from mcp_server.workflow.context import (
    CHARS_PER_TOKEN,
    CompactionLevel,
    compact_paper,
    estimate_tokens,
    extract_aggressive,
    extract_sections,
)


class TestEstimateTokens:
    """Token estimation tests."""

    def test_empty_string(self) -> None:
        # min 1 token for any input (includes empty string)
        assert estimate_tokens("") == 1

    def test_short_string(self) -> None:
        result = estimate_tokens("hello")
        assert result == len("hello") // CHARS_PER_TOKEN or result >= 1

    def test_approximate_accuracy(self) -> None:
        text = "word " * 1000
        tokens = estimate_tokens(text)
        assert 500 <= tokens <= 2000


class TestExtractSections:
    """Section extraction tests."""

    def test_extracts_standard_sections(self) -> None:
        markdown = """# Title
## Abstract
This is the abstract.

## 1. Introduction
This is the introduction.

## 2. Methodology
This is the methodology.

## 3. Results
These are the results.

## 4. Discussion
This is the discussion.

## 5. Conclusion
This is the conclusion.

## References
[1] Some reference
"""
        result = extract_sections(markdown)
        assert "abstract" in result.lower()
        assert "methodology" in result.lower()
        assert "conclusion" in result.lower()
        # References should be stripped
        assert "Some reference" not in result

    def test_preserves_content_without_sections(self) -> None:
        plain = "Just some text without any section headings."
        result = extract_sections(plain)
        assert len(result) > 0

    def test_strips_references(self) -> None:
        markdown = """## Abstract
Content here.

## References
[1] A reference
[2] Another reference
"""
        result = extract_sections(markdown)
        assert "A reference" not in result


class TestExtractAggressive:
    """Aggressive extraction tests."""

    def test_keeps_abstract_and_methodology(self) -> None:
        markdown = """## Abstract
This is the abstract.

## Introduction
Long introduction content that should be removed.

## Methodology
This is the methodology.

## Results
These are the results.

## Discussion
This should be removed.

## Conclusion
This should be removed.
"""
        result = extract_aggressive(markdown)
        assert "abstract" in result.lower()
        assert "methodology" in result.lower()
        assert "results" in result.lower()

    def test_handles_missing_sections(self) -> None:
        markdown = "Just a paper with no clear sections."
        result = extract_aggressive(markdown)
        assert len(result) > 0


class TestCompactPaper:
    """Paper compaction with budget awareness."""

    def test_full_paper_fits(self) -> None:
        short_paper = "A short paper. " * 100  # ~400 words, ~1600 chars
        content, level = compact_paper(short_paper, max_tokens=10000)
        assert level == CompactionLevel.FULL
        assert content == short_paper

    def test_sections_extracted_when_too_long(self) -> None:
        long_paper = (
            """## Abstract
Short abstract.

## Introduction
"""
            + "Long introduction. " * 5000
            + """
## Methodology
Short methodology.

## Results
Short results.

## References
[1] A reference.
"""
        )
        content, level = compact_paper(long_paper, max_tokens=500)
        assert level in (
            CompactionLevel.SECTIONS,
            CompactionLevel.AGGRESSIVE,
            CompactionLevel.ABSTRACT_ONLY,
        )
        assert len(content) < len(long_paper)

    def test_abstract_only_for_tiny_budget(self) -> None:
        long_paper = "## Abstract\nShort abstract.\n\n" + "x" * 100000
        content, level = compact_paper(long_paper, max_tokens=50)
        assert level == CompactionLevel.ABSTRACT_ONLY

    def test_zero_budget_returns_truncated(self) -> None:
        content, level = compact_paper("Some paper content", max_tokens=0)
        # With 0 budget, abstract truncation to 0 chars
        assert level == CompactionLevel.ABSTRACT_ONLY


class TestCompactionLevel:
    """CompactionLevel enum tests."""

    def test_ordering(self) -> None:
        levels = list(CompactionLevel)
        assert levels[0] == CompactionLevel.FULL
        assert levels[-1] == CompactionLevel.SKIPPED

    def test_all_levels(self) -> None:
        expected = {"full", "sections", "aggressive", "abstract_only", "skipped"}
        assert {c.value for c in CompactionLevel} == expected
