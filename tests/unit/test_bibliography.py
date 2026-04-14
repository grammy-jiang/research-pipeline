"""Unit tests for bibliography extraction from markdown."""

from research_pipeline.extraction.bibliography import (
    BibEntry,
    _extract_bib_section,
    _parse_entry,
    _split_entries,
    extract_bibliography,
)

SAMPLE_BIB_MARKDOWN = """# Some Paper Title

## Introduction

This is the introduction.

## Methods

We used several methods.

## References

[1] A. Author, B. Author, "Attention Is All You Need" (2017). arXiv:1706.03762.

[2] C. Author, *BERT: Pre-training of Deep Bidirectional Transformers* \
(2019). doi: 10.18653/v1/N19-1423.

[3] D. Author, E. Author, "GPT-4 Technical Report" (2023). arXiv:2303.08774v2.

[4] F. Author, "An old-style arXiv paper" (2006). arXiv: hep-ph/0601001.
"""

SAMPLE_BIB_MARKDOWN_BULLETS = """# Paper

## References

- A. Author, "First Paper" (2020). arXiv:2001.00001.
- B. Author, "Second Paper" (2021). doi: 10.1000/example.123.
- C. Author, "Third Paper" (2022).
"""


class TestExtractBibSection:
    """Tests for _extract_bib_section."""

    def test_extracts_references(self) -> None:
        section = _extract_bib_section(SAMPLE_BIB_MARKDOWN)
        assert "Attention Is All You Need" in section
        assert "BERT" in section

    def test_no_bib_section(self) -> None:
        md = "# Title\n\n## Introduction\n\nNo bibliography here."
        assert _extract_bib_section(md) == ""

    def test_stops_at_next_heading(self) -> None:
        md = "## References\n\n[1] Paper.\n\n## Appendix\n\nStuff."
        section = _extract_bib_section(md)
        assert "Paper" in section
        assert "Appendix" not in section
        assert "Stuff" not in section


class TestSplitEntries:
    """Tests for _split_entries."""

    def test_numbered_entries(self) -> None:
        section = _extract_bib_section(SAMPLE_BIB_MARKDOWN)
        entries = _split_entries(section)
        assert len(entries) == 4

    def test_bullet_entries(self) -> None:
        section = _extract_bib_section(SAMPLE_BIB_MARKDOWN_BULLETS)
        entries = _split_entries(section)
        assert len(entries) == 3


class TestParseEntry:
    """Tests for _parse_entry."""

    def test_arxiv_id(self) -> None:
        entry = _parse_entry('A. Author, "Title" (2017). arXiv:1706.03762.')
        assert entry.arxiv_id == "1706.03762"

    def test_arxiv_id_with_version(self) -> None:
        entry = _parse_entry('D. Author, "GPT-4" (2023). arXiv:2303.08774v2.')
        assert entry.arxiv_id == "2303.08774v2"

    def test_old_style_arxiv_id(self) -> None:
        entry = _parse_entry("F. Author, Title (2006). arXiv: hep-ph/0601001.")
        assert entry.arxiv_id == "hep-ph/0601001"

    def test_doi(self) -> None:
        entry = _parse_entry("C. Author, Title (2019). doi: 10.18653/v1/N19-1423.")
        assert entry.doi == "10.18653/v1/N19-1423"

    def test_year(self) -> None:
        entry = _parse_entry('Author, "Some Title" (2023). Info.')
        assert entry.year == 2023

    def test_title_in_quotes(self) -> None:
        entry = _parse_entry('A. Author, "My Great Title" (2024).')
        assert entry.title == "My Great Title"

    def test_title_in_italics(self) -> None:
        entry = _parse_entry("A. Author, *My Italic Title* (2024).")
        assert entry.title == "My Italic Title"

    def test_no_identifiers(self) -> None:
        entry = _parse_entry("Author, Some title without any IDs, 2024.")
        assert entry.arxiv_id == ""
        assert entry.doi == ""
        assert entry.year == 2024

    def test_raw_text_preserved(self) -> None:
        raw = 'A. Author, "Title" (2024). arXiv:2401.00001.'
        entry = _parse_entry(raw)
        assert entry.raw_text == raw


class TestExtractBibliography:
    """Integration tests for extract_bibliography."""

    def test_full_extraction(self) -> None:
        entries = extract_bibliography(SAMPLE_BIB_MARKDOWN)
        assert len(entries) == 4

        # First entry: arXiv + title + year
        assert entries[0].arxiv_id == "1706.03762"
        assert entries[0].title == "Attention Is All You Need"
        assert entries[0].year == 2017

        # Second entry: DOI + title
        assert entries[1].doi == "10.18653/v1/N19-1423"
        assert "BERT" in entries[1].title

        # Third entry: arXiv with version
        assert entries[2].arxiv_id == "2303.08774v2"

        # Fourth entry: old-style arXiv
        assert entries[3].arxiv_id == "hep-ph/0601001"

    def test_no_references_section(self) -> None:
        md = "# Title\n\nJust content, no references."
        assert extract_bibliography(md) == []

    def test_empty_markdown(self) -> None:
        assert extract_bibliography("") == []

    def test_bullet_format(self) -> None:
        entries = extract_bibliography(SAMPLE_BIB_MARKDOWN_BULLETS)
        assert len(entries) == 3
        assert entries[0].arxiv_id == "2001.00001"
        assert entries[1].doi == "10.1000/example.123"

    def test_bib_entry_is_dataclass(self) -> None:
        entry = BibEntry(raw_text="test")
        assert entry.raw_text == "test"
        assert entry.title == ""
        assert entry.authors == []
        assert entry.year is None
        assert entry.arxiv_id == ""
        assert entry.doi == ""
