"""Tests for summarization.bibtex_export — rich BibTeX from CandidateRecord."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from research_pipeline.models.candidate import CandidateRecord
from research_pipeline.summarization.bibtex_export import (
    _escape,
    _extract_year,
    _format_authors,
    _make_citation_key,
    candidate_to_bibtex,
    export_candidates_bibtex,
    load_candidates_from_jsonl,
)


def _make_candidate(**overrides: object) -> CandidateRecord:
    """Create a minimal CandidateRecord with overrides."""
    defaults: dict[str, object] = {
        "arxiv_id": "2401.12345",
        "version": "v1",
        "title": "Attention Is All You Need",
        "authors": ["Ashish Vaswani", "Noam Shazeer"],
        "abstract": "We propose a new architecture.",
        "published": "2024-01-15T00:00:00Z",
        "updated": "2024-01-15T00:00:00Z",
        "categories": ["cs.CL", "cs.AI"],
        "primary_category": "cs.CL",
        "abs_url": "https://arxiv.org/abs/2401.12345",
        "pdf_url": "https://arxiv.org/pdf/2401.12345",
        "source": "arxiv",
    }
    defaults.update(overrides)
    return CandidateRecord.model_validate(defaults)


class TestEscape:
    """Tests for _escape()."""

    def test_no_special_chars(self) -> None:
        assert _escape("Hello World") == "Hello World"

    def test_ampersand(self) -> None:
        assert _escape("A & B") == r"A \& B"

    def test_percent(self) -> None:
        assert _escape("100%") == r"100\%"

    def test_dollar(self) -> None:
        assert _escape("$100") == r"\$100"

    def test_hash(self) -> None:
        assert _escape("#1") == r"\#1"

    def test_underscore(self) -> None:
        assert _escape("hello_world") == r"hello\_world"

    def test_multiple_specials(self) -> None:
        assert _escape("A & B_C") == r"A \& B\_C"


class TestExtractYear:
    """Tests for _extract_year()."""

    def test_from_year_field(self) -> None:
        rec = _make_candidate(year=2023)
        assert _extract_year(rec) == "2023"

    def test_from_arxiv_id(self) -> None:
        rec = _make_candidate(year=None)
        assert _extract_year(rec) == "2024"

    def test_from_published_field(self) -> None:
        from datetime import datetime

        rec = _make_candidate(
            arxiv_id="unknown-id",
            year=None,
            published=datetime(2022, 6, 15),
            updated=datetime(2022, 6, 15),
        )
        assert _extract_year(rec) == "2022"

    def test_unknown_fallback(self) -> None:
        rec = _make_candidate(arxiv_id="unknown-id", year=None)
        assert _extract_year(rec) == "2024"  # falls back to published field

    def test_90s_arxiv_id(self) -> None:
        rec = _make_candidate(arxiv_id="9901.12345", year=None)
        assert _extract_year(rec) == "1999"


class TestMakeCitationKey:
    """Tests for _make_citation_key()."""

    def test_normal_key(self) -> None:
        rec = _make_candidate()
        key = _make_citation_key(rec)
        assert key == "vaswani2024attention"

    def test_no_authors(self) -> None:
        rec = _make_candidate(authors=[])
        key = _make_citation_key(rec)
        assert "2024" in key
        assert "attention" in key

    def test_single_name_author(self) -> None:
        rec = _make_candidate(authors=["Plato"])
        key = _make_citation_key(rec)
        assert key.startswith("plato")


class TestFormatAuthors:
    """Tests for _format_authors()."""

    def test_single_author(self) -> None:
        result = _format_authors(["John Smith"])
        assert result == "John Smith"

    def test_multiple_authors(self) -> None:
        result = _format_authors(["Alice", "Bob", "Charlie"])
        assert result == "Alice and Bob and Charlie"

    def test_special_chars_escaped(self) -> None:
        result = _format_authors(["O'Brien & Co."])
        assert r"\&" in result


class TestCandidateToBibtex:
    """Tests for candidate_to_bibtex()."""

    def test_arxiv_article(self) -> None:
        rec = _make_candidate(
            doi="10.1234/test",
            venue="NeurIPS",
            abs_url="https://arxiv.org/abs/2401.12345",
            primary_category="cs.CL",
        )
        bib = candidate_to_bibtex(rec)
        assert bib.startswith("@article{")
        assert "title = {" in bib
        assert "author = {" in bib
        assert "eprint = {2401.12345}" in bib
        assert "archivePrefix = {arXiv}" in bib
        assert "primaryClass = {cs.CL}" in bib
        assert "doi = {10.1234/test}" in bib
        assert "journal = {NeurIPS}" in bib
        assert "url = {https://arxiv.org/abs/2401.12345}" in bib
        assert "abstract = {" in bib

    def test_non_arxiv_misc(self) -> None:
        rec = _make_candidate(arxiv_id="custom-paper-id")
        bib = candidate_to_bibtex(rec)
        assert bib.startswith("@misc{")
        assert "eprint" not in bib
        assert "archivePrefix" not in bib

    def test_no_abstract(self) -> None:
        rec = _make_candidate(abstract="")
        bib = candidate_to_bibtex(rec)
        assert "abstract" not in bib

    def test_entry_ends_with_brace(self) -> None:
        rec = _make_candidate()
        bib = candidate_to_bibtex(rec)
        assert bib.rstrip().endswith("}")


class TestExportCandidatesBibtex:
    """Tests for export_candidates_bibtex()."""

    def test_writes_file(self, tmp_path: Path) -> None:
        out = tmp_path / "refs.bib"
        candidates = [_make_candidate(), _make_candidate(arxiv_id="2302.99999")]
        count = export_candidates_bibtex(candidates, out)
        assert count == 2
        assert out.exists()
        content = out.read_text()
        assert content.count("@article{") == 2

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        out = tmp_path / "sub" / "dir" / "refs.bib"
        count = export_candidates_bibtex([_make_candidate()], out)
        assert count == 1
        assert out.exists()

    def test_empty_list(self, tmp_path: Path) -> None:
        out = tmp_path / "refs.bib"
        count = export_candidates_bibtex([], out)
        assert count == 0
        assert out.exists()


class TestLoadCandidatesFromJsonl:
    """Tests for load_candidates_from_jsonl()."""

    def test_loads_records(self, tmp_path: Path) -> None:
        jsonl = tmp_path / "candidates.jsonl"
        records = [
            {
                "arxiv_id": "2401.12345",
                "version": "v1",
                "title": "Paper A",
                "authors": ["Author One"],
                "published": "2024-01-15T00:00:00Z",
                "updated": "2024-01-15T00:00:00Z",
                "categories": ["cs.CL"],
                "primary_category": "cs.CL",
                "abstract": "Abstract A.",
                "abs_url": "https://arxiv.org/abs/2401.12345",
                "pdf_url": "https://arxiv.org/pdf/2401.12345",
                "source": "arxiv",
            },
            {
                "arxiv_id": "2402.99999",
                "version": "v1",
                "title": "Paper B",
                "authors": [],
                "published": "2024-02-01T00:00:00Z",
                "updated": "2024-02-01T00:00:00Z",
                "categories": ["cs.AI"],
                "primary_category": "cs.AI",
                "abstract": "Abstract B.",
                "abs_url": "https://arxiv.org/abs/2402.99999",
                "pdf_url": "https://arxiv.org/pdf/2402.99999",
                "source": "scholar",
            },
        ]
        jsonl.write_text(
            "\n".join(json.dumps(r) for r in records) + "\n",
            encoding="utf-8",
        )
        loaded = load_candidates_from_jsonl(jsonl)
        assert len(loaded) == 2
        assert loaded[0].title == "Paper A"
        assert loaded[1].title == "Paper B"

    def test_skips_blank_lines(self, tmp_path: Path) -> None:
        jsonl = tmp_path / "candidates.jsonl"
        record = json.dumps(
            {
                "arxiv_id": "2401.00001",
                "version": "v1",
                "title": "X",
                "authors": [],
                "published": "2024-01-01T00:00:00Z",
                "updated": "2024-01-01T00:00:00Z",
                "categories": ["cs.CL"],
                "primary_category": "cs.CL",
                "abstract": "Test.",
                "abs_url": "https://arxiv.org/abs/2401.00001",
                "pdf_url": "https://arxiv.org/pdf/2401.00001",
                "source": "arxiv",
            }
        )
        jsonl.write_text(f"\n{record}\n\n", encoding="utf-8")
        loaded = load_candidates_from_jsonl(jsonl)
        assert len(loaded) == 1
