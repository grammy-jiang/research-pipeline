"""Tests for FACT citation verification scoring."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from research_pipeline.quality.fact_scoring import (
    FACTScore,
    _extract_citations,
    _normalize_title,
    compute_fact_score,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PAPER_IDS = [
    "2301.12345",
    "2302.67890",
    "2303.11111",
    "2304.22222",
    "2305.33333",
]

PAPER_TITLES = [
    "Attention Is All You Need",
    "BERT: Pre-training of Deep Bidirectional Transformers",
    "GPT-4 Technical Report",
    "Scaling Laws for Neural Language Models",
    "Chain-of-Thought Prompting Elicits Reasoning",
]


# ---------------------------------------------------------------------------
# FACTScore model tests
# ---------------------------------------------------------------------------


class TestFACTScoreModel:
    """Tests for the FACTScore Pydantic model."""

    def test_model_roundtrip(self) -> None:
        """Construct → serialize → deserialize → assert equal."""
        score = FACTScore(
            citation_accuracy=0.8,
            effective_citation_ratio=0.6,
            total_citations=5,
            verified_citations=4,
            unsupported_citations=["99"],
            uncited_papers=["2304.22222", "2305.33333"],
            citation_map={"1": "2301.12345", "2": "2302.67890"},
        )
        data = score.model_dump()
        restored = FACTScore.model_validate(data)
        assert restored == score

    def test_model_json_roundtrip(self) -> None:
        """JSON serialize → deserialize roundtrip."""
        score = FACTScore(
            citation_accuracy=1.0,
            effective_citation_ratio=1.0,
            total_citations=3,
            verified_citations=3,
            unsupported_citations=[],
            uncited_papers=[],
            citation_map={"1": "a", "2": "b", "3": "c"},
        )
        json_str = score.model_dump_json()
        restored = FACTScore.model_validate_json(json_str)
        assert restored == score

    def test_model_rejects_out_of_range(self) -> None:
        """Values outside [0, 1] should raise ValidationError."""
        with pytest.raises(ValidationError):
            FACTScore(
                citation_accuracy=1.5,
                effective_citation_ratio=0.5,
                total_citations=1,
                verified_citations=1,
            )

    def test_model_defaults(self) -> None:
        """Lists and dict should default to empty."""
        score = FACTScore(
            citation_accuracy=0.0,
            effective_citation_ratio=0.0,
            total_citations=0,
            verified_citations=0,
        )
        assert score.unsupported_citations == []
        assert score.uncited_papers == []
        assert score.citation_map == {}


# ---------------------------------------------------------------------------
# Citation extraction tests
# ---------------------------------------------------------------------------


class TestExtractCitations:
    """Tests for the internal _extract_citations helper."""

    def test_numeric_citations(self) -> None:
        """Numeric [n] references are detected."""
        text = "As shown in [1] and [2], the method works."
        cites = _extract_citations(text)
        refs = {ref for ref, _ in cites}
        assert "1" in refs
        assert "2" in refs

    def test_arxiv_citations(self) -> None:
        """ArXiv-style [YYMM.NNNNN] references are detected."""
        text = "See [2301.12345] for details."
        cites = _extract_citations(text)
        refs = {ref for ref, kind in cites if kind == "arxiv"}
        assert "2301.12345" in refs

    def test_author_year_citations(self) -> None:
        """Author et al., YYYY references are detected."""
        text = "According to [Vaswani et al., 2017], attention is key."
        cites = _extract_citations(text)
        refs = {ref for ref, kind in cites if kind == "author_year"}
        assert "Vaswani et al., 2017" in refs

    def test_deduplication(self) -> None:
        """Repeated citations are deduplicated."""
        text = "[1] shows X. As [1] also confirms, Y holds."
        cites = _extract_citations(text)
        refs = [ref for ref, _ in cites]
        assert refs.count("1") == 1


# ---------------------------------------------------------------------------
# compute_fact_score tests
# ---------------------------------------------------------------------------


class TestComputeFactScore:
    """Tests for the main compute_fact_score function."""

    def test_empty_text_no_citations(self) -> None:
        """Empty text yields perfect accuracy (vacuously true), 0% coverage."""
        result = compute_fact_score("", PAPER_IDS, PAPER_TITLES)
        assert result.citation_accuracy == 1.0
        assert result.effective_citation_ratio == 0.0
        assert result.total_citations == 0
        assert result.verified_citations == 0
        assert len(result.uncited_papers) == len(PAPER_IDS)

    def test_all_numeric_citations_verified(self) -> None:
        """All numeric citations map to valid papers."""
        text = "Results from [1], [2], [3], [4], and [5] confirm the hypothesis."
        result = compute_fact_score(text, PAPER_IDS, PAPER_TITLES)
        assert result.citation_accuracy == 1.0
        assert result.effective_citation_ratio == 1.0
        assert result.total_citations == 5
        assert result.verified_citations == 5
        assert result.unsupported_citations == []
        assert result.uncited_papers == []

    def test_some_unverified_numeric_citations(self) -> None:
        """Numeric citations beyond corpus size are unsupported."""
        text = "See [1], [2], and [99] for details."
        result = compute_fact_score(text, PAPER_IDS, PAPER_TITLES)
        assert result.verified_citations == 2
        assert result.total_citations == 3
        assert "99" in result.unsupported_citations
        assert result.citation_accuracy == pytest.approx(2 / 3, abs=0.001)

    def test_arxiv_citations_matched(self) -> None:
        """ArXiv-style citations match against paper_ids."""
        text = "As shown in [2301.12345] and [2303.11111], the approach works."
        result = compute_fact_score(text, PAPER_IDS, PAPER_TITLES)
        assert result.verified_citations == 2
        assert result.citation_accuracy == 1.0
        assert "2301.12345" in result.citation_map
        assert result.citation_map["2301.12345"] == "2301.12345"

    def test_arxiv_unmatched(self) -> None:
        """ArXiv citations not in the corpus are unsupported."""
        text = "See [9999.99999] for an unrelated study."
        result = compute_fact_score(text, PAPER_IDS, PAPER_TITLES)
        assert result.verified_citations == 0
        assert "9999.99999" in result.unsupported_citations

    def test_effective_citation_ratio_partial(self) -> None:
        """Only some papers are cited, ratio is partial."""
        text = "We refer to [1] and [3] throughout."
        result = compute_fact_score(text, PAPER_IDS, PAPER_TITLES)
        assert result.effective_citation_ratio == pytest.approx(2 / 5, abs=0.001)
        assert len(result.uncited_papers) == 3

    def test_uncited_papers_listed(self) -> None:
        """Uncited papers appear in the uncited_papers list."""
        text = "Only [1] is discussed."
        result = compute_fact_score(text, PAPER_IDS, PAPER_TITLES)
        assert "2302.67890" in result.uncited_papers
        assert "2301.12345" not in result.uncited_papers

    def test_duplicate_citations_counted_once(self) -> None:
        """Duplicate citation refs are deduplicated in extraction."""
        text = "[1] shows X. As [1] also confirms, Y."
        result = compute_fact_score(text, PAPER_IDS, PAPER_TITLES)
        assert result.total_citations == 1
        assert result.verified_citations == 1

    def test_mixed_citation_styles(self) -> None:
        """Numeric and arxiv citations in the same text."""
        text = "See [1] and [2303.11111] for evidence."
        result = compute_fact_score(text, PAPER_IDS, PAPER_TITLES)
        assert result.verified_citations == 2
        assert result.citation_accuracy == 1.0
        cited = set(result.citation_map.values())
        assert "2301.12345" in cited
        assert "2303.11111" in cited

    def test_empty_corpus(self) -> None:
        """Empty corpus with no text gives perfect scores (vacuous)."""
        result = compute_fact_score("", [], [])
        assert result.citation_accuracy == 1.0
        assert result.effective_citation_ratio == 1.0

    def test_empty_corpus_with_citations(self) -> None:
        """Citations against an empty corpus are all unsupported."""
        text = "See [1] and [2]."
        result = compute_fact_score(text, [], [])
        assert result.citation_accuracy == 0.0
        assert result.total_citations == 2
        assert result.verified_citations == 0

    def test_mismatched_ids_titles_raises(self) -> None:
        """Different-length paper_ids and paper_titles raise ValueError."""
        with pytest.raises(ValueError, match="same length"):
            compute_fact_score("text", ["a", "b"], ["title1"])

    def test_author_year_citation_match(self) -> None:
        """Author-year citation resolves via fuzzy title match."""
        ids = ["2301.00001"]
        titles = ["Vaswani Attention Model for Sequences"]
        text = "As noted in [Vaswani et al., 2017], attention helps."
        result = compute_fact_score(text, ids, titles)
        assert result.verified_citations == 1
        assert result.citation_map["Vaswani et al., 2017"] == "2301.00001"

    def test_normalize_title_helper(self) -> None:
        """Title normalization strips punctuation and lowercases."""
        assert _normalize_title("BERT: A Model!") == "bert a model"
        assert _normalize_title("  Spaces  ") == "spaces"
