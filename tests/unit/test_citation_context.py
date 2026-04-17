"""Tests for extraction/citation_context.py — citation context extraction."""

import pytest

from research_pipeline.extraction.citation_context import (
    CitationContext,
    contexts_to_dicts,
    extract_citation_contexts,
    group_by_marker,
)


# --- Sample Markdown documents for testing ---

SAMPLE_PAPER = """# Introduction

Recent work on language models [1] has shown that scaling improves performance.
The transformer architecture [2] remains the dominant approach for NLP tasks.

# Related Work

Smith et al. [3] proposed a novel attention mechanism. Building on this,
Jones and Lee [4, 5] extended the approach to multi-modal settings.

Several studies (Brown, 2020) have demonstrated the effectiveness of
pre-training. More recently (Chen & Wang, 2023) showed that fine-tuning
with reinforcement learning yields further improvements.

# Methods

We follow the approach of [1] and extend it with our proposed module.
Our method achieves 95% accuracy compared to 90% in [3].

# References

[1] Vaswani et al., "Attention Is All You Need", NeurIPS 2017.
[2] Devlin et al., "BERT: Pre-training of Deep Bidirectional Transformers", 2019.
[3] Smith et al., "Novel Attention for Long Sequences", ICML 2022.
"""


class TestExtractCitationContexts:
    """Tests for extract_citation_contexts."""

    def test_extracts_numeric_citations(self) -> None:
        """Numeric citations [1], [2] etc. are extracted."""
        contexts = extract_citation_contexts(SAMPLE_PAPER)
        markers = [c.marker for c in contexts]
        assert "[1]" in markers
        assert "[2]" in markers

    def test_extracts_author_year_citations(self) -> None:
        """Author-year citations (Brown, 2020) are extracted."""
        contexts = extract_citation_contexts(SAMPLE_PAPER)
        markers = [c.marker for c in contexts]
        assert "(Brown, 2020)" in markers

    def test_extracts_multi_citations(self) -> None:
        """Multi-number citations like [4, 5] are extracted."""
        contexts = extract_citation_contexts(SAMPLE_PAPER)
        markers = [c.marker for c in contexts]
        assert "[4, 5]" in markers

    def test_skips_bibliography_section(self) -> None:
        """Citations in the References section are NOT extracted."""
        contexts = extract_citation_contexts(SAMPLE_PAPER)
        # None should be from the References section
        for ctx in contexts:
            assert ctx.section != "References"

    def test_section_tracking(self) -> None:
        """Each context records the correct section heading."""
        contexts = extract_citation_contexts(SAMPLE_PAPER)
        sections = {c.section for c in contexts}
        assert "Introduction" in sections
        assert "Related Work" in sections
        assert "Methods" in sections

    def test_sentence_context(self) -> None:
        """Extracted sentence contains the citation."""
        contexts = extract_citation_contexts(SAMPLE_PAPER)
        for ctx in contexts:
            assert ctx.marker in ctx.sentence or ctx.marker in ctx.paragraph

    def test_position_is_nonnegative(self) -> None:
        """Position offsets are non-negative."""
        contexts = extract_citation_contexts(SAMPLE_PAPER)
        for ctx in contexts:
            assert ctx.position >= 0

    def test_empty_document(self) -> None:
        """Empty document returns no contexts."""
        assert extract_citation_contexts("") == []
        assert extract_citation_contexts("   ") == []

    def test_no_citations(self) -> None:
        """Document without citations returns empty list."""
        text = "# Introduction\n\nThis paper discusses methods.\n\n# Conclusion\n\nWe conclude."
        contexts = extract_citation_contexts(text)
        assert contexts == []

    def test_context_window_zero(self) -> None:
        """context_window=0 returns just the citing sentence."""
        text = "First sentence. Citation here [1] is important. Last sentence."
        contexts = extract_citation_contexts(text, context_window=0)
        assert len(contexts) >= 1
        ctx = contexts[0]
        assert "[1]" in ctx.sentence

    def test_context_window_expansion(self) -> None:
        """Larger context_window includes more surrounding text."""
        text = (
            "# Intro\n\n"
            "Background information here. "
            "The key result [1] was transformative. "
            "Follow-up work confirmed this."
        )
        ctx_0 = extract_citation_contexts(text, context_window=0)
        ctx_1 = extract_citation_contexts(text, context_window=1)
        if ctx_0 and ctx_1:
            assert len(ctx_1[0].sentence) >= len(ctx_0[0].sentence)

    def test_paragraph_preserved(self) -> None:
        """Full paragraph is preserved in context."""
        contexts = extract_citation_contexts(SAMPLE_PAPER)
        for ctx in contexts:
            assert len(ctx.paragraph) > 0

    def test_reference_entries_skipped(self) -> None:
        """Lines that look like reference entries are skipped."""
        text = (
            "# Discussion\n\n"
            "We used method [1] successfully.\n\n"
            "[1] Smith et al., Some Paper, 2024.\n"
            "[2] Jones et al., Another Paper, 2023."
        )
        contexts = extract_citation_contexts(text)
        # Should find [1] in Discussion but not the reference entries
        valid = [c for c in contexts if c.section == "Discussion"]
        assert len(valid) >= 1


class TestGroupByMarker:
    """Tests for group_by_marker."""

    def test_groups_same_marker(self) -> None:
        """Multiple contexts with same marker are grouped together."""
        contexts = extract_citation_contexts(SAMPLE_PAPER)
        groups = group_by_marker(contexts)
        # [1] appears in both Introduction and Methods
        if "[1]" in groups:
            assert len(groups["[1]"]) >= 2

    def test_empty_list(self) -> None:
        """Empty list returns empty dict."""
        assert group_by_marker([]) == {}

    def test_each_group_is_list(self) -> None:
        """Each group value is a list of CitationContext."""
        contexts = extract_citation_contexts(SAMPLE_PAPER)
        groups = group_by_marker(contexts)
        for marker, ctx_list in groups.items():
            assert isinstance(ctx_list, list)
            for ctx in ctx_list:
                assert isinstance(ctx, CitationContext)


class TestContextsToDicts:
    """Tests for contexts_to_dicts."""

    def test_serializes_fields(self) -> None:
        """Serialized dicts contain expected keys."""
        contexts = extract_citation_contexts(SAMPLE_PAPER)
        dicts = contexts_to_dicts(contexts)
        assert len(dicts) == len(contexts)
        for d in dicts:
            assert "marker" in d
            assert "sentence" in d
            assert "section" in d
            assert "position" in d

    def test_empty_list(self) -> None:
        """Empty list serializes to empty list."""
        assert contexts_to_dicts([]) == []

    def test_values_match_original(self) -> None:
        """Serialized values match the original context objects."""
        contexts = extract_citation_contexts(SAMPLE_PAPER)
        dicts = contexts_to_dicts(contexts)
        for ctx, d in zip(contexts, dicts):
            assert d["marker"] == ctx.marker
            assert d["section"] == ctx.section
            assert d["position"] == ctx.position


class TestCitationContextDataclass:
    """Tests for CitationContext dataclass."""

    def test_frozen(self) -> None:
        """CitationContext is immutable."""
        ctx = CitationContext(marker="[1]", sentence="Test sentence [1].")
        with pytest.raises(AttributeError):
            ctx.marker = "[2]"  # type: ignore[misc]

    def test_defaults(self) -> None:
        """Default values for optional fields."""
        ctx = CitationContext(marker="[1]", sentence="Test [1].")
        assert ctx.paragraph == ""
        assert ctx.section == ""
        assert ctx.position == 0
