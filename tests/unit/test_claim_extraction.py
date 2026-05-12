"""Unit tests for heuristic claim extraction (extraction.claims module)."""

from __future__ import annotations

from pathlib import Path

from research_pipeline.extraction.claims import (
    _deduplicate,
    _is_claim_sentence,
    _section_confidence,
    _split_sentences,
    extract_claims_heuristic,
)
from research_pipeline.extraction.extractor import extract_from_markdown
from research_pipeline.models.extraction import ChunkMetadata

# ---------------------------------------------------------------------------
# Helper fixtures
# ---------------------------------------------------------------------------

ABSTRACT_CHUNK = (
    ChunkMetadata(
        paper_id="2401.00001",
        section_path="Abstract",
        chunk_id="chunk_abstract_0",
        source_span="lines 1-5",
        token_count=60,
    ),
    (
        "We propose a transformer architecture achieving state-of-the-art results "
        "on multiple benchmarks. Our model reduces inference latency by 40% "
        "while maintaining accuracy. The approach shows significant improvements "
        "over existing baselines across five evaluation datasets."
    ),
)

CONCLUSION_CHUNK = (
    ChunkMetadata(
        paper_id="2401.00001",
        section_path="Conclusion",
        chunk_id="chunk_conclusion_0",
        source_span="lines 100-110",
        token_count=55,
    ),
    (
        "This paper demonstrates sparse attention with learned positional encodings "
        "improves performance on long-sequence tasks. Our results confirm that "
        "the proposed method outperforms all baseline models by at least 5 BLEU. "
        "Future work will explore applying this approach to multilingual settings."
    ),
)

METHODS_CHUNK = (
    ChunkMetadata(
        paper_id="2401.00001",
        section_path="Methods",
        chunk_id="chunk_methods_0",
        source_span="lines 20-40",
        token_count=80,
    ),
    (
        "We use Adam optimizer with a learning rate of 0.001 and batch size 32. "
        "The model is trained for 100 epochs on a single GPU. "
        "For the loss function, we apply cross-entropy on the token-level predictions."
    ),
)


class TestSplitSentences:
    def test_basic_split(self) -> None:
        text = (
            "The model achieves high accuracy on all benchmark datasets. "
            "It generalizes well to unseen evaluation examples in practice. "
            "We confirm this across five independent experimental runs."
        )
        sentences = _split_sentences(text)
        assert len(sentences) >= 2

    def test_removes_citations(self) -> None:
        text = (
            "This approach improves results [1, 2] significantly. See [3] for details."
        )
        sentences = _split_sentences(text)
        assert all("[" not in s for s in sentences)

    def test_removes_urls(self) -> None:
        text = (
            "Code is available at https://github.com/example/repo for reproducibility."
        )
        sentences = _split_sentences(text)
        assert all("https://" not in s for s in sentences)

    def test_empty_text(self) -> None:
        sentences = _split_sentences("")
        assert sentences == []


class TestIsClaimSentence:
    def test_valid_assertion(self) -> None:
        assert _is_claim_sentence("Our approach achieves state-of-the-art performance.")

    def test_valid_negative_result(self) -> None:
        assert _is_claim_sentence(
            "The model reduces latency by 40% while maintaining accuracy."
        )

    def test_skips_questions(self) -> None:
        assert not _is_claim_sentence("Can we improve performance?")

    def test_skips_we_describe(self) -> None:
        assert not _is_claim_sentence(
            "We describe the experimental setup in this section."
        )

    def test_skips_in_this_paper(self) -> None:
        assert not _is_claim_sentence("In this paper we present a new approach.")

    def test_skips_too_short(self) -> None:
        assert not _is_claim_sentence("Short sentence.")

    def test_skips_too_long(self) -> None:
        long = "This is a claim. " * 30
        assert not _is_claim_sentence(long)


class TestSectionConfidence:
    def test_abstract_high(self) -> None:
        assert _section_confidence("Abstract") == 0.85

    def test_conclusion_high(self) -> None:
        assert _section_confidence("Conclusion") > 0.75

    def test_results_medium_high(self) -> None:
        assert _section_confidence("Results") >= 0.70

    def test_unknown_section_low(self) -> None:
        assert _section_confidence("Related Work") == 0.50

    def test_case_insensitive(self) -> None:
        assert _section_confidence("ABSTRACT") == _section_confidence("abstract")


class TestDeduplicate:
    def test_removes_near_duplicates(self) -> None:
        candidates = [
            ("Our model achieves high performance on benchmarks", 0.8, "c1"),
            ("Our model achieves high performance on benchmarks today", 0.75, "c2"),
            ("Completely different statement about inference speed", 0.7, "c3"),
        ]
        result = _deduplicate(candidates)
        assert len(result) == 2

    def test_keeps_distinct_claims(self) -> None:
        candidates = [
            ("The model reduces latency significantly", 0.8, "c1"),
            ("Accuracy is maintained at high levels", 0.75, "c2"),
            ("Training converges in fewer epochs", 0.7, "c3"),
        ]
        result = _deduplicate(candidates)
        assert len(result) == 3

    def test_empty_input(self) -> None:
        assert _deduplicate([]) == []


class TestExtractClaimsHeuristic:
    def test_returns_claims_from_abstract(self) -> None:
        chunks = [ABSTRACT_CHUNK, METHODS_CHUNK]
        claims = extract_claims_heuristic(chunks, max_claims=10)
        assert len(claims) >= 1
        # All claims should have chunk IDs
        for claim in claims:
            assert len(claim.chunk_ids) >= 1

    def test_returns_claims_from_conclusion(self) -> None:
        chunks = [METHODS_CHUNK, CONCLUSION_CHUNK]
        claims = extract_claims_heuristic(chunks, max_claims=10)
        assert len(claims) >= 1

    def test_skips_methods_only(self) -> None:
        # Methods section has no high-signal assertions
        chunks = [METHODS_CHUNK]
        claims = extract_claims_heuristic(chunks, max_claims=10)
        # May have 0 or few claims from methods (depends on sentences)
        # All returned claims must be valid ExtractedClaim
        for claim in claims:
            assert 0.0 <= claim.confidence <= 1.0
            assert len(claim.claim) > 0

    def test_max_claims_respected(self) -> None:
        chunks = [ABSTRACT_CHUNK, CONCLUSION_CHUNK, METHODS_CHUNK]
        claims = extract_claims_heuristic(chunks, max_claims=2)
        assert len(claims) <= 2

    def test_confidence_range(self) -> None:
        chunks = [ABSTRACT_CHUNK, CONCLUSION_CHUNK]
        claims = extract_claims_heuristic(chunks)
        for claim in claims:
            assert 0.0 <= claim.confidence <= 1.0

    def test_sorted_by_confidence(self) -> None:
        chunks = [ABSTRACT_CHUNK, CONCLUSION_CHUNK, METHODS_CHUNK]
        claims = extract_claims_heuristic(chunks)
        confidences = [c.confidence for c in claims]
        assert confidences == sorted(confidences, reverse=True)

    def test_empty_chunks(self) -> None:
        claims = extract_claims_heuristic([])
        assert claims == []

    def test_chunk_ids_valid(self) -> None:
        chunks = [ABSTRACT_CHUNK, CONCLUSION_CHUNK]
        all_chunk_ids = {meta.chunk_id for meta, _ in chunks}
        claims = extract_claims_heuristic(chunks)
        for claim in claims:
            for cid in claim.chunk_ids:
                assert cid in all_chunk_ids


class TestExtractFromMarkdown:
    def test_produces_claims(self, tmp_path: Path) -> None:
        md = tmp_path / "paper.md"
        md.write_text(
            "# Abstract\n\n"
            "We propose a transformer model achieving state-of-the-art results. "
            "Our approach shows 40% latency reduction with comparable accuracy.\n\n"
            "# Methods\n\n"
            "We train with Adam optimizer for 100 epochs on GPU.\n\n"
            "# Results\n\n"
            "Our model outperforms baselines by 5 BLEU points on all benchmarks.\n\n"
            "# Conclusion\n\n"
            "This work confirms sparse attention improves performance significantly.",
            encoding="utf-8",
        )
        result = extract_from_markdown(md, "2401.00001", "v1")
        assert len(result.chunks) >= 1
        assert len(result.sections) >= 1
        # Heuristic extraction should produce at least 1 claim
        assert len(result.claims) >= 1
        for claim in result.claims:
            assert 0.0 <= claim.confidence <= 1.0
            assert len(claim.chunk_ids) >= 1

    def test_empty_markdown_no_claims(self, tmp_path: Path) -> None:
        md = tmp_path / "empty.md"
        md.write_text("", encoding="utf-8")
        result = extract_from_markdown(md, "2401.99999", "v1")
        assert result.claims == []

    def test_max_claims_parameter(self, tmp_path: Path) -> None:
        md = tmp_path / "paper.md"
        md.write_text(
            "# Abstract\n\n"
            + " ".join(
                [f"Our model achieves result {i} on benchmark {i}." for i in range(20)]
            )
            + "\n\n# Conclusion\n\n"
            + " ".join(
                [f"This demonstrates finding {i} with confidence." for i in range(20)]
            ),
            encoding="utf-8",
        )
        result = extract_from_markdown(md, "2401.00002", "v1", max_claims=5)
        assert len(result.claims) <= 5
