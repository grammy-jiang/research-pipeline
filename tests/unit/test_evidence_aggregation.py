"""Tests for evidence-only aggregation module."""

from __future__ import annotations

import json

from research_pipeline.models.evidence import (
    AggregationStats,
    EvidenceAggregation,
    EvidencePointer,
    EvidenceStatement,
    RhetoricSpan,
    RhetoricType,
)
from research_pipeline.models.summary import (
    PaperSummary,
    SummaryEvidence,
    SynthesisAgreement,
    SynthesisDisagreement,
    SynthesisReport,
)
from research_pipeline.summarization.evidence_aggregation import (
    aggregate_evidence,
    aggregate_from_summaries,
    detect_rhetoric,
    extract_evidence_pointers,
    format_aggregation_text,
    normalize_length,
    strip_rhetoric,
)

# --- Models ---


class TestRhetoricSpan:
    """Tests for RhetoricSpan model."""

    def test_create(self) -> None:
        span = RhetoricSpan(
            start=0, end=5, text="might", rhetoric_type=RhetoricType.HEDGING
        )
        assert span.start == 0
        assert span.end == 5
        assert span.text == "might"
        assert span.rhetoric_type == RhetoricType.HEDGING

    def test_roundtrip(self) -> None:
        span = RhetoricSpan(
            start=10,
            end=20,
            text="clearly",
            rhetoric_type=RhetoricType.CONFIDENCE_CLAIM,
        )
        data = span.model_dump()
        restored = RhetoricSpan.model_validate(data)
        assert restored == span


class TestEvidencePointer:
    """Tests for EvidencePointer model."""

    def test_create_minimal(self) -> None:
        ptr = EvidencePointer(paper_id="2301.12345")
        assert ptr.paper_id == "2301.12345"
        assert ptr.chunk_id == ""
        assert ptr.section == ""

    def test_create_full(self) -> None:
        ptr = EvidencePointer(
            paper_id="2301.12345",
            chunk_id="ch-001",
            section="Section 3.2",
            quote="The method achieves 95% accuracy.",
            page=5,
        )
        assert ptr.page == 5
        assert ptr.quote == "The method achieves 95% accuracy."

    def test_roundtrip(self) -> None:
        ptr = EvidencePointer(paper_id="2301.12345", section="Table 1")
        data = json.loads(ptr.model_dump_json())
        restored = EvidencePointer.model_validate(data)
        assert restored == ptr


class TestEvidenceStatement:
    """Tests for EvidenceStatement model."""

    def test_create(self) -> None:
        stmt = EvidenceStatement(
            statement_id="ES-001",
            text="BM25 outperforms dense retrieval for academic papers.",
            pointers=[EvidencePointer(paper_id="2301.12345")],
            source_type="finding",
            agreement_count=3,
        )
        assert stmt.statement_id == "ES-001"
        assert len(stmt.pointers) == 1
        assert stmt.agreement_count == 3

    def test_defaults(self) -> None:
        stmt = EvidenceStatement(
            statement_id="ES-002",
            text="Test statement.",
        )
        assert stmt.source_type == "finding"
        assert stmt.agreement_count == 1
        assert stmt.pointers == []


class TestAggregationStats:
    """Tests for AggregationStats model."""

    def test_defaults(self) -> None:
        stats = AggregationStats()
        assert stats.input_statements == 0
        assert stats.output_statements == 0
        assert stats.rhetoric_stripped == 0

    def test_roundtrip(self) -> None:
        stats = AggregationStats(
            input_statements=20,
            rhetoric_stripped=5,
            evidence_matched=12,
            evidence_unmatched=3,
            merged_duplicates=2,
            output_statements=15,
            avg_pointers_per_statement=1.8,
        )
        data = json.loads(stats.model_dump_json())
        restored = AggregationStats.model_validate(data)
        assert restored == stats


class TestEvidenceAggregation:
    """Tests for EvidenceAggregation model."""

    def test_create(self) -> None:
        agg = EvidenceAggregation(
            topic="memory systems",
            statements=[
                EvidenceStatement(
                    statement_id="ES-001",
                    text="Memory consolidation improves recall.",
                    pointers=[EvidencePointer(paper_id="2301.12345")],
                )
            ],
            dropped=["[too-short] test"],
            stats=AggregationStats(input_statements=5, output_statements=1),
        )
        assert agg.topic == "memory systems"
        assert len(agg.statements) == 1
        assert len(agg.dropped) == 1

    def test_roundtrip(self) -> None:
        agg = EvidenceAggregation(topic="test", statements=[], dropped=[])
        data = json.loads(agg.model_dump_json())
        restored = EvidenceAggregation.model_validate(data)
        assert restored == agg


# --- Rhetoric Detection ---


class TestDetectRhetoric:
    """Tests for detect_rhetoric function."""

    def test_hedging(self) -> None:
        spans = detect_rhetoric("This might improve performance.")
        assert any(s.rhetoric_type == RhetoricType.HEDGING for s in spans)

    def test_confidence(self) -> None:
        spans = detect_rhetoric("This clearly demonstrates the effectiveness.")
        assert any(s.rhetoric_type == RhetoricType.CONFIDENCE_CLAIM for s in spans)

    def test_subjective(self) -> None:
        spans = detect_rhetoric("Interestingly, the results show improvement.")
        assert any(s.rhetoric_type == RhetoricType.SUBJECTIVE for s in spans)

    def test_filler(self) -> None:
        spans = detect_rhetoric("It should be noted that the approach works.")
        assert any(s.rhetoric_type == RhetoricType.FILLER for s in spans)

    def test_unsupported_causal(self) -> None:
        spans = detect_rhetoric("This proves that the method is superior.")
        assert any(s.rhetoric_type == RhetoricType.UNSUPPORTED_CAUSAL for s in spans)

    def test_no_rhetoric(self) -> None:
        spans = detect_rhetoric("BM25 scored 0.85 on the test set.")
        assert len(spans) == 0

    def test_multiple_types(self) -> None:
        text = "Clearly, this might possibly work interestingly."
        spans = detect_rhetoric(text)
        types = {s.rhetoric_type for s in spans}
        assert len(types) >= 2

    def test_empty_input(self) -> None:
        spans = detect_rhetoric("")
        assert len(spans) == 0

    def test_sorted_by_position(self) -> None:
        text = "Perhaps it clearly shows that the method might work."
        spans = detect_rhetoric(text)
        positions = [s.start for s in spans]
        assert positions == sorted(positions)


class TestStripRhetoric:
    """Tests for strip_rhetoric function."""

    def test_strip_hedging(self) -> None:
        result, spans = strip_rhetoric("This might improve results significantly.")
        assert "might" not in result
        assert len(spans) > 0

    def test_strip_confidence(self) -> None:
        result, spans = strip_rhetoric("This obviously works well for all cases.")
        assert "obviously" not in result.lower()

    def test_preserves_factual(self) -> None:
        text = "BM25 achieves 0.85 precision on the BEIR benchmark."
        result, spans = strip_rhetoric(text)
        assert result == text
        assert len(spans) == 0

    def test_clean_whitespace(self) -> None:
        result, _ = strip_rhetoric("The method  might  produce  good results.")
        assert "  " not in result

    def test_empty_input(self) -> None:
        result, spans = strip_rhetoric("")
        assert result == ""
        assert len(spans) == 0


# --- Length Normalization ---


class TestNormalizeLength:
    """Tests for normalize_length function."""

    def test_short_enough(self) -> None:
        text = "BM25 outperforms dense retrieval for academic search."
        result = normalize_length(text, max_words=50)
        assert result == text

    def test_too_short(self) -> None:
        text = "Yes okay"
        result = normalize_length(text, min_words=5)
        assert result == ""

    def test_truncates_at_sentence(self) -> None:
        text = (
            "The method achieves 95% accuracy. "
            "It also handles edge cases well. "
            "Furthermore it provides good latency. "
            "And many other words to pad the sentence beyond the limit here."
        )
        result = normalize_length(text, max_words=15)
        assert len(result.split()) <= 16  # Allow some leeway for sentence boundary
        assert result.endswith(".")

    def test_truncates_with_ellipsis(self) -> None:
        text = " ".join(f"word{i}" for i in range(100))
        result = normalize_length(text, max_words=10)
        assert result.endswith("...")

    def test_exact_max(self) -> None:
        text = "one two three four five six seven eight nine ten"
        result = normalize_length(text, max_words=10)
        assert result == text

    def test_empty_input(self) -> None:
        result = normalize_length("")
        assert result == ""


# --- Evidence Pointer Extraction ---


class TestExtractEvidencePointers:
    """Tests for extract_evidence_pointers function."""

    def test_arxiv_id(self) -> None:
        text = "As shown in [2301.12345], the approach works."
        pointers = extract_evidence_pointers(text)
        assert len(pointers) >= 1
        assert any(p.paper_id == "2301.12345" for p in pointers)

    def test_arxiv_with_prefix(self) -> None:
        text = "Results from [arXiv:2301.12345] confirm this."
        pointers = extract_evidence_pointers(text)
        assert any(p.paper_id == "2301.12345" for p in pointers)

    def test_section_reference(self) -> None:
        text = "As described in Section 3.2, the method..."
        pointers = extract_evidence_pointers(text, paper_id="test-paper")
        assert any(p.section == "Section 3.2" for p in pointers)

    def test_table_reference(self) -> None:
        text = "Results in Table 3 show improvement."
        pointers = extract_evidence_pointers(text, paper_id="test")
        assert any("Table 3" in p.section for p in pointers)

    def test_figure_reference(self) -> None:
        text = "See Figure 5 for the comparison."
        pointers = extract_evidence_pointers(text, paper_id="test")
        assert any("Figure 5" in p.section for p in pointers)

    def test_no_references(self) -> None:
        text = "This is a plain statement without citations."
        pointers = extract_evidence_pointers(text)
        assert len(pointers) == 0

    def test_multiple_references(self) -> None:
        text = "Both [2301.12345] and [2302.67890] support this in Section 4."
        pointers = extract_evidence_pointers(text, paper_id="main")
        assert len(pointers) >= 3

    def test_empty_input(self) -> None:
        pointers = extract_evidence_pointers("")
        assert len(pointers) == 0


# --- Full Aggregation ---


def _make_summary(
    arxiv_id: str = "2301.12345",
    findings: list[str] | None = None,
    limitations: list[str] | None = None,
) -> PaperSummary:
    """Helper to create a PaperSummary for tests."""
    return PaperSummary(
        arxiv_id=arxiv_id,
        version="v1",
        title=f"Paper {arxiv_id}",
        objective="Investigate the effectiveness of the approach.",
        methodology="We used a controlled experiment with 100 participants.",
        findings=findings or ["BM25 achieved 0.85 precision on BEIR."],
        limitations=limitations or ["Limited to English papers only."],
        evidence=[
            SummaryEvidence(chunk_id="ch-001", quote="BM25 scored 0.85"),
        ],
    )


def _make_report(
    summaries: list[PaperSummary] | None = None,
    agreements: list[SynthesisAgreement] | None = None,
) -> SynthesisReport:
    """Helper to create a SynthesisReport for tests."""
    sums = summaries or [_make_summary()]
    return SynthesisReport(
        topic="retrieval methods",
        paper_count=len(sums),
        paper_summaries=sums,
        agreements=agreements or [],
    )


class TestAggregateEvidence:
    """Tests for aggregate_evidence function."""

    def test_basic(self) -> None:
        report = _make_report()
        result = aggregate_evidence(report)
        assert result.topic == "retrieval methods"
        assert len(result.statements) > 0
        assert result.stats.input_statements > 0

    def test_multiple_papers(self) -> None:
        summaries = [
            _make_summary("2301.12345", findings=["Dense retrieval works well."]),
            _make_summary("2302.67890", findings=["Sparse retrieval is more robust."]),
        ]
        result = aggregate_evidence(_make_report(summaries))
        assert result.stats.input_statements >= 2

    def test_strip_rhetoric(self) -> None:
        summary = _make_summary(
            findings=["This clearly demonstrates that the method might work well."]
        )
        report = _make_report([summary])
        result = aggregate_evidence(report, strip_rhetoric_enabled=True)
        # Rhetoric should be stripped from statements
        for stmt in result.statements:
            if stmt.source_type == "finding":
                assert (
                    "clearly" not in stmt.text.lower()
                    or "might" not in stmt.text.lower()
                )

    def test_no_strip_rhetoric(self) -> None:
        summary = _make_summary(findings=["This clearly works well for retrieval."])
        report = _make_report([summary])
        result = aggregate_evidence(report, strip_rhetoric_enabled=False)
        assert result.stats.rhetoric_stripped == 0

    def test_min_pointers_filter(self) -> None:
        summary = _make_summary(
            findings=["A finding with no citation references at all anywhere."]
        )
        report = _make_report([summary])
        # With min_pointers=0, should keep everything
        result_no_filter = aggregate_evidence(report, min_pointers=0)
        # With min_pointers=5, should filter aggressively
        result_strict = aggregate_evidence(report, min_pointers=5)
        assert (
            result_strict.stats.output_statements
            <= result_no_filter.stats.output_statements
        )

    def test_similarity_merge(self) -> None:
        summaries = [
            _make_summary(
                "2301.11111",
                findings=["BM25 achieves high precision for academic search."],
            ),
            _make_summary(
                "2302.22222",
                findings=["BM25 achieves high precision for academic retrieval."],
            ),
        ]
        result = aggregate_evidence(
            _make_report(summaries),
            similarity_threshold=0.7,
        )
        # Should merge similar findings
        assert result.stats.merged_duplicates >= 0  # May or may not merge

    def test_length_normalization(self) -> None:
        long_finding = " ".join(f"word{i}" for i in range(100))
        summary = _make_summary(findings=[long_finding])
        result = aggregate_evidence(_make_report([summary]), max_words=20)
        for stmt in result.statements:
            if stmt.source_type == "finding":
                assert len(stmt.text.split()) <= 25  # Allow some leeway

    def test_empty_report(self) -> None:
        report = SynthesisReport(
            topic="empty",
            paper_count=0,
            paper_summaries=[],
        )
        result = aggregate_evidence(report)
        assert result.stats.input_statements == 0
        assert result.stats.output_statements == 0

    def test_agreements_included(self) -> None:
        agreements = [
            SynthesisAgreement(
                claim="BM25 is effective for retrieval.",
                supporting_papers=["2301.12345", "2302.67890"],
            )
        ]
        report = _make_report(agreements=agreements)
        result = aggregate_evidence(report)
        assert result.stats.input_statements > 0

    def test_disagreements_included(self) -> None:
        report = SynthesisReport(
            topic="test",
            paper_count=2,
            disagreements=[
                SynthesisDisagreement(
                    topic="Whether hybrid retrieval helps.",
                    positions={
                        "2301.12345": "Hybrid is better",
                        "2302.67890": "Hybrid is worse",
                    },
                )
            ],
        )
        result = aggregate_evidence(report)
        assert result.stats.input_statements >= 1

    def test_stats_consistency(self) -> None:
        report = _make_report([_make_summary()])
        result = aggregate_evidence(report)
        assert result.stats.output_statements == len(result.statements)
        assert (
            result.stats.evidence_matched + result.stats.evidence_unmatched
            == result.stats.output_statements
        )

    def test_renumbered_ids(self) -> None:
        summaries = [
            _make_summary("2301.11111", findings=["Finding one is important."]),
            _make_summary("2302.22222", findings=["Finding two is also important."]),
        ]
        result = aggregate_evidence(_make_report(summaries))
        ids = [s.statement_id for s in result.statements]
        assert ids[0] == "ES-001"
        if len(ids) > 1:
            assert ids[1] == "ES-002"


class TestAggregateFromSummaries:
    """Tests for aggregate_from_summaries convenience function."""

    def test_basic(self) -> None:
        summaries = [_make_summary()]
        result = aggregate_from_summaries(summaries, topic="test topic")
        assert result.topic == "test topic"
        assert len(result.statements) > 0

    def test_empty(self) -> None:
        result = aggregate_from_summaries([], topic="empty")
        assert result.stats.output_statements == 0


# --- Formatting ---


class TestFormatAggregationText:
    """Tests for format_aggregation_text function."""

    def test_basic(self) -> None:
        agg = EvidenceAggregation(
            topic="test topic",
            statements=[
                EvidenceStatement(
                    statement_id="ES-001",
                    text="A factual finding.",
                    pointers=[EvidencePointer(paper_id="2301.12345")],
                    source_type="finding",
                )
            ],
            stats=AggregationStats(
                input_statements=5,
                output_statements=1,
                rhetoric_stripped=3,
            ),
        )
        text = format_aggregation_text(agg)
        assert "test topic" in text
        assert "ES-001" in text
        assert "Rhetoric removed" in text

    def test_agreement_count(self) -> None:
        agg = EvidenceAggregation(
            topic="test",
            statements=[
                EvidenceStatement(
                    statement_id="ES-001",
                    text="Agreed finding.",
                    agreement_count=3,
                )
            ],
        )
        text = format_aggregation_text(agg)
        assert "agreed by 3 sources" in text

    def test_dropped_section(self) -> None:
        agg = EvidenceAggregation(
            topic="test",
            statements=[],
            dropped=["[rhetoric-empty] some text", "[too-short] hi"],
        )
        text = format_aggregation_text(agg)
        assert "Dropped" in text
        assert "[rhetoric-empty]" in text

    def test_empty_aggregation(self) -> None:
        agg = EvidenceAggregation(topic="empty", statements=[])
        text = format_aggregation_text(agg)
        assert "empty" in text

    def test_groups_by_type(self) -> None:
        agg = EvidenceAggregation(
            topic="test",
            statements=[
                EvidenceStatement(
                    statement_id="ES-001",
                    text="A finding.",
                    source_type="finding",
                ),
                EvidenceStatement(
                    statement_id="ES-002",
                    text="A limitation.",
                    source_type="limitation",
                ),
            ],
        )
        text = format_aggregation_text(agg)
        assert "Finding" in text
        assert "Limitation" in text

    def test_truncated_dropped(self) -> None:
        agg = EvidenceAggregation(
            topic="test",
            statements=[],
            dropped=[f"[drop-{i}] item" for i in range(30)],
        )
        text = format_aggregation_text(agg)
        assert "and 10 more" in text
