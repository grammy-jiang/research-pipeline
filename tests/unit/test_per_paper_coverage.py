"""Tests for uncovered lines in summarization/per_paper.py."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from research_pipeline.llm.base import LLMProvider
from research_pipeline.models.extraction import ChunkMetadata
from research_pipeline.models.summary import (
    ConfidenceLevel,
    EvidenceSnippet,
    ExtractedStatement,
    PaperExtractionRecord,
    PaperSummary,
    StatementType,
)
from research_pipeline.summarization.per_paper import (
    _as_statement_items,
    _build_extraction_prompt,
    _coerce_confidence,
    _coerce_statement_type,
    _parse_extraction_response,
    _parse_statement_items,
    extract_paper,
    project_extraction_to_summary,
    score_extraction_quality,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_chunk(
    paper_id: str = "2301.00001",
    section: str = "Introduction",
    chunk_id: str = "c1",
    span: str = "lines 1-10",
    tokens: int = 100,
) -> ChunkMetadata:
    return ChunkMetadata(
        paper_id=paper_id,
        section_path=section,
        chunk_id=chunk_id,
        source_span=span,
        token_count=tokens,
    )


def _make_evidence(
    evidence_id: str = "E001",
    paper_id: str = "2301.00001",
    chunk_id: str = "c1",
) -> EvidenceSnippet:
    return EvidenceSnippet(
        evidence_id=evidence_id,
        paper_id=paper_id,
        chunk_id=chunk_id,
        line_range="lines 1-10",
        section="Introduction",
        quote="sample quote",
    )


def _make_statement(
    sid: str = "p:cat:001",
    text: str = "A finding",
    category: str = "results",
    evidence_ids: list[str] | None = None,
    st_type: StatementType = StatementType.AUTHOR_CLAIM,
    confidence: ConfidenceLevel = ConfidenceLevel.MEDIUM,
) -> ExtractedStatement:
    return ExtractedStatement(
        statement_id=sid,
        statement=text,
        category=category,
        statement_type=st_type,
        confidence=confidence,
        evidence_ids=evidence_ids or [],
    )


def _make_mock_provider(
    return_value: dict | Exception,  # type: ignore[type-arg]
) -> LLMProvider:
    provider = MagicMock(spec=LLMProvider)
    provider.model_name.return_value = "mock-model"
    if isinstance(return_value, Exception):
        provider.call.side_effect = return_value
    else:
        provider.call.return_value = return_value
    return provider


# ---------------------------------------------------------------------------
# _build_extraction_prompt (lines 210-230)
# ---------------------------------------------------------------------------


class TestBuildExtractionPrompt:
    def test_basic_prompt(self) -> None:
        chunks = [(_make_chunk(chunk_id="c1"), "Text of chunk 1", 1.5)]
        evidence = [_make_evidence(evidence_id="E001", chunk_id="c1")]
        prompt = _build_extraction_prompt(
            "My Paper", ["topic1", "topic2"], chunks, evidence
        )
        assert "My Paper" in prompt
        assert "topic1, topic2" in prompt
        assert "[E001] [Chunk 1]" in prompt
        assert "Text of chunk 1" in prompt
        assert "E001: c1" in prompt

    def test_multiple_chunks(self) -> None:
        chunks = [
            (_make_chunk(chunk_id="c1"), "First", 2.0),
            (_make_chunk(chunk_id="c2"), "Second", 0.5),
        ]
        evidence = [
            _make_evidence(evidence_id="E001", chunk_id="c1"),
            _make_evidence(evidence_id="E002", chunk_id="c2"),
        ]
        prompt = _build_extraction_prompt("T", ["t"], chunks, evidence)
        assert "[E001] [Chunk 1]" in prompt
        assert "[E002] [Chunk 2]" in prompt
        assert "---" in prompt


# ---------------------------------------------------------------------------
# _coerce_confidence (lines 233-242)
# ---------------------------------------------------------------------------


class TestCoerceConfidence:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("HIGH", ConfidenceLevel.HIGH),
            ("MEDIUM", ConfidenceLevel.MEDIUM),
            ("LOW", ConfidenceLevel.LOW),
            ("high", ConfidenceLevel.HIGH),
            ("Hmmm", ConfidenceLevel.HIGH),
            ("Lightly", ConfidenceLevel.LOW),
            (None, ConfidenceLevel.MEDIUM),
            ("garbage", ConfidenceLevel.MEDIUM),
            ("", ConfidenceLevel.MEDIUM),
            ("MODERATE", ConfidenceLevel.MEDIUM),
        ],
    )
    def test_coerce(self, value: object, expected: ConfidenceLevel) -> None:
        assert _coerce_confidence(value) == expected


# ---------------------------------------------------------------------------
# _coerce_statement_type (lines 245-258)
# ---------------------------------------------------------------------------


class TestCoerceStatementType:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("author_claim", StatementType.AUTHOR_CLAIM),
            ("empirical_result", StatementType.EMPIRICAL_RESULT),
            ("interpretation", StatementType.INTERPRETATION),
            ("model_inference", StatementType.MODEL_INFERENCE),
            ("author", StatementType.AUTHOR_CLAIM),
            ("claim", StatementType.AUTHOR_CLAIM),
            ("result", StatementType.EMPIRICAL_RESULT),
            ("empirical", StatementType.EMPIRICAL_RESULT),
            ("inference", StatementType.MODEL_INFERENCE),
            ("inferred", StatementType.MODEL_INFERENCE),
            (None, StatementType.AUTHOR_CLAIM),
            ("junk", StatementType.INTERPRETATION),
        ],
    )
    def test_coerce(self, value: object, expected: StatementType) -> None:
        assert _coerce_statement_type(value) == expected


# ---------------------------------------------------------------------------
# _as_statement_items (lines 261-269)
# ---------------------------------------------------------------------------


class TestAsStatementItems:
    def test_none_returns_empty(self) -> None:
        assert _as_statement_items(None) == []

    def test_list_passthrough(self) -> None:
        items = ["a", "b"]
        assert _as_statement_items(items) is items

    def test_tuple_converted(self) -> None:
        assert _as_statement_items(("x", "y")) == ["x", "y"]

    def test_single_value_wrapped(self) -> None:
        assert _as_statement_items("hello") == ["hello"]

    def test_single_dict_wrapped(self) -> None:
        d = {"statement": "s"}
        assert _as_statement_items(d) == [d]


# ---------------------------------------------------------------------------
# _parse_statement_items (lines 272-324)
# ---------------------------------------------------------------------------


class TestParseStatementItems:
    def test_dict_items(self) -> None:
        items = [
            {
                "statement": "The model outperforms",
                "statement_type": "empirical_result",
                "confidence": "HIGH",
                "evidence_ids": ["E001"],
                "notes": "Table 3",
            }
        ]
        result = _parse_statement_items(items, "results", "2301.00001", "E001")
        assert len(result) == 1
        s = result[0]
        assert s.statement == "The model outperforms"
        assert s.statement_type == StatementType.EMPIRICAL_RESULT
        assert s.confidence == ConfidenceLevel.HIGH
        assert s.evidence_ids == ["E001"]
        assert s.notes == "Table 3"
        assert s.statement_id == "2301.00001:results:001"

    def test_dict_text_key(self) -> None:
        result = _parse_statement_items([{"text": "A claim"}], "context", "p1", "E001")
        assert result[0].statement == "A claim"

    def test_dict_claim_key(self) -> None:
        result = _parse_statement_items(
            [{"claim": "Something"}], "context", "p1", "E001"
        )
        assert result[0].statement == "Something"

    def test_dict_description_key(self) -> None:
        result = _parse_statement_items(
            [{"description": "A desc"}], "context", "p1", "E001"
        )
        assert result[0].statement == "A desc"

    def test_string_items(self) -> None:
        result = _parse_statement_items(["plain text"], "methods", "p1", "E001")
        assert len(result) == 1
        assert result[0].statement == "plain text"
        assert result[0].evidence_ids == ["E001"]

    def test_empty_statement_skipped(self) -> None:
        result = _parse_statement_items([{"statement": "  "}], "methods", "p1", "E001")
        assert result == []

    def test_not_reported_handling(self) -> None:
        result = _parse_statement_items(["not_reported"], "datasets", "p1", "E001")
        assert len(result) == 1
        s = result[0]
        assert s.statement_type == StatementType.INTERPRETATION
        assert s.confidence == ConfidenceLevel.LOW
        assert s.evidence_ids == []

    def test_default_evidence_assigned(self) -> None:
        result = _parse_statement_items(
            [{"statement": "Observation"}], "results", "p1", "E001"
        )
        assert result[0].evidence_ids == ["E001"]

    def test_no_default_evidence(self) -> None:
        result = _parse_statement_items(
            [{"statement": "A finding"}], "results", "p1", ""
        )
        assert result[0].evidence_ids == []

    def test_evidence_single_string_normalized(self) -> None:
        result = _parse_statement_items(
            [{"statement": "A", "evidence_ids": "E002"}], "results", "p1", ""
        )
        assert result[0].evidence_ids == ["E002"]

    def test_evidence_from_evidence_key(self) -> None:
        result = _parse_statement_items(
            [{"statement": "A", "evidence": ["E003"]}], "results", "p1", ""
        )
        assert result[0].evidence_ids == ["E003"]

    def test_none_input(self) -> None:
        assert _parse_statement_items(None, "methods", "p1", "E001") == []


# ---------------------------------------------------------------------------
# score_extraction_quality warnings (lines 382-386)
# ---------------------------------------------------------------------------


class TestScoreExtractionQualityWarnings:
    def test_unsupported_warning(self) -> None:
        record = PaperExtractionRecord(
            paper_id="p1",
            title="T",
            results=[_make_statement(text="Real finding", evidence_ids=[])],
            evidence=[_make_evidence()],
        )
        q = score_extraction_quality(record)
        assert any("lack evidence" in w for w in q.warnings)

    def test_generic_warning(self) -> None:
        record = PaperExtractionRecord(
            paper_id="p1",
            title="T",
            results=[
                _make_statement(text="as above", evidence_ids=["E001"]),
            ],
            evidence=[_make_evidence()],
        )
        q = score_extraction_quality(record)
        assert any("generic" in w for w in q.warnings)

    def test_no_evidence_warning(self) -> None:
        record = PaperExtractionRecord(
            paper_id="p1",
            title="T",
            results=[_make_statement(text="A result", evidence_ids=["E001"])],
            evidence=[],
        )
        q = score_extraction_quality(record)
        assert any("No evidence" in w for w in q.warnings)

    def test_missing_critical_fields_warning(self) -> None:
        record = PaperExtractionRecord(
            paper_id="p1",
            title="T",
            evidence=[_make_evidence()],
        )
        q = score_extraction_quality(record)
        assert any("Missing critical" in w for w in q.warnings)
        assert q.completeness_score < 1.0

    def test_all_warnings_combined(self) -> None:
        record = PaperExtractionRecord(
            paper_id="p1",
            title="T",
            results=[_make_statement(text="none", evidence_ids=[])],
            evidence=[],
        )
        q = score_extraction_quality(record)
        assert any("lack evidence" in w for w in q.warnings)
        assert any("generic" in w for w in q.warnings)
        assert any("No evidence" in w for w in q.warnings)


# ---------------------------------------------------------------------------
# _parse_extraction_response (lines 398-463)
# ---------------------------------------------------------------------------


class TestParseExtractionResponse:
    def test_new_schema(self) -> None:
        response = {
            "context": [{"statement": "Background info", "confidence": "HIGH"}],
            "problem": [{"statement": "The problem", "evidence_ids": ["E001"]}],
            "contributions": [],
            "methods": [{"statement": "Our method"}],
            "results": [{"statement": "Performance improved"}],
            "limitations": [{"statement": "Small dataset"}],
            "uncertainties": ["Not sure about X"],
        }
        evidence = [_make_evidence()]
        record = _parse_extraction_response(
            response, "2301.00001", "v1", "Title", evidence, "gpt-4"
        )
        assert record.paper_id == "2301.00001"
        assert record.version == "v1"
        assert record.title == "Title"
        assert record.extraction_metadata.mode == "structured"
        assert record.extraction_metadata.model == "gpt-4"
        assert len(record.problem) == 1
        assert record.problem[0].statement == "The problem"
        assert "Not sure about X" in record.uncertainties
        assert record.quality.completeness_score >= 0.0

    def test_backward_compat_path(self) -> None:
        """Old paper_summary schema: objective/methodology keys.

        No extraction categories.
        """
        response = {
            "objective": "Study X",
            "methodology": "Approach Y",
            "findings": ["F1", "F2"],
            "lims": ["L1"],  # not a valid extraction category key
            "uncertainties": ["U1"],
        }
        evidence = [_make_evidence()]
        record = _parse_extraction_response(response, "p1", "v1", "T", evidence)
        assert len(record.problem) == 1
        assert record.problem[0].statement == "Study X"
        assert len(record.methods) == 1
        assert record.methods[0].statement == "Approach Y"
        assert len(record.results) == 2
        assert len(record.limitations) == 0  # "lims" not mapped

    def test_empty_response(self) -> None:
        record = _parse_extraction_response({}, "p1", "v1", "T", [_make_evidence()])
        assert record.paper_id == "p1"
        assert record.quality is not None


# ---------------------------------------------------------------------------
# project_extraction_to_summary — "See paper:" fallback (line 578)
# ---------------------------------------------------------------------------


class TestProjectExtractionToSummary:
    def test_see_paper_fallback(self) -> None:
        """When all statements are 'not_reported', first_statement returns fallback."""
        record = PaperExtractionRecord(
            paper_id="p1",
            version="v1",
            title="My Title",
            problem=[_make_statement(text="not_reported", category="problem")],
            context=[_make_statement(text="not_reported", category="context")],
            methods=[_make_statement(text="not_reported", category="methods")],
            contributions=[
                _make_statement(text="not_reported", category="contributions")
            ],
            results=[_make_statement(text="not_reported", category="results")],
            limitations=[_make_statement(text="not_reported", category="limitations")],
            evidence=[],
        )
        summary = project_extraction_to_summary(record)
        assert isinstance(summary, PaperSummary)
        assert summary.objective == "See paper: My Title"
        assert summary.methodology == "See paper: My Title"

    def test_normal_projection(self) -> None:
        record = PaperExtractionRecord(
            paper_id="p1",
            version="v1",
            title="Title",
            problem=[_make_statement(text="The problem", category="problem")],
            methods=[_make_statement(text="The method", category="methods")],
            contributions=[
                _make_statement(text="A contribution", category="contributions")
            ],
            results=[_make_statement(text="A result", category="results")],
            limitations=[_make_statement(text="A limit", category="limitations")],
            evidence=[_make_evidence()],
        )
        summary = project_extraction_to_summary(record)
        assert summary.objective == "The problem"
        assert summary.methodology == "The method"
        assert "A contribution" in summary.findings
        assert "A limit" in summary.limitations


# ---------------------------------------------------------------------------
# extract_paper (lines 673, 679-701)
# ---------------------------------------------------------------------------

_SIMPLE_MD = """\
# Introduction

This paper introduces a novel method.

# Methods

We propose a transformer approach.

# Results

Performance improved by 10%.
"""


class TestExtractPaper:
    def test_llm_success(self, tmp_path: Path) -> None:
        md_file = tmp_path / "paper.md"
        md_file.write_text(_SIMPLE_MD, encoding="utf-8")
        llm_response = {
            "problem": [{"statement": "Novel method"}],
            "methods": [{"statement": "Transformer approach"}],
            "results": [{"statement": "10% improvement"}],
            "contributions": [{"statement": "New arch"}],
            "limitations": [{"statement": "Small data"}],
            "assumptions": [{"statement": "IID assumption"}],
        }
        provider = _make_mock_provider(llm_response)
        record = extract_paper(
            md_file,
            "2301.00001",
            "v1",
            "Test Paper",
            ["transformers"],
            llm_provider=provider,
        )
        assert record.extraction_metadata.mode == "structured"
        assert record.extraction_metadata.model == "mock-model"
        provider.call.assert_called_once()

    def test_llm_failure_falls_back(self, tmp_path: Path) -> None:
        md_file = tmp_path / "paper.md"
        md_file.write_text(_SIMPLE_MD, encoding="utf-8")
        provider = _make_mock_provider(RuntimeError("LLM failed"))
        record = extract_paper(
            md_file,
            "2301.00001",
            "v1",
            "Test Paper",
            ["transformers"],
            llm_provider=provider,
        )
        assert record.extraction_metadata.mode == "template_fallback"

    def test_no_llm_provider(self, tmp_path: Path) -> None:
        md_file = tmp_path / "paper.md"
        md_file.write_text(_SIMPLE_MD, encoding="utf-8")
        record = extract_paper(
            md_file,
            "2301.00001",
            "v1",
            "Test Paper",
            ["transformers"],
        )
        assert record.extraction_metadata.mode == "template_fallback"
        assert len(record.evidence) > 0

    def test_empty_relevant_fallback(self, tmp_path: Path) -> None:
        """When retrieve_relevant_chunks returns empty, uses raw chunks."""
        md_file = tmp_path / "paper.md"
        md_file.write_text(_SIMPLE_MD, encoding="utf-8")
        with patch(
            "research_pipeline.summarization.per_paper.retrieve_relevant_chunks",
            return_value=[],
        ):
            record = extract_paper(
                md_file,
                "2301.00001",
                "v1",
                "Paper",
                ["test"],
            )
        assert record.extraction_metadata.mode == "template_fallback"
        assert len(record.evidence) > 0
