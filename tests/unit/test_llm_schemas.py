"""Tests for LLM I/O Pydantic schemas."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from research_pipeline.llm.schemas import (
    RelevanceJudgmentInput,
    RelevanceJudgmentOutput,
    SummarizationInput,
    SummarizationOutput,
    SynthesisOutput,
)


class TestRelevanceJudgmentInput:
    """Tests for RelevanceJudgmentInput schema."""

    def test_construct(self) -> None:
        obj = RelevanceJudgmentInput(
            topic="AI memory",
            must_terms=["memory", "agent"],
            paper_title="MemoryBank",
            paper_abstract="We propose MemoryBank...",
            paper_categories=["cs.AI"],
        )
        assert obj.topic == "AI memory"
        assert obj.must_terms == ["memory", "agent"]

    def test_roundtrip(self) -> None:
        """Serialize and deserialize preserves data."""
        obj = RelevanceJudgmentInput(
            topic="t",
            must_terms=["a"],
            paper_title="p",
            paper_abstract="abs",
            paper_categories=["cs.CL"],
        )
        data = obj.model_dump()
        restored = RelevanceJudgmentInput.model_validate(data)
        assert restored == obj

    def test_json_roundtrip(self) -> None:
        """JSON serialization roundtrip."""
        obj = RelevanceJudgmentInput(
            topic="t",
            must_terms=[],
            paper_title="p",
            paper_abstract="a",
            paper_categories=[],
        )
        json_str = obj.model_dump_json()
        restored = RelevanceJudgmentInput.model_validate_json(json_str)
        assert restored == obj


class TestRelevanceJudgmentOutput:
    """Tests for RelevanceJudgmentOutput schema."""

    def _make(self, score: float = 0.5) -> RelevanceJudgmentOutput:
        return RelevanceJudgmentOutput(
            llm_score=score,
            label="relevant",
            rationale=["matches topic"],
            evidence_quotes=[{"quote": "data"}],
            uncertainties=[],
            needs_fulltext_validation=[],
        )

    def test_construct(self) -> None:
        obj = self._make(0.8)
        assert obj.llm_score == 0.8
        assert obj.label == "relevant"

    def test_score_zero_ok(self) -> None:
        obj = self._make(0.0)
        assert obj.llm_score == 0.0

    def test_score_one_ok(self) -> None:
        obj = self._make(1.0)
        assert obj.llm_score == 1.0

    def test_score_below_zero_fails(self) -> None:
        with pytest.raises(ValidationError):
            self._make(-0.1)

    def test_score_above_one_fails(self) -> None:
        with pytest.raises(ValidationError):
            self._make(1.1)

    def test_roundtrip(self) -> None:
        obj = self._make(0.75)
        data = obj.model_dump()
        restored = RelevanceJudgmentOutput.model_validate(data)
        assert restored == obj

    def test_json_roundtrip(self) -> None:
        obj = self._make(0.5)
        json_str = obj.model_dump_json()
        restored = RelevanceJudgmentOutput.model_validate_json(json_str)
        assert restored == obj


class TestSummarizationInput:
    """Tests for SummarizationInput schema."""

    def test_construct(self) -> None:
        obj = SummarizationInput(
            topic="topic",
            paper_title="title",
            chunks=[{"section": "intro", "text": "..."}],
        )
        assert obj.topic == "topic"
        assert len(obj.chunks) == 1

    def test_roundtrip(self) -> None:
        obj = SummarizationInput(
            topic="t",
            paper_title="p",
            chunks=[],
        )
        data = obj.model_dump()
        restored = SummarizationInput.model_validate(data)
        assert restored == obj


class TestSummarizationOutput:
    """Tests for SummarizationOutput schema."""

    def test_construct(self) -> None:
        obj = SummarizationOutput(
            objective="study X",
            methodology="we used Y",
            findings=["f1"],
            limitations=["l1"],
            evidence=[{"claim": "c1", "source": "s1"}],
            uncertainties=["u1"],
        )
        assert obj.objective == "study X"
        assert obj.findings == ["f1"]

    def test_roundtrip(self) -> None:
        obj = SummarizationOutput(
            objective="o",
            methodology="m",
            findings=[],
            limitations=[],
            evidence=[],
            uncertainties=[],
        )
        data = obj.model_dump()
        restored = SummarizationOutput.model_validate(data)
        assert restored == obj

    def test_json_roundtrip(self) -> None:
        obj = SummarizationOutput(
            objective="o",
            methodology="m",
            findings=["a"],
            limitations=[],
            evidence=[],
            uncertainties=[],
        )
        json_str = obj.model_dump_json()
        restored = SummarizationOutput.model_validate_json(json_str)
        assert restored == obj


class TestSynthesisOutput:
    """Tests for SynthesisOutput schema."""

    def test_construct(self) -> None:
        obj = SynthesisOutput(
            agreements=[{"topic": "X", "papers": ["p1", "p2"]}],
            disagreements=[],
            open_questions=["Q1"],
        )
        assert len(obj.agreements) == 1
        assert obj.open_questions == ["Q1"]

    def test_roundtrip(self) -> None:
        obj = SynthesisOutput(
            agreements=[],
            disagreements=[],
            open_questions=[],
        )
        data = obj.model_dump()
        restored = SynthesisOutput.model_validate(data)
        assert restored == obj

    def test_json_roundtrip(self) -> None:
        obj = SynthesisOutput(
            agreements=[{"a": 1}],
            disagreements=[{"b": 2}],
            open_questions=["q"],
        )
        json_str = obj.model_dump_json()
        restored = SynthesisOutput.model_validate_json(json_str)
        assert restored == obj
