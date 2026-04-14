"""Tests for the LLM-based relevance judge."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest.mock import MagicMock

from research_pipeline.llm.base import LLMProvider
from research_pipeline.models.candidate import CandidateRecord
from research_pipeline.models.screening import LLMJudgment
from research_pipeline.screening.llm_judge import (
    _build_prompt,
    _clamp,
    _parse_response,
    judge_batch,
    judge_candidate,
)


def _make_candidate(**overrides: Any) -> CandidateRecord:
    """Create a minimal CandidateRecord for testing."""
    defaults: dict[str, Any] = {
        "arxiv_id": "2401.00001",
        "version": "v1",
        "title": "Attention Is All You Need",
        "authors": ["A. Vaswani"],
        "published": datetime(2024, 1, 1, tzinfo=UTC),
        "updated": datetime(2024, 1, 1, tzinfo=UTC),
        "categories": ["cs.CL", "cs.AI"],
        "primary_category": "cs.CL",
        "abstract": "We propose a new architecture based on attention mechanisms.",
        "abs_url": "https://arxiv.org/abs/2401.00001",
        "pdf_url": "https://arxiv.org/pdf/2401.00001",
    }
    defaults.update(overrides)
    return CandidateRecord(**defaults)


def _make_provider(response: dict[str, Any]) -> LLMProvider:
    """Create a mock LLMProvider returning a fixed response."""
    provider = MagicMock(spec=LLMProvider)
    provider.call.return_value = response
    provider.model_name.return_value = "mock-model"
    return provider


def _valid_response(**overrides: Any) -> dict[str, Any]:
    """Return a valid LLM response dict."""
    base: dict[str, Any] = {
        "llm_score": 0.85,
        "label": "high",
        "rationale": ["Paper directly addresses transformer architectures."],
        "evidence_quotes": [
            {"text": "attention mechanisms", "source": "abstract"},
        ],
        "uncertainties": ["Exact novelty unclear from abstract alone."],
        "needs_fulltext_validation": ["Claims about efficiency gains."],
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# judge_candidate — None provider
# ---------------------------------------------------------------------------


class TestJudgeCandidateNoProvider:
    """Tests for judge_candidate when no LLM provider is given."""

    def test_returns_none_when_no_provider(self) -> None:
        candidate = _make_candidate()
        result = judge_candidate(
            candidate, "transformers", ["attention"], llm_provider=None
        )
        assert result is None

    def test_returns_none_with_default_provider(self) -> None:
        candidate = _make_candidate()
        result = judge_candidate(candidate, "transformers", ["attention"])
        assert result is None


# ---------------------------------------------------------------------------
# judge_candidate — valid responses
# ---------------------------------------------------------------------------


class TestJudgeCandidateValid:
    """Tests for judge_candidate with a valid mock provider response."""

    def test_returns_llm_judgment(self) -> None:
        provider = _make_provider(_valid_response())
        candidate = _make_candidate()
        result = judge_candidate(
            candidate, "transformers", ["attention"], llm_provider=provider
        )

        assert result is not None
        assert isinstance(result, LLMJudgment)
        assert result.llm_score == 0.85
        assert result.label == "high"
        assert len(result.rationale) == 1
        assert len(result.uncertainties) == 1

    def test_provider_called_with_correct_args(self) -> None:
        provider = _make_provider(_valid_response())
        candidate = _make_candidate()
        judge_candidate(candidate, "transformers", ["attention"], llm_provider=provider)

        provider.call.assert_called_once()
        call_kwargs = provider.call.call_args
        assert call_kwargs.kwargs["schema_id"] == "relevance_judgment"
        assert call_kwargs.kwargs["temperature"] == 0.0

    def test_prompt_contains_topic_and_title(self) -> None:
        provider = _make_provider(_valid_response())
        candidate = _make_candidate(title="My Special Paper")
        judge_candidate(candidate, "deep learning", ["neural"], llm_provider=provider)

        prompt = provider.call.call_args.kwargs["prompt"]
        assert "deep learning" in prompt
        assert "My Special Paper" in prompt
        assert "neural" in prompt

    def test_all_labels_accepted(self) -> None:
        for label in ("high", "medium", "low", "off_topic"):
            provider = _make_provider(_valid_response(label=label))
            candidate = _make_candidate()
            result = judge_candidate(
                candidate, "transformers", ["attention"], llm_provider=provider
            )
            assert result is not None
            assert result.label == label


# ---------------------------------------------------------------------------
# judge_candidate — error handling
# ---------------------------------------------------------------------------


class TestJudgeCandidateErrors:
    """Tests for judge_candidate error handling."""

    def test_invalid_json_returns_none(self) -> None:
        provider = MagicMock(spec=LLMProvider)
        provider.call.return_value = "not a dict"
        candidate = _make_candidate()
        result = judge_candidate(
            candidate, "transformers", ["attention"], llm_provider=provider
        )
        assert result is None

    def test_provider_exception_returns_none(self) -> None:
        provider = MagicMock(spec=LLMProvider)
        provider.call.side_effect = RuntimeError("API timeout")
        candidate = _make_candidate()
        result = judge_candidate(
            candidate, "transformers", ["attention"], llm_provider=provider
        )
        assert result is None

    def test_score_above_one_clamped(self) -> None:
        provider = _make_provider(_valid_response(llm_score=1.5))
        candidate = _make_candidate()
        result = judge_candidate(
            candidate, "transformers", ["attention"], llm_provider=provider
        )
        assert result is not None
        assert result.llm_score == 1.0

    def test_score_below_zero_clamped(self) -> None:
        provider = _make_provider(_valid_response(llm_score=-0.3))
        candidate = _make_candidate()
        result = judge_candidate(
            candidate, "transformers", ["attention"], llm_provider=provider
        )
        assert result is not None
        assert result.llm_score == 0.0

    def test_invalid_label_defaults_to_medium(self) -> None:
        provider = _make_provider(_valid_response(label="super_relevant"))
        candidate = _make_candidate()
        result = judge_candidate(
            candidate, "transformers", ["attention"], llm_provider=provider
        )
        assert result is not None
        assert result.label == "medium"

    def test_missing_optional_fields_still_parses(self) -> None:
        provider = _make_provider({"llm_score": 0.5, "label": "low"})
        candidate = _make_candidate()
        result = judge_candidate(
            candidate, "transformers", ["attention"], llm_provider=provider
        )
        assert result is not None
        assert result.rationale == []
        assert result.uncertainties == []
        assert result.needs_fulltext_validation == []


# ---------------------------------------------------------------------------
# judge_batch
# ---------------------------------------------------------------------------


class TestJudgeBatch:
    """Tests for the judge_batch function."""

    def test_batch_returns_correct_length(self) -> None:
        provider = _make_provider(_valid_response())
        candidates = [_make_candidate(arxiv_id=f"2401.{i:05d}") for i in range(5)]
        results = judge_batch(
            candidates, "transformers", ["attention"], llm_provider=provider
        )
        assert len(results) == 5
        assert all(r is not None for r in results)

    def test_batch_with_none_provider_returns_all_nones(self) -> None:
        candidates = [_make_candidate(arxiv_id=f"2401.{i:05d}") for i in range(3)]
        results = judge_batch(
            candidates, "transformers", ["attention"], llm_provider=None
        )
        assert len(results) == 3
        assert all(r is None for r in results)

    def test_batch_empty_list(self) -> None:
        provider = _make_provider(_valid_response())
        results = judge_batch([], "transformers", ["attention"], llm_provider=provider)
        assert results == []


# ---------------------------------------------------------------------------
# _build_prompt
# ---------------------------------------------------------------------------


class TestBuildPrompt:
    """Tests for the internal _build_prompt helper."""

    def test_contains_all_fields(self) -> None:
        candidate = _make_candidate(
            title="Test Paper",
            abstract="Test abstract text.",
            categories=["cs.LG", "stat.ML"],
        )
        prompt = _build_prompt(candidate, "machine learning", ["neural", "network"])
        assert "machine learning" in prompt
        assert "neural, network" in prompt
        assert "Test Paper" in prompt
        assert "Test abstract text." in prompt
        assert "cs.LG" in prompt
        assert "stat.ML" in prompt

    def test_empty_must_terms(self) -> None:
        candidate = _make_candidate()
        prompt = _build_prompt(candidate, "topic", [])
        assert "(none)" in prompt


# ---------------------------------------------------------------------------
# _clamp
# ---------------------------------------------------------------------------


class TestClamp:
    """Tests for the _clamp helper."""

    def test_within_range(self) -> None:
        assert _clamp(0.5, 0.0, 1.0) == 0.5

    def test_below_range(self) -> None:
        assert _clamp(-1.0, 0.0, 1.0) == 0.0

    def test_above_range(self) -> None:
        assert _clamp(2.0, 0.0, 1.0) == 1.0


# ---------------------------------------------------------------------------
# _parse_response
# ---------------------------------------------------------------------------


class TestParseResponse:
    """Tests for the internal _parse_response helper."""

    def test_valid_response(self) -> None:
        result = _parse_response(_valid_response())
        assert result is not None
        assert result.llm_score == 0.85
        assert result.label == "high"

    def test_non_list_rationale_becomes_empty(self) -> None:
        result = _parse_response(_valid_response(rationale="not a list"))
        assert result is not None
        assert result.rationale == []

    def test_non_list_uncertainties_becomes_empty(self) -> None:
        result = _parse_response(_valid_response(uncertainties=42))
        assert result is not None
        assert result.uncertainties == []

    def test_score_string_coerced_to_float(self) -> None:
        result = _parse_response(_valid_response(llm_score="0.7"))
        assert result is not None
        assert result.llm_score == 0.7
