"""Tests for LLM-powered per-paper summarization and cross-paper synthesis."""

from __future__ import annotations

import textwrap
from pathlib import Path
from unittest.mock import MagicMock

from research_pipeline.llm.base import LLMProvider
from research_pipeline.models.summary import (
    PaperSummary,
    SummaryEvidence,
    SynthesisReport,
)
from research_pipeline.summarization.per_paper import summarize_paper
from research_pipeline.summarization.synthesis import synthesize

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SAMPLE_MARKDOWN = textwrap.dedent("""\
    # Introduction

    This paper studies transformer architectures for time series forecasting.
    We propose a novel attention mechanism that captures temporal patterns.

    # Methodology

    We use a multi-head self-attention mechanism combined with positional
    encoding tailored to temporal data. The model is trained end-to-end.

    # Results

    Our experiments on five benchmark datasets show significant improvement
    over baseline models. The proposed method achieves state-of-the-art
    performance on ETTh1 and ETTm2 datasets.

    # Limitations

    The model requires substantial computational resources. Training on
    very long sequences remains challenging.
""")

_ARXIV_ID = "2401.00001"
_VERSION = "v1"
_TITLE = "Temporal Transformers for Time Series"
_TOPIC_TERMS = ["transformer", "time series", "forecasting"]


def _make_mock_provider(return_value: dict | Exception) -> LLMProvider:  # type: ignore[type-arg]
    """Create a mock LLM provider that returns *return_value* or raises it."""
    provider = MagicMock(spec=LLMProvider)
    provider.model_name.return_value = "mock-model"
    if isinstance(return_value, Exception):
        provider.call.side_effect = return_value
    else:
        provider.call.return_value = return_value
    return provider


def _write_markdown(tmp_path: Path) -> Path:
    md_path = tmp_path / "paper.md"
    md_path.write_text(_SAMPLE_MARKDOWN, encoding="utf-8")
    return md_path


def _make_paper_summary(
    arxiv_id: str = _ARXIV_ID,
    title: str = _TITLE,
    findings: list[str] | None = None,
    limitations: list[str] | None = None,
) -> PaperSummary:
    return PaperSummary(
        arxiv_id=arxiv_id,
        version=_VERSION,
        title=title,
        objective=f"Objective of {title}",
        methodology="Some methodology",
        findings=findings or ["Finding A", "Finding B"],
        limitations=limitations or ["Limitation X"],
        evidence=[],
        uncertainties=[],
    )


# ===================================================================
# Task Q — per-paper summarize_paper
# ===================================================================


class TestSummarizePaperTemplateMode:
    """Template mode (no LLM) should work exactly as before."""

    def test_template_mode_returns_paper_summary(self, tmp_path: Path) -> None:
        md_path = _write_markdown(tmp_path)
        result = summarize_paper(md_path, _ARXIV_ID, _VERSION, _TITLE, _TOPIC_TERMS)
        assert isinstance(result, PaperSummary)
        assert result.arxiv_id == _ARXIV_ID
        assert result.version == _VERSION

    def test_template_mode_methodology_says_template(self, tmp_path: Path) -> None:
        md_path = _write_markdown(tmp_path)
        result = summarize_paper(md_path, _ARXIV_ID, _VERSION, _TITLE, _TOPIC_TERMS)
        assert "template mode" in result.methodology.lower()

    def test_template_mode_has_evidence(self, tmp_path: Path) -> None:
        md_path = _write_markdown(tmp_path)
        result = summarize_paper(md_path, _ARXIV_ID, _VERSION, _TITLE, _TOPIC_TERMS)
        assert len(result.evidence) > 0
        assert all(isinstance(ev, SummaryEvidence) for ev in result.evidence)

    def test_template_mode_none_llm_provider(self, tmp_path: Path) -> None:
        """Explicitly passing None should behave identically to omitting it."""
        md_path = _write_markdown(tmp_path)
        result = summarize_paper(
            md_path,
            _ARXIV_ID,
            _VERSION,
            _TITLE,
            _TOPIC_TERMS,
            llm_provider=None,
        )
        assert "template mode" in result.methodology.lower()


class TestSummarizePaperLLMMode:
    """LLM-powered per-paper summarization."""

    _VALID_LLM_RESPONSE: dict = {  # type: ignore[type-arg]
        "objective": "Study temporal transformers for forecasting.",
        "methodology": "Multi-head attention with temporal positional encoding.",
        "findings": [
            "State-of-the-art on ETTh1",
            "Outperforms baselines on ETTm2",
        ],
        "limitations": ["High compute cost", "Long-sequence issues"],
        "uncertainties": ["Generalisation to non-stationary data unclear"],
    }

    def test_llm_mode_returns_llm_summary(self, tmp_path: Path) -> None:
        md_path = _write_markdown(tmp_path)
        provider = _make_mock_provider(self._VALID_LLM_RESPONSE)
        result = summarize_paper(
            md_path,
            _ARXIV_ID,
            _VERSION,
            _TITLE,
            _TOPIC_TERMS,
            llm_provider=provider,
        )
        assert result.objective == "Study temporal transformers for forecasting."
        assert "[LLM]" in result.methodology
        assert len(result.findings) == 2
        assert len(result.limitations) == 2

    def test_llm_mode_calls_provider_once(self, tmp_path: Path) -> None:
        md_path = _write_markdown(tmp_path)
        provider = _make_mock_provider(self._VALID_LLM_RESPONSE)
        summarize_paper(
            md_path,
            _ARXIV_ID,
            _VERSION,
            _TITLE,
            _TOPIC_TERMS,
            llm_provider=provider,
        )
        provider.call.assert_called_once()
        assert "paper_summary" in str(provider.call.call_args)

    def test_llm_mode_preserves_evidence(self, tmp_path: Path) -> None:
        md_path = _write_markdown(tmp_path)
        provider = _make_mock_provider(self._VALID_LLM_RESPONSE)
        result = summarize_paper(
            md_path,
            _ARXIV_ID,
            _VERSION,
            _TITLE,
            _TOPIC_TERMS,
            llm_provider=provider,
        )
        assert len(result.evidence) > 0

    def test_llm_mode_fallback_on_exception(self, tmp_path: Path) -> None:
        md_path = _write_markdown(tmp_path)
        provider = _make_mock_provider(RuntimeError("API unavailable"))
        result = summarize_paper(
            md_path,
            _ARXIV_ID,
            _VERSION,
            _TITLE,
            _TOPIC_TERMS,
            llm_provider=provider,
        )
        # Should fall back to template
        assert "template mode" in result.methodology.lower()

    def test_llm_mode_fallback_on_missing_key(self, tmp_path: Path) -> None:
        """If the LLM response is missing a required key, fall back."""
        md_path = _write_markdown(tmp_path)
        bad_response = {"objective": "ok"}  # missing other keys
        provider = _make_mock_provider(bad_response)
        result = summarize_paper(
            md_path,
            _ARXIV_ID,
            _VERSION,
            _TITLE,
            _TOPIC_TERMS,
            llm_provider=provider,
        )
        assert "template mode" in result.methodology.lower()


# ===================================================================
# Task R — cross-paper synthesize
# ===================================================================


class TestSynthesizeTemplateMode:
    """Template mode synthesis (improved)."""

    def test_template_mode_returns_report(self) -> None:
        summaries = [_make_paper_summary()]
        result = synthesize(summaries, "time series")
        assert isinstance(result, SynthesisReport)
        assert result.paper_count == 1

    def test_template_mode_collects_findings_as_agreements(self) -> None:
        s1 = _make_paper_summary(
            arxiv_id="2401.00001",
            findings=["Finding A", "Finding B"],
        )
        s2 = _make_paper_summary(
            arxiv_id="2401.00002",
            title="Another Paper",
            findings=["Finding C"],
        )
        result = synthesize([s1, s2], "time series")
        # Each finding becomes an agreement entry
        assert len(result.agreements) == 3
        claims = [a.claim for a in result.agreements]
        assert "Finding A" in claims
        assert "Finding C" in claims

    def test_template_mode_collects_limitations(self) -> None:
        s1 = _make_paper_summary(
            arxiv_id="2401.00001",
            limitations=["Lim 1"],
        )
        s2 = _make_paper_summary(
            arxiv_id="2401.00002",
            title="Paper Two",
            limitations=["Lim 2", "Lim 3"],
        )
        result = synthesize([s1, s2], "topic")
        # Limitations should appear in open_questions combined text
        combined = " ".join(result.open_questions)
        assert "Lim 1" in combined
        assert "Lim 2" in combined

    def test_template_mode_open_question_mentions_llm(self) -> None:
        result = synthesize([_make_paper_summary()], "topic")
        assert any("LLM" in q for q in result.open_questions)

    def test_template_mode_empty_summaries(self) -> None:
        result = synthesize([], "topic")
        assert result.paper_count == 0
        assert result.agreements == []


class TestSynthesizeLLMMode:
    """LLM-powered cross-paper synthesis."""

    _VALID_SYNTHESIS_RESPONSE: dict = {  # type: ignore[type-arg]
        "agreements": [
            {
                "claim": "Transformers outperform baselines on time series",
                "supporting_papers": ["2401.00001", "2401.00002"],
            },
        ],
        "disagreements": [
            {
                "topic": "Optimal attention mechanism",
                "positions": {
                    "2401.00001": "Full self-attention is best",
                    "2401.00002": "Sparse attention is more efficient",
                },
            },
        ],
        "open_questions": ["How to handle non-stationary data?"],
    }

    def test_llm_mode_returns_structured_report(self) -> None:
        summaries = [
            _make_paper_summary(arxiv_id="2401.00001"),
            _make_paper_summary(arxiv_id="2401.00002", title="Paper Two"),
        ]
        provider = _make_mock_provider(self._VALID_SYNTHESIS_RESPONSE)
        result = synthesize(summaries, "time series", llm_provider=provider)

        assert len(result.agreements) == 1
        assert result.agreements[0].claim == (
            "Transformers outperform baselines on time series"
        )
        assert len(result.disagreements) == 1
        assert result.disagreements[0].topic == "Optimal attention mechanism"
        assert len(result.open_questions) == 1

    def test_llm_mode_calls_provider(self) -> None:
        summaries = [_make_paper_summary()]
        provider = _make_mock_provider(self._VALID_SYNTHESIS_RESPONSE)
        synthesize(summaries, "time series", llm_provider=provider)
        provider.call.assert_called_once()
        assert "synthesis" in str(provider.call.call_args)

    def test_llm_mode_fallback_on_exception(self) -> None:
        summaries = [_make_paper_summary()]
        provider = _make_mock_provider(RuntimeError("API error"))
        result = synthesize(summaries, "topic", llm_provider=provider)
        # Should fall back to template mode
        assert any("LLM" in q for q in result.open_questions)

    def test_llm_mode_fallback_on_bad_response(self) -> None:
        summaries = [_make_paper_summary()]
        provider = _make_mock_provider({"bad": "data"})
        result = synthesize(summaries, "topic", llm_provider=provider)
        # Missing keys → falls back to template
        assert any("LLM" in q for q in result.open_questions)

    def test_llm_mode_preserves_paper_summaries(self) -> None:
        summaries = [
            _make_paper_summary(arxiv_id="2401.00001"),
            _make_paper_summary(arxiv_id="2401.00002", title="Paper Two"),
        ]
        provider = _make_mock_provider(self._VALID_SYNTHESIS_RESPONSE)
        result = synthesize(summaries, "time series", llm_provider=provider)
        assert len(result.paper_summaries) == 2
        assert result.paper_summaries[0].arxiv_id == "2401.00001"
