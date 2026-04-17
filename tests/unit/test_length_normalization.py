"""Tests for llm.length_normalization — token budget enforcement."""

from __future__ import annotations

from research_pipeline.llm.length_normalization import (
    BUDGET_PROFILES,
    BatchStats,
    BudgetMode,
    NormalizationResult,
    TokenBudget,
    compute_batch_stats,
    estimate_tokens,
    normalize_batch,
    normalize_length,
)

# ── Token estimation ─────────────────────────────────────────────────


class TestEstimateTokens:
    def test_empty_string(self) -> None:
        assert estimate_tokens("") == 1  # +1 rounding

    def test_single_word(self) -> None:
        result = estimate_tokens("hello")
        assert result >= 1

    def test_known_length(self) -> None:
        # 15 words → ~20 tokens at 0.75 words/token
        text = " ".join(["word"] * 15)
        result = estimate_tokens(text)
        assert 15 <= result <= 25

    def test_long_text(self) -> None:
        text = " ".join(["word"] * 300)
        result = estimate_tokens(text)
        assert result > 300  # 300 words / 0.75 > 300 tokens


# ── Token budget dataclass ───────────────────────────────────────────


class TestTokenBudget:
    def test_default_mode(self) -> None:
        budget = TokenBudget(max_tokens=500)
        assert budget.mode == BudgetMode.SOFT
        assert budget.task_type == "default"

    def test_frozen(self) -> None:
        budget = TokenBudget(max_tokens=100)
        try:
            budget.max_tokens = 200  # type: ignore[misc]
            raise AssertionError("Should be frozen")
        except AttributeError:
            pass


# ── Budget profiles ──────────────────────────────────────────────────


class TestBudgetProfiles:
    def test_all_profiles_present(self) -> None:
        expected = {
            "screening",
            "extraction",
            "summary",
            "synthesis",
            "analysis",
            "plan",
        }
        assert set(BUDGET_PROFILES.keys()) == expected

    def test_screening_budget(self) -> None:
        b = BUDGET_PROFILES["screening"]
        assert b.max_tokens == 300
        assert b.mode == BudgetMode.SOFT

    def test_synthesis_is_largest(self) -> None:
        assert BUDGET_PROFILES["synthesis"].max_tokens >= max(
            p.max_tokens for k, p in BUDGET_PROFILES.items() if k != "synthesis"
        )


# ── normalize_length ─────────────────────────────────────────────────


class TestNormalizeLength:
    def test_under_budget_no_truncation(self) -> None:
        text = "Short answer."
        result = normalize_length(text, max_tokens=1000)
        assert not result.truncated
        assert result.text == text
        assert result.original_tokens == result.final_tokens

    def test_soft_mode_warns_but_keeps_text(self) -> None:
        text = " ".join(["word"] * 500)
        result = normalize_length(text, max_tokens=50, mode=BudgetMode.SOFT)
        assert not result.truncated
        assert result.text == text  # Soft mode does NOT truncate
        assert result.original_tokens > 50

    def test_hard_mode_truncates(self) -> None:
        text = "First sentence. Second sentence. Third sentence. Fourth sentence."
        result = normalize_length(text, max_tokens=10, mode=BudgetMode.HARD)
        assert result.truncated
        assert result.final_tokens <= result.original_tokens
        assert len(result.text) < len(text)

    def test_budget_object_overrides_params(self) -> None:
        budget = TokenBudget(max_tokens=5, mode=BudgetMode.HARD, task_type="test")
        text = "This is a long text that should be truncated to fit the budget."
        result = normalize_length(text, budget=budget, max_tokens=9999)
        assert result.budget.max_tokens == 5
        assert result.budget.task_type == "test"

    def test_empty_text(self) -> None:
        result = normalize_length("", max_tokens=100)
        assert not result.truncated
        assert result.text == ""

    def test_default_budget_is_1200(self) -> None:
        result = normalize_length("hello")
        assert result.budget.max_tokens == 1200

    def test_returns_normalization_result(self) -> None:
        result = normalize_length("test text")
        assert isinstance(result, NormalizationResult)
        assert isinstance(result.budget, TokenBudget)

    def test_hard_truncation_at_sentence_boundary(self) -> None:
        text = "First. Second. Third. Fourth. Fifth. Sixth. Seventh. Eighth."
        result = normalize_length(text, max_tokens=5, mode=BudgetMode.HARD)
        assert result.truncated
        # Should end at a sentence boundary (period)
        assert result.text.rstrip().endswith(".")

    def test_hard_truncation_single_long_sentence(self) -> None:
        # One very long sentence — forces word-level fallback
        text = " ".join(["longword"] * 200)
        result = normalize_length(text, max_tokens=10, mode=BudgetMode.HARD)
        assert result.truncated
        assert result.final_tokens <= result.original_tokens


# ── normalize_batch ──────────────────────────────────────────────────


class TestNormalizeBatch:
    def test_empty_batch(self) -> None:
        results = normalize_batch([], max_tokens=100)
        assert results == []

    def test_batch_applies_same_budget(self) -> None:
        texts = ["short", "also short", "tiny"]
        results = normalize_batch(texts, max_tokens=1000)
        assert len(results) == 3
        assert all(not r.truncated for r in results)

    def test_batch_hard_mode(self) -> None:
        texts = [
            "Very short.",
            " ".join(["word"] * 500) + ".",
        ]
        results = normalize_batch(texts, max_tokens=50, mode=BudgetMode.HARD)
        assert len(results) == 2
        assert not results[0].truncated
        assert results[1].truncated


# ── compute_batch_stats ──────────────────────────────────────────────


class TestComputeBatchStats:
    def test_empty_results(self) -> None:
        stats = compute_batch_stats([])
        assert stats.total_texts == 0
        assert stats.truncated_count == 0
        assert stats.over_budget_ratio == 0.0

    def test_no_truncation(self) -> None:
        results = normalize_batch(["hello", "world"], max_tokens=1000)
        stats = compute_batch_stats(results)
        assert stats.total_texts == 2
        assert stats.truncated_count == 0
        assert stats.over_budget_ratio == 0.0

    def test_mixed_truncation(self) -> None:
        results = [
            normalize_length("short", max_tokens=1000),
            normalize_length(
                " ".join(["word"] * 500), max_tokens=50, mode=BudgetMode.HARD
            ),
        ]
        stats = compute_batch_stats(results)
        assert stats.total_texts == 2
        assert stats.truncated_count == 1
        assert stats.over_budget_ratio == 0.5

    def test_details_populated(self) -> None:
        results = normalize_batch(["hello", "world"], max_tokens=1000)
        stats = compute_batch_stats(results)
        assert len(stats.details) == 2
        assert "original_tokens" in stats.details[0]
        assert "truncated" in stats.details[0]

    def test_returns_batch_stats(self) -> None:
        results = normalize_batch(["a", "b"], max_tokens=100)
        stats = compute_batch_stats(results)
        assert isinstance(stats, BatchStats)
        assert stats.avg_original_tokens > 0
