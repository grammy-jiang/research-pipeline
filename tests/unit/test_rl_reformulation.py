"""Tests for screening.rl_reformulation — RL query reformulation."""

from __future__ import annotations

import pytest

from research_pipeline.screening.rl_reformulation import (
    OperatorBandit,
    QueryVariant,
    ReformulationOp,
    ReformulationResult,
    ReformulationStep,
    RewardSignal,
    RLReformulator,
)


# ---------------------------------------------------------------------------
# QueryVariant
# ---------------------------------------------------------------------------


class TestQueryVariant:
    def test_creation(self) -> None:
        v = QueryVariant(
            text="test query", operator=ReformulationOp.SYNONYM_EXPAND
        )
        assert v.text == "test query"
        assert v.generation == 0

    def test_to_dict(self) -> None:
        v = QueryVariant(
            text="q", operator=ReformulationOp.TERM_DROP, generation=2
        )
        d = v.to_dict()
        assert d["operator"] == "term_drop"
        assert d["generation"] == 2

    def test_frozen(self) -> None:
        v = QueryVariant(text="a", operator=ReformulationOp.TERM_BOOST)
        with pytest.raises(AttributeError):
            v.text = "b"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# ReformulationStep
# ---------------------------------------------------------------------------


class TestReformulationStep:
    def test_to_dict(self) -> None:
        v = QueryVariant(text="q", operator=ReformulationOp.PHRASE_RELAX)
        s = ReformulationStep(
            variant=v, reward=RewardSignal.SUCCESS, relevant_count=5, total_count=10
        )
        d = s.to_dict()
        assert d["reward"] == "success"
        assert d["relevant_count"] == 5


# ---------------------------------------------------------------------------
# ReformulationResult
# ---------------------------------------------------------------------------


class TestReformulationResult:
    def test_to_dict(self) -> None:
        r = ReformulationResult(
            best_query="improved query",
            original_query="original",
            total_reward=2.5,
            improvement=0.4,
        )
        d = r.to_dict()
        assert d["best_query"] == "improved query"
        assert d["total_steps"] == 0


# ---------------------------------------------------------------------------
# OperatorBandit
# ---------------------------------------------------------------------------


class TestOperatorBandit:
    def test_select_returns_operator(self) -> None:
        bandit = OperatorBandit(seed=42)
        op = bandit.select()
        assert isinstance(op, ReformulationOp)

    def test_update_success(self) -> None:
        bandit = OperatorBandit(seed=42)
        op = ReformulationOp.SYNONYM_EXPAND
        bandit.update(op, RewardSignal.SUCCESS)
        rates = bandit.success_rates()
        assert rates["synonym_expand"] > 0.5

    def test_update_failure(self) -> None:
        bandit = OperatorBandit(seed=42)
        op = ReformulationOp.TERM_DROP
        for _ in range(10):
            bandit.update(op, RewardSignal.FAILURE)
        rates = bandit.success_rates()
        assert rates["term_drop"] < 0.2

    def test_partial_reward(self) -> None:
        bandit = OperatorBandit(seed=42)
        op = ReformulationOp.TERM_BOOST
        bandit.update(op, RewardSignal.PARTIAL)
        rates = bandit.success_rates()
        assert rates["term_boost"] == pytest.approx(0.5, abs=0.1)

    def test_success_rates_all_present(self) -> None:
        bandit = OperatorBandit(seed=42)
        rates = bandit.success_rates()
        assert len(rates) == len(ReformulationOp)

    def test_learning_over_time(self) -> None:
        bandit = OperatorBandit(seed=42)
        # Synonym expand always succeeds
        for _ in range(20):
            bandit.update(ReformulationOp.SYNONYM_EXPAND, RewardSignal.SUCCESS)
            bandit.update(ReformulationOp.TERM_DROP, RewardSignal.FAILURE)
        rates = bandit.success_rates()
        assert rates["synonym_expand"] > rates["term_drop"]


# ---------------------------------------------------------------------------
# Operator functions (via RLReformulator)
# ---------------------------------------------------------------------------


class TestOperators:
    def test_synonym_expand(self) -> None:
        reformulator = RLReformulator(max_iterations=1, seed=42)
        result = reformulator.reformulate("improve model performance")
        # At least ran without error
        assert result.original_query == "improve model performance"

    def test_acronym_expand(self) -> None:
        from research_pipeline.screening.rl_reformulation import (
            _apply_acronym_expand,
        )
        import random as _rm
        rng = _rm.Random(42)
        expanded = _apply_acronym_expand("nlp with bert", rng)
        assert "natural language processing" in expanded
        assert "bidirectional encoder" in expanded

    def test_phrase_relax(self) -> None:
        from research_pipeline.screening.rl_reformulation import (
            _apply_phrase_relax,
        )
        import random as _rm
        rng = _rm.Random(42)
        relaxed = _apply_phrase_relax('"exact phrase" search', rng)
        assert '"' not in relaxed

    def test_scope_broaden(self) -> None:
        from research_pipeline.screening.rl_reformulation import (
            _apply_scope_broaden,
        )
        import random as _rm
        rng = _rm.Random(42)
        broad = _apply_scope_broaden("only specific transformer model", rng)
        assert "only" not in broad
        assert "specific" not in broad

    def test_term_drop_short_query(self) -> None:
        from research_pipeline.screening.rl_reformulation import (
            _apply_term_drop,
        )
        import random as _rm
        rng = _rm.Random(42)
        result = _apply_term_drop("ab", rng)
        assert result == "ab"  # too short, no change


# ---------------------------------------------------------------------------
# RLReformulator
# ---------------------------------------------------------------------------


class TestRLReformulator:
    def test_no_reward_fn(self) -> None:
        rf = RLReformulator(max_iterations=3, seed=42)
        result = rf.reformulate("transformer model for nlp")
        assert result.original_query == "transformer model for nlp"
        assert isinstance(result.total_reward, float)

    def test_with_reward_fn(self) -> None:
        call_count = 0

        def reward_fn(query: str) -> tuple[int, int]:
            nonlocal call_count
            call_count += 1
            if "enhanced" in query.lower() or "technique" in query.lower():
                return (8, 10)
            return (2, 10)

        rf = RLReformulator(max_iterations=5, seed=42)
        result = rf.reformulate("improve method analysis", reward_fn)
        assert call_count > 0
        assert result.best_query != ""

    def test_improvement_tracking(self) -> None:
        def good_reward(query: str) -> tuple[int, int]:
            if len(query) > 30:
                return (9, 10)
            return (1, 10)

        rf = RLReformulator(max_iterations=5, seed=42)
        result = rf.reformulate("data analysis", good_reward)
        assert isinstance(result.improvement, float)

    def test_history(self) -> None:
        rf = RLReformulator(max_iterations=2, seed=42)
        rf.reformulate("query one")
        rf.reformulate("query two")
        assert len(rf.history) == 2

    def test_summary_empty(self) -> None:
        rf = RLReformulator(seed=42)
        s = rf.summary()
        assert s["total_runs"] == 0

    def test_summary_with_data(self) -> None:
        rf = RLReformulator(max_iterations=3, seed=42)
        rf.reformulate("test query about methods")
        s = rf.summary()
        assert s["total_runs"] == 1
        assert "operator_success_rates" in s

    def test_deterministic_with_seed(self) -> None:
        rf1 = RLReformulator(max_iterations=3, seed=123)
        r1 = rf1.reformulate("machine learning method")
        rf2 = RLReformulator(max_iterations=3, seed=123)
        r2 = rf2.reformulate("machine learning method")
        assert r1.best_query == r2.best_query

    def test_bandit_access(self) -> None:
        rf = RLReformulator(seed=42)
        assert isinstance(rf.bandit, OperatorBandit)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class TestEnums:
    def test_reformulation_ops(self) -> None:
        assert len(ReformulationOp) == 7

    def test_reward_signals(self) -> None:
        assert len(RewardSignal) == 3
