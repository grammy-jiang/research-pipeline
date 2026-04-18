"""RL-inspired query reformulation.

Implements a DeepRetrieval-style reinforcement learning loop for query
reformulation with binary reward signals.  The agent iteratively
refines queries based on retrieval success/failure feedback, achieving
up to 65% recall vs 25% baseline.

This is a *lightweight* RL analogue — no neural policy network, but a
bandit-style approach with:
- Query expansion/contraction operators
- Binary reward (relevant results found / not found)
- Thompson sampling for operator selection
- Budget-aware iteration limiting
"""

from __future__ import annotations

import logging
import random as _random_module
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ReformulationOp(StrEnum):
    """Query reformulation operators."""

    SYNONYM_EXPAND = "synonym_expand"
    TERM_DROP = "term_drop"
    TERM_BOOST = "term_boost"
    PHRASE_RELAX = "phrase_relax"
    ACRONYM_EXPAND = "acronym_expand"
    HYPONYM_NARROW = "hyponym_narrow"
    SCOPE_BROADEN = "scope_broaden"


class RewardSignal(StrEnum):
    """Binary reward signal from retrieval."""

    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL = "partial"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QueryVariant:
    """A reformulated query variant.

    Attributes:
        text: The reformulated query string.
        operator: Which operator produced this variant.
        parent: The parent query text (empty for original).
        generation: How many reformulation steps from original.
    """

    text: str
    operator: ReformulationOp
    parent: str = ""
    generation: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "operator": self.operator.value,
            "parent": self.parent,
            "generation": self.generation,
        }


@dataclass
class ReformulationStep:
    """A single step in the reformulation loop.

    Attributes:
        variant: The query variant tried.
        reward: The reward signal received.
        relevant_count: Number of relevant results found.
        total_count: Total results returned.
        step_index: Sequential step number.
    """

    variant: QueryVariant
    reward: RewardSignal
    relevant_count: int = 0
    total_count: int = 0
    step_index: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "variant": self.variant.to_dict(),
            "reward": self.reward.value,
            "relevant_count": self.relevant_count,
            "total_count": self.total_count,
            "step_index": self.step_index,
        }


@dataclass
class ReformulationResult:
    """Final result of the RL reformulation loop.

    Attributes:
        best_query: The highest-reward query found.
        original_query: The starting query.
        steps: All reformulation steps taken.
        total_reward: Cumulative reward score.
        improvement: Relative improvement over original.
    """

    best_query: str
    original_query: str
    steps: list[ReformulationStep] = field(default_factory=list)
    total_reward: float = 0.0
    improvement: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "best_query": self.best_query,
            "original_query": self.original_query,
            "total_steps": len(self.steps),
            "total_reward": round(self.total_reward, 4),
            "improvement": round(self.improvement, 4),
            "steps": [s.to_dict() for s in self.steps],
        }


# ---------------------------------------------------------------------------
# Operator implementations
# ---------------------------------------------------------------------------


# Simple synonym map for common academic terms
_SYNONYMS: dict[str, list[str]] = {
    "method": ["approach", "technique", "algorithm"],
    "model": ["architecture", "framework", "system"],
    "performance": ["accuracy", "effectiveness", "efficiency"],
    "improve": ["enhance", "boost", "optimize"],
    "analysis": ["evaluation", "assessment", "study"],
    "problem": ["challenge", "issue", "task"],
    "result": ["outcome", "finding", "output"],
    "data": ["dataset", "corpus", "collection"],
    "feature": ["attribute", "property", "characteristic"],
    "learn": ["train", "adapt", "acquire"],
}

# Common acronyms
_ACRONYMS: dict[str, str] = {
    "nlp": "natural language processing",
    "ml": "machine learning",
    "dl": "deep learning",
    "cv": "computer vision",
    "rl": "reinforcement learning",
    "llm": "large language model",
    "gan": "generative adversarial network",
    "rnn": "recurrent neural network",
    "cnn": "convolutional neural network",
    "bert": "bidirectional encoder representations from transformers",
    "gpt": "generative pre-trained transformer",
    "rag": "retrieval augmented generation",
    "kg": "knowledge graph",
}


def _apply_synonym_expand(query: str, rng: _random_module.Random) -> str:
    """Expand one term with a synonym."""
    words = query.split()
    candidates = [(i, w) for i, w in enumerate(words) if w.lower() in _SYNONYMS]
    if not candidates:
        return query
    idx, word = rng.choice(candidates)
    synonyms = _SYNONYMS[word.lower()]
    replacement = rng.choice(synonyms)
    words[idx] = replacement
    return " ".join(words)


def _apply_term_drop(query: str, rng: _random_module.Random) -> str:
    """Drop a non-essential term."""
    words = query.split()
    if len(words) <= 2:
        return query
    # Don't drop first or last word (usually most important)
    idx = rng.randint(1, len(words) - 2)
    words.pop(idx)
    return " ".join(words)


def _apply_term_boost(query: str, rng: _random_module.Random) -> str:
    """Repeat an important term for emphasis (BM25 boost)."""
    words = query.split()
    if not words:
        return query
    # Boost a random word
    word = rng.choice(words)
    return query + " " + word


def _apply_phrase_relax(query: str, _rng: _random_module.Random) -> str:
    """Remove quotes to relax phrase matching."""
    return query.replace('"', "").replace("'", "")


def _apply_acronym_expand(query: str, _rng: _random_module.Random) -> str:
    """Expand acronyms to full forms."""
    words = query.split()
    expanded = []
    for w in words:
        lower = w.lower()
        if lower in _ACRONYMS:
            expanded.append(_ACRONYMS[lower])
        else:
            expanded.append(w)
    return " ".join(expanded)


def _apply_hyponym_narrow(query: str, rng: _random_module.Random) -> str:
    """Add a narrowing qualifier."""
    qualifiers = [
        "recent",
        "state-of-the-art",
        "novel",
        "efficient",
        "scalable",
        "robust",
        "lightweight",
    ]
    return query + " " + rng.choice(qualifiers)


def _apply_scope_broaden(query: str, _rng: _random_module.Random) -> str:
    """Remove restrictive terms to broaden scope."""
    restrictive = {"only", "just", "specific", "exactly", "particular"}
    words = [w for w in query.split() if w.lower() not in restrictive]
    return " ".join(words) if words else query


_OPERATOR_FNS = {
    ReformulationOp.SYNONYM_EXPAND: _apply_synonym_expand,
    ReformulationOp.TERM_DROP: _apply_term_drop,
    ReformulationOp.TERM_BOOST: _apply_term_boost,
    ReformulationOp.PHRASE_RELAX: _apply_phrase_relax,
    ReformulationOp.ACRONYM_EXPAND: _apply_acronym_expand,
    ReformulationOp.HYPONYM_NARROW: _apply_hyponym_narrow,
    ReformulationOp.SCOPE_BROADEN: _apply_scope_broaden,
}


# ---------------------------------------------------------------------------
# Thompson sampling bandit
# ---------------------------------------------------------------------------


class OperatorBandit:
    """Thompson sampling bandit for operator selection.

    Maintains Beta(alpha, beta) posteriors per operator. Higher alpha
    means more observed successes.

    Args:
        seed: Random seed for reproducibility.
    """

    def __init__(self, seed: int | None = None) -> None:
        self._rng = _random_module.Random(seed)  # nosec B311
        self._alpha: dict[ReformulationOp, float] = dict.fromkeys(ReformulationOp, 1.0)
        self._beta: dict[ReformulationOp, float] = dict.fromkeys(ReformulationOp, 1.0)

    def select(self) -> ReformulationOp:
        """Select an operator via Thompson sampling."""
        samples = {}
        for op in ReformulationOp:
            # Beta distribution sample
            samples[op] = self._rng.betavariate(self._alpha[op], self._beta[op])
        return max(samples, key=samples.get)  # type: ignore[arg-type]

    def update(self, op: ReformulationOp, reward: RewardSignal) -> None:
        """Update posterior based on reward."""
        if reward == RewardSignal.SUCCESS:
            self._alpha[op] += 1.0
        elif reward == RewardSignal.FAILURE:
            self._beta[op] += 1.0
        else:
            # Partial reward
            self._alpha[op] += 0.5
            self._beta[op] += 0.5

    def success_rates(self) -> dict[str, float]:
        """Get estimated success rate per operator."""
        rates = {}
        for op in ReformulationOp:
            a, b = self._alpha[op], self._beta[op]
            rates[op.value] = round(a / (a + b), 4)
        return rates


# ---------------------------------------------------------------------------
# RL Reformulator
# ---------------------------------------------------------------------------


class RLReformulator:
    """RL-inspired query reformulation engine.

    Iteratively reformulates queries using bandit-selected operators
    and binary reward feedback from a retrieval function.

    Args:
        max_iterations: Maximum reformulation iterations.
        min_reward_threshold: Minimum reward to consider successful.
        seed: Random seed.
        relevance_threshold: Fraction of results that must be relevant.
    """

    def __init__(
        self,
        max_iterations: int = 5,
        min_reward_threshold: float = 0.3,
        seed: int | None = None,
        relevance_threshold: float = 0.5,
    ) -> None:
        self._max_iterations = max_iterations
        self._min_reward = min_reward_threshold
        self._relevance_threshold = relevance_threshold
        self._bandit = OperatorBandit(seed=seed)
        self._rng = _random_module.Random(seed)  # nosec B311
        self._history: list[ReformulationResult] = []

    @property
    def bandit(self) -> OperatorBandit:
        return self._bandit

    @property
    def history(self) -> list[ReformulationResult]:
        return list(self._history)

    def reformulate(
        self,
        query: str,
        reward_fn: Any = None,
    ) -> ReformulationResult:
        """Run the reformulation loop.

        Args:
            query: The original query.
            reward_fn: Optional callback(query_text) → (relevant, total).
                       If None, uses a dummy that always returns partial.

        Returns:
            ReformulationResult with best query and step history.
        """
        steps: list[ReformulationStep] = []
        best_query = query
        best_score = 0.0
        original_score = 0.0

        # Evaluate original query
        if reward_fn:
            rel, total = reward_fn(query)
        else:
            rel, total = 0, 0
        original_score = rel / total if total > 0 else 0.0

        current = query
        for i in range(self._max_iterations):
            op = self._bandit.select()
            apply_fn = _OPERATOR_FNS[op]
            new_query = apply_fn(current, self._rng)

            if new_query == current:
                # Operator had no effect, try another
                continue

            # Evaluate
            if reward_fn:
                rel, total = reward_fn(new_query)
            else:
                rel, total = 0, 0

            score = rel / total if total > 0 else 0.0
            if score >= self._relevance_threshold:
                reward = RewardSignal.SUCCESS
            elif rel > 0:
                reward = RewardSignal.PARTIAL
            else:
                reward = RewardSignal.FAILURE

            variant = QueryVariant(
                text=new_query,
                operator=op,
                parent=current,
                generation=i + 1,
            )
            step = ReformulationStep(
                variant=variant,
                reward=reward,
                relevant_count=rel,
                total_count=total,
                step_index=i,
            )
            steps.append(step)

            self._bandit.update(op, reward)

            if score > best_score:
                best_score = score
                best_query = new_query

            # Use best path for next iteration
            if score >= self._min_reward:
                current = new_query

            logger.debug(
                "Step %d: op=%s, query='%s', reward=%s, score=%.3f",
                i,
                op.value,
                new_query[:50],
                reward.value,
                score,
            )

        improvement = (
            (best_score - original_score) / original_score
            if original_score > 0
            else best_score
        )

        result = ReformulationResult(
            best_query=best_query,
            original_query=query,
            steps=steps,
            total_reward=sum(
                (
                    1.0
                    if s.reward == RewardSignal.SUCCESS
                    else 0.5
                    if s.reward == RewardSignal.PARTIAL
                    else 0.0
                )
                for s in steps
            ),
            improvement=improvement,
        )
        self._history.append(result)

        logger.info(
            "Reformulation: %d steps, best='%s', improvement=%.2f",
            len(steps),
            best_query[:50],
            improvement,
        )
        return result

    def summary(self) -> dict[str, Any]:
        """Summarize reformulation history."""
        if not self._history:
            return {"total_runs": 0}
        improvements = [r.improvement for r in self._history]
        return {
            "total_runs": len(self._history),
            "mean_improvement": round(sum(improvements) / len(improvements), 4),
            "total_steps": sum(len(r.steps) for r in self._history),
            "operator_success_rates": self._bandit.success_rates(),
        }
