"""Adaptive difficulty routing.

Routes queries to local (cheap) vs cloud (expensive) models based on
complexity scoring.  Easy queries (metadata lookup, simple formatting)
go to a local 7B model; hard queries (synthesis, evaluation) go to a
cloud API.  Provides 10-20× cost savings on routine tasks.

Complexity dimensions:
- Token count / query length
- Vocabulary richness (type-token ratio)
- Domain specificity (technical term density)
- Reasoning depth indicators (causal, comparative, multi-hop)
- Output schema complexity
"""

from __future__ import annotations

import logging
import math
import re
import statistics
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class DifficultyLevel(StrEnum):
    """Query difficulty classification."""

    TRIVIAL = "trivial"
    EASY = "easy"
    MODERATE = "moderate"
    HARD = "hard"
    EXPERT = "expert"


class RoutingTarget(StrEnum):
    """Where to route the query."""

    LOCAL = "local"
    CLOUD = "cloud"
    PREMIUM = "premium"


# ---------------------------------------------------------------------------
# Complexity features
# ---------------------------------------------------------------------------


# Technical / domain-specific terms that increase difficulty
_TECHNICAL_TERMS = frozenset(
    [
        "transformer",
        "attention",
        "embedding",
        "gradient",
        "backpropagation",
        "convolution",
        "recurrent",
        "encoder",
        "decoder",
        "tokenizer",
        "fine-tune",
        "pre-train",
        "benchmark",
        "ablation",
        "hyperparameter",
        "regularization",
        "dropout",
        "batch normalization",
        "cross-entropy",
        "softmax",
        "sigmoid",
        "relu",
        "perceptron",
        "autoregressive",
        "diffusion",
        "variational",
        "bayesian",
        "markov",
        "monte carlo",
        "reinforcement learning",
        "reward",
        "policy",
        "episodic",
        "ontology",
        "knowledge graph",
        "entity",
        "relation",
        "triple",
        "citation",
        "h-index",
        "impact factor",
        "peer review",
        "methodology",
        "hypothesis",
        "correlation",
        "causation",
        "statistical significance",
        "p-value",
        "confidence interval",
    ]
)

# Reasoning depth indicators
_REASONING_INDICATORS = frozenset(
    [
        "because",
        "therefore",
        "however",
        "although",
        "whereas",
        "compare",
        "contrast",
        "evaluate",
        "synthesize",
        "analyze",
        "implications",
        "trade-off",
        "trade-offs",
        "consequences",
        "assuming",
        "given that",
        "if and only if",
        "necessary",
        "sufficient",
        "prove",
        "demonstrate",
        "derive",
        "multi-step",
        "chain of thought",
        "reasoning",
    ]
)


@dataclass(frozen=True)
class ComplexityFeatures:
    """Extracted complexity features from a query.

    Attributes:
        token_count: Number of whitespace-separated tokens.
        unique_ratio: Type-token ratio (vocabulary richness).
        technical_density: Fraction of tokens that are technical terms.
        reasoning_depth: Count of reasoning indicator matches.
        question_count: Number of questions (? marks).
        sentence_count: Number of sentences.
        avg_word_length: Average word length in characters.
        has_code: Whether the query contains code-like patterns.
    """

    token_count: int = 0
    unique_ratio: float = 0.0
    technical_density: float = 0.0
    reasoning_depth: int = 0
    question_count: int = 0
    sentence_count: int = 0
    avg_word_length: float = 0.0
    has_code: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "token_count": self.token_count,
            "unique_ratio": round(self.unique_ratio, 4),
            "technical_density": round(self.technical_density, 4),
            "reasoning_depth": self.reasoning_depth,
            "question_count": self.question_count,
            "sentence_count": self.sentence_count,
            "avg_word_length": round(self.avg_word_length, 2),
            "has_code": self.has_code,
        }


def extract_features(text: str) -> ComplexityFeatures:
    """Extract complexity features from a text query.

    Args:
        text: The query or prompt text.

    Returns:
        ComplexityFeatures with computed metrics.
    """
    if not text.strip():
        return ComplexityFeatures()

    words = text.lower().split()
    token_count = len(words)
    unique_words = set(words)
    unique_ratio = len(unique_words) / token_count if token_count > 0 else 0.0

    lower_text = text.lower()
    tech_matches = sum(1 for t in _TECHNICAL_TERMS if t in lower_text)
    technical_density = tech_matches / token_count if token_count > 0 else 0.0

    reasoning_depth = sum(1 for r in _REASONING_INDICATORS if r in lower_text)

    question_count = text.count("?")
    sentence_count = max(1, len(re.split(r"[.!?]+", text.strip())))

    char_lengths = [len(w) for w in words]
    avg_word_length = statistics.mean(char_lengths) if char_lengths else 0.0

    has_code = bool(re.search(r"```|def |class |import |function |=>|\{.*\}", text))

    return ComplexityFeatures(
        token_count=token_count,
        unique_ratio=unique_ratio,
        technical_density=technical_density,
        reasoning_depth=reasoning_depth,
        question_count=question_count,
        sentence_count=sentence_count,
        avg_word_length=avg_word_length,
        has_code=has_code,
    )


# ---------------------------------------------------------------------------
# Difficulty scoring
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DifficultyScore:
    """Composite difficulty assessment.

    Attributes:
        level: Classified difficulty level.
        score: Numeric score [0, 1].
        features: The extracted complexity features.
        target: Recommended routing target.
        confidence: How confident the classifier is [0, 1].
        reasoning: Why this difficulty was assigned.
    """

    level: DifficultyLevel
    score: float
    features: ComplexityFeatures
    target: RoutingTarget
    confidence: float = 1.0
    reasoning: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "level": self.level.value,
            "score": round(self.score, 4),
            "target": self.target.value,
            "confidence": round(self.confidence, 4),
            "reasoning": self.reasoning,
            "features": self.features.to_dict(),
        }


# Default thresholds for difficulty levels
_LEVEL_THRESHOLDS: list[tuple[float, DifficultyLevel]] = [
    (0.2, DifficultyLevel.TRIVIAL),
    (0.4, DifficultyLevel.EASY),
    (0.6, DifficultyLevel.MODERATE),
    (0.8, DifficultyLevel.HARD),
    (1.0, DifficultyLevel.EXPERT),
]

# Routing rules: difficulty → target
_ROUTING_MAP: dict[DifficultyLevel, RoutingTarget] = {
    DifficultyLevel.TRIVIAL: RoutingTarget.LOCAL,
    DifficultyLevel.EASY: RoutingTarget.LOCAL,
    DifficultyLevel.MODERATE: RoutingTarget.CLOUD,
    DifficultyLevel.HARD: RoutingTarget.CLOUD,
    DifficultyLevel.EXPERT: RoutingTarget.PREMIUM,
}


def score_difficulty(
    text: str,
    features: ComplexityFeatures | None = None,
) -> DifficultyScore:
    """Score the difficulty of a query.

    Combines multiple complexity features into a single [0, 1] score
    using a weighted sum with sigmoid normalization.

    Args:
        text: The query text.
        features: Pre-computed features (computed if None).

    Returns:
        DifficultyScore with level, target, and reasoning.
    """
    feat = features or extract_features(text)

    # Weighted feature combination
    raw = (
        0.15 * min(feat.token_count / 100, 1.0)
        + 0.10 * feat.unique_ratio
        + 0.25 * min(feat.technical_density * 5, 1.0)
        + 0.20 * min(feat.reasoning_depth / 5, 1.0)
        + 0.10 * min(feat.question_count / 3, 1.0)
        + 0.10 * min(feat.avg_word_length / 10, 1.0)
        + 0.10 * (1.0 if feat.has_code else 0.0)
    )

    # Sigmoid normalization
    score = 1.0 / (1.0 + math.exp(-6 * (raw - 0.5)))

    # Map to difficulty level
    level = DifficultyLevel.EXPERT
    for threshold, lvl in _LEVEL_THRESHOLDS:
        if score <= threshold:
            level = lvl
            break

    target = _ROUTING_MAP[level]

    reasons = []
    if feat.technical_density > 0.05:
        reasons.append(f"technical_density={feat.technical_density:.2f}")
    if feat.reasoning_depth > 2:
        reasons.append(f"reasoning_depth={feat.reasoning_depth}")
    if feat.token_count > 50:
        reasons.append(f"long_query={feat.token_count} tokens")
    if feat.has_code:
        reasons.append("contains_code")

    confidence = 0.5 + 0.5 * abs(score - 0.5) * 2

    return DifficultyScore(
        level=level,
        score=score,
        features=feat,
        target=target,
        confidence=confidence,
        reasoning="; ".join(reasons) if reasons else "standard_query",
    )


# ---------------------------------------------------------------------------
# Difficulty router
# ---------------------------------------------------------------------------


@dataclass
class RoutingDecision:
    """A routing decision with audit trail."""

    query_hash: str
    target: RoutingTarget
    difficulty: DifficultyScore
    override: bool = False
    override_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_hash": self.query_hash,
            "target": self.target.value,
            "difficulty": self.difficulty.to_dict(),
            "override": self.override,
            "override_reason": self.override_reason,
        }


class DifficultyRouter:
    """Route queries based on adaptive difficulty scoring.

    Supports per-stage overrides (e.g. force 'security_gate' to PREMIUM)
    and tracks routing statistics.

    Args:
        stage_overrides: Stage name → forced RoutingTarget.
        custom_routing: DifficultyLevel → RoutingTarget overrides.
    """

    def __init__(
        self,
        stage_overrides: dict[str, RoutingTarget] | None = None,
        custom_routing: dict[DifficultyLevel, RoutingTarget] | None = None,
    ) -> None:
        self._stage_overrides = stage_overrides or {}
        self._routing_map = dict(_ROUTING_MAP)
        if custom_routing:
            self._routing_map.update(custom_routing)
        self._history: list[RoutingDecision] = []

    @property
    def history(self) -> list[RoutingDecision]:
        return list(self._history)

    def route(
        self,
        query: str,
        stage: str = "",
    ) -> RoutingDecision:
        """Route a query to the appropriate target.

        Args:
            query: The query or prompt text.
            stage: Optional pipeline stage name for override lookup.

        Returns:
            RoutingDecision with target and audit trail.
        """
        import hashlib

        query_hash = hashlib.sha256(query.encode()).hexdigest()[:16]
        difficulty = score_difficulty(query)

        # Check stage override
        if stage and stage in self._stage_overrides:
            forced = self._stage_overrides[stage]
            decision = RoutingDecision(
                query_hash=query_hash,
                target=forced,
                difficulty=difficulty,
                override=True,
                override_reason=f"stage_override: {stage} → {forced.value}",
            )
        else:
            target = self._routing_map.get(difficulty.level, RoutingTarget.CLOUD)
            decision = RoutingDecision(
                query_hash=query_hash,
                target=target,
                difficulty=difficulty,
            )

        self._history.append(decision)
        logger.info(
            "Routed query to %s (difficulty=%s, score=%.3f, stage=%s)",
            decision.target.value,
            difficulty.level.value,
            difficulty.score,
            stage or "none",
        )
        return decision

    def cost_summary(self) -> dict[str, Any]:
        """Summarize routing decisions and estimated cost savings."""
        if not self._history:
            return {"total_queries": 0}

        targets = [d.target for d in self._history]
        local_count = sum(1 for t in targets if t == RoutingTarget.LOCAL)
        cloud_count = sum(1 for t in targets if t == RoutingTarget.CLOUD)
        premium_count = sum(1 for t in targets if t == RoutingTarget.PREMIUM)

        # Cost model: local=1, cloud=10, premium=30
        actual_cost = local_count * 1 + cloud_count * 10 + premium_count * 30
        naive_cost = len(self._history) * 10  # assume all cloud without routing
        savings = 1.0 - (actual_cost / naive_cost) if naive_cost > 0 else 0.0

        return {
            "total_queries": len(self._history),
            "local": local_count,
            "cloud": cloud_count,
            "premium": premium_count,
            "estimated_savings": round(max(0.0, savings), 4),
            "overrides": sum(1 for d in self._history if d.override),
        }
