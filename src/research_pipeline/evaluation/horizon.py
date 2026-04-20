"""Unified Horizon Metric (UHM) — A3-5 remaining-gap closure.

Resolves the A3-5 AMBER sub-gap from the Deep Research Report: "Building
blocks exist (normalized scoring, difficulty vectors); no unified proposal."

The Unified Horizon Metric combines four well-established signals from the
long-horizon-agent literature into a single scalar in ``[0, 1]``:

1. **Difficulty-weighted score** — Claw-Eval / ACE-Bench normalized task
   quality, up-weighted by task difficulty so that solving a hard task scores
   higher than solving an easy one at the same raw quality.
2. **Horizon efficiency** — achievement relative to the benchmark horizon
   (tokens, turns, tool calls). Long trajectories earn more credit, with a
   square-root saturation to reflect diminishing returns.
3. **Stability factor** — penalizes in-context locking as formalized by
   UltraHorizon: declining token entropy over the trajectory indicates
   agent behavior is narrowing and is a reliable failure predictor.
4. **Reliability floor** — Pass[k] reliability (all-k-correct probability)
   acts as a multiplicative gate so a single unreliable run cannot inflate
   the score.

The three positive signals are combined as a geometric mean (so a weak
component caps the score) and then multiplied by the reliability floor:

    UHM = (difficulty_score * horizon_eff * stability) ** (1/3) * reliability

All inputs are clamped to ``[0, 1]`` before combination, and the output is
guaranteed to be in ``[0, 1]``. Components and inputs are preserved on the
result for auditability and downstream evaluation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "HorizonInputs",
    "UnifiedHorizonResult",
    "compute_unified_horizon_metric",
]


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Clamp *value* to the closed interval ``[lo, hi]``."""
    if value < lo:
        return lo
    if value > hi:
        return hi
    return value


@dataclass
class HorizonInputs:
    """Inputs for the Unified Horizon Metric.

    Attributes:
        normalized_score: Normalized task quality in ``[0, 1]`` (e.g., a
            Claw-Eval or RACE score).
        difficulty: Task difficulty in ``[0, 1]`` (0 = trivial, 1 = hardest).
        achieved_steps: Trajectory length actually completed (turns, tool
            calls, or tokens — must match ``target_steps`` units).
        target_steps: Benchmark target horizon. Determines the saturation
            point of the horizon-efficiency term.
        entropy_trend: UltraHorizon-style token-entropy slope across the
            trajectory (negative = declining / locking, 0 = stable, positive
            = maintained diversity).
        reliability: Optional Pass[k] reliability floor in ``[0, 1]`` from
            ``evaluation.dual_metrics``. Defaults to ``1.0`` when only a
            single run is scored.
        difficulty_boost: How much weight to give difficulty (default 0.5;
            the difficulty-weighted score is ``score * (1 + boost * diff)``).
    """

    normalized_score: float
    difficulty: float
    achieved_steps: int
    target_steps: int
    entropy_trend: float = 0.0
    reliability: float = 1.0
    difficulty_boost: float = 0.5


@dataclass
class UnifiedHorizonResult:
    """Output of the Unified Horizon Metric computation."""

    uhm: float
    difficulty_weighted_score: float
    horizon_efficiency: float
    stability_factor: float
    reliability: float
    components: dict[str, float] = field(default_factory=dict)
    inputs: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {
            "uhm": self.uhm,
            "difficulty_weighted_score": self.difficulty_weighted_score,
            "horizon_efficiency": self.horizon_efficiency,
            "stability_factor": self.stability_factor,
            "reliability": self.reliability,
            "components": dict(self.components),
            "inputs": dict(self.inputs),
        }


def _difficulty_weighted_score(
    normalized_score: float, difficulty: float, boost: float
) -> float:
    """Up-weight normalized score by difficulty, clamped to ``[0, 1]``."""
    score = _clamp(normalized_score)
    diff = _clamp(difficulty)
    boosted = score * (1.0 + boost * diff)
    return _clamp(boosted)


def _horizon_efficiency(achieved: int, target: int) -> float:
    """Square-root-saturated horizon efficiency.

    Returns ``0.0`` for zero/negative steps and ``1.0`` when ``achieved >=
    target``. Between, grows as ``sqrt(achieved / target)`` so that doubling
    trajectory length roughly multiplies credit by ``sqrt(2)``.
    """
    if achieved <= 0 or target <= 0:
        return 0.0
    ratio = achieved / target
    if ratio >= 1.0:
        return 1.0
    return math.sqrt(ratio)


def _stability_factor(entropy_trend: float) -> float:
    """Map UltraHorizon token-entropy trend to a ``[0, 1]`` stability score.

    - ``trend <= -0.5``: severe in-context locking → 0.5.
    - ``trend = 0``:     stable → 1.0.
    - ``trend > 0``:     widening exploration → capped at 1.0 (not a bonus).

    The half-floor of 0.5 keeps one bad signal from zeroing the metric; the
    geometric mean still propagates the penalty.
    """
    # Clamp the entropy trend before mapping, then interpolate.
    trend = entropy_trend
    if trend <= -0.5:
        return 0.5
    if trend >= 0.0:
        return 1.0
    # Linear interpolation between (-0.5, 0.5) and (0, 1.0)
    return 0.5 + (trend + 0.5)


def compute_unified_horizon_metric(inputs: HorizonInputs) -> UnifiedHorizonResult:
    """Compute the Unified Horizon Metric from *inputs*.

    See module docstring for the formula. All components are returned on
    the result for auditability.
    """
    diff_score = _difficulty_weighted_score(
        inputs.normalized_score, inputs.difficulty, inputs.difficulty_boost
    )
    horizon_eff = _horizon_efficiency(inputs.achieved_steps, inputs.target_steps)
    stability = _stability_factor(inputs.entropy_trend)
    reliability = _clamp(inputs.reliability)

    # Geometric mean of the three positive signals, gated by reliability.
    geo_mean = (diff_score * horizon_eff * stability) ** (1.0 / 3.0)
    uhm = _clamp(geo_mean * reliability)

    return UnifiedHorizonResult(
        uhm=uhm,
        difficulty_weighted_score=diff_score,
        horizon_efficiency=horizon_eff,
        stability_factor=stability,
        reliability=reliability,
        components={
            "geometric_mean": geo_mean,
            "reliability_gate": reliability,
        },
        inputs={
            "normalized_score": inputs.normalized_score,
            "difficulty": inputs.difficulty,
            "achieved_steps": inputs.achieved_steps,
            "target_steps": inputs.target_steps,
            "entropy_trend": inputs.entropy_trend,
            "reliability": inputs.reliability,
            "difficulty_boost": inputs.difficulty_boost,
        },
    )
