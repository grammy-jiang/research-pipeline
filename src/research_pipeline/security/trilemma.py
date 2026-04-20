"""Defense Trilemma depth-aware telemetry (Theme 15 / Paper 111).

The Defense Trilemma paper (2604.06436) formalises content-security defense
as a choice among (1) input sanitisation, (2) output filtering, (3) depth
limits. Each call through a pipeline stage contributes to a cumulative
distortion budget: treating the per-stage transformation as Lipschitz with
constant ``K``, the composition of ``n`` stages has a worst-case
distortion proxy of ``K^n``.

We cannot compute a true Lipschitz constant without embeddings, so we use
three cheap edit-distance-style proxies per stage — token-budget inflation,
sanitisation-delta, and character edit ratio — and combine them into a
per-stage K estimate. The orchestrator can feed this monitor at every stage
boundary to get early warning before the K^n proxy exceeds a configured
budget.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


def _levenshtein(a: str, b: str) -> int:
    """Iterative Levenshtein distance with O(min(len_a, len_b)) space."""
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i]
        for j, cb in enumerate(b, 1):
            ins = curr[j - 1] + 1
            dele = prev[j] + 1
            sub = prev[j - 1] + (ca != cb)
            curr.append(min(ins, dele, sub))
        prev = curr
    return prev[-1]


@dataclass(frozen=True)
class StageDistortion:
    """Per-stage distortion reading."""

    stage: str
    token_growth: float
    edit_ratio: float
    sanitization_delta: float
    lipschitz_proxy: float


@dataclass
class TrilemmaMonitor:
    """Accumulates per-stage distortion and checks the K^n budget."""

    depth_budget: float = 8.0
    stages: list[StageDistortion] = field(default_factory=list)

    @property
    def cumulative_lipschitz(self) -> float:
        """Product of per-stage K proxies (the K^n scalar)."""
        k = 1.0
        for s in self.stages:
            k *= max(s.lipschitz_proxy, 1e-9)
        return k

    @property
    def alarm(self) -> bool:
        return self.cumulative_lipschitz > self.depth_budget

    def record(
        self,
        stage: str,
        before: str,
        after: str,
        *,
        sanitization_delta: float = 0.0,
    ) -> StageDistortion:
        """Record one stage's input→output transformation."""
        before_tokens = max(1, len(before.split()))
        after_tokens = max(1, len(after.split()))
        token_growth = after_tokens / before_tokens
        ed = _levenshtein(before, after)
        edit_ratio = ed / max(1, len(before))
        lipschitz_proxy = max(
            token_growth,
            1.0 + edit_ratio,
            1.0 + abs(sanitization_delta),
        )
        reading = StageDistortion(
            stage=stage,
            token_growth=round(token_growth, 6),
            edit_ratio=round(edit_ratio, 6),
            sanitization_delta=round(sanitization_delta, 6),
            lipschitz_proxy=round(lipschitz_proxy, 6),
        )
        self.stages.append(reading)
        if self.alarm:
            logger.warning(
                "Defense Trilemma budget exceeded at stage '%s': K^n=%.3f > %.3f",
                stage,
                self.cumulative_lipschitz,
                self.depth_budget,
            )
        return reading

    def reset(self) -> None:
        self.stages.clear()

    def summary(self) -> dict[str, float | int | bool]:
        return {
            "stage_count": len(self.stages),
            "cumulative_lipschitz": round(self.cumulative_lipschitz, 6),
            "budget": self.depth_budget,
            "alarm": self.alarm,
        }
