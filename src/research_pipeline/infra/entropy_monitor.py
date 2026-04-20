"""Token-entropy monitor for in-context locking detection (A3-4).

The research report (gap A3-4) flags "in-context locking" as a failure mode
for long-running agent trajectories: the model's outputs progressively lose
lexical diversity, entering a narrow attractor that masks as confident
reasoning but is actually a collapsed distribution.

This module provides a lightweight detector: we compute rolling Shannon
entropy on tokenised LLM outputs over a sliding window. When entropy drops
below a threshold, or monotonically decreases over consecutive windows,
the monitor emits an alarm that the orchestrator can surface to the
evaluation logger.
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter, deque
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def shannon_entropy(tokens: list[str]) -> float:
    """Shannon entropy (base-2) of a token list.

    Returns 0.0 for empty input.
    """
    if not tokens:
        return 0.0
    counts = Counter(tokens)
    total = len(tokens)
    return -sum((c / total) * math.log2(c / total) for c in counts.values())


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokenisation."""
    return [m.group(0).lower() for m in _TOKEN_RE.finditer(text)]


@dataclass(frozen=True)
class EntropyReading:
    """A single entropy reading in the rolling series."""

    index: int
    entropy: float
    sample_size: int
    alarm: bool


@dataclass
class EntropyMonitor:
    """Sliding-window entropy monitor.

    Args:
        window_size: Number of samples kept for the rolling window.
        threshold: Absolute entropy floor below which an alarm fires.
        monotonic_drop_count: Number of consecutive entropy decreases
            that also triggers an alarm.
    """

    window_size: int = 5
    threshold: float = 2.0
    monotonic_drop_count: int = 3
    readings: list[EntropyReading] = field(default_factory=list)
    _window: deque[list[str]] = field(default_factory=deque, repr=False)

    def __post_init__(self) -> None:
        self._window = deque(maxlen=max(1, self.window_size))

    def observe(self, text: str) -> EntropyReading:
        """Ingest one LLM output and return its entropy reading."""
        tokens = tokenize(text)
        self._window.append(tokens)
        combined = [t for sample in self._window for t in sample]
        ent = shannon_entropy(combined)
        index = len(self.readings) + 1
        alarm = False
        if ent < self.threshold and len(combined) > 0:
            alarm = True
        if self._is_monotonic_drop(ent):
            alarm = True
        reading = EntropyReading(
            index=index,
            entropy=round(ent, 6),
            sample_size=len(combined),
            alarm=alarm,
        )
        self.readings.append(reading)
        if alarm:
            logger.warning(
                "Entropy alarm @%d: entropy=%.3f threshold=%.3f",
                index,
                ent,
                self.threshold,
            )
        return reading

    def _is_monotonic_drop(self, current: float) -> bool:
        """True if the last N readings (+ current) are strictly decreasing."""
        if len(self.readings) < self.monotonic_drop_count - 1:
            return False
        recent = [r.entropy for r in self.readings[-(self.monotonic_drop_count - 1) :]]
        values = [*recent, current]
        return all(values[i] > values[i + 1] for i in range(len(values) - 1))

    def recent_entropy(self) -> float | None:
        """Latest entropy reading, or ``None`` if no observations yet."""
        if not self.readings:
            return None
        return self.readings[-1].entropy

    def alarm_count(self) -> int:
        return sum(1 for r in self.readings if r.alarm)

    def reset(self) -> None:
        self.readings.clear()
        self._window.clear()
