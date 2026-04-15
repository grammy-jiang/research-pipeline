"""Taint tracking for pipeline content.

Marks each piece of content with its provenance (source, stage) and
trust level. Taint propagates: if tainted content is used to generate
new content, the new content inherits the taint.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class TrustLevel(str, Enum):
    """Trust classification for content provenance."""

    TRUSTED = "trusted"
    SEMI_TRUSTED = "semi_trusted"
    UNTRUSTED = "untrusted"


@dataclass
class TaintLabel:
    """Taint metadata for a piece of content."""

    source: str
    stage: str
    trust_level: TrustLevel
    sanitized: bool = False
    classified: bool = False
    risk_flags: list[str] = field(default_factory=list)

    @property
    def is_safe(self) -> bool:
        """Content is safe if trusted or sanitized."""
        return self.trust_level == TrustLevel.TRUSTED or self.sanitized


class TaintTracker:
    """Tracks taint labels for pipeline content.

    Content is identified by a key (e.g., ``paper:2401.12345`` or
    ``abstract:2401.12345``).
    """

    def __init__(self) -> None:
        self._labels: dict[str, TaintLabel] = {}

    def mark(self, key: str, label: TaintLabel) -> None:
        """Mark content with a taint label."""
        self._labels[key] = label
        logger.debug("Taint: %s → %s (%s)", key, label.trust_level.value, label.source)

    def get(self, key: str) -> TaintLabel | None:
        """Get taint label for content."""
        return self._labels.get(key)

    def mark_sanitized(self, key: str) -> None:
        """Mark content as sanitized."""
        label = self._labels.get(key)
        if label is not None:
            label.sanitized = True
            logger.debug("Taint sanitized: %s", key)

    def mark_classified(self, key: str, risk_flags: list[str] | None = None) -> None:
        """Mark content as classified (security-checked)."""
        label = self._labels.get(key)
        if label is not None:
            label.classified = True
            if risk_flags:
                label.risk_flags = risk_flags

    def propagate(
        self, source_key: str, target_key: str, target_stage: str
    ) -> TaintLabel | None:
        """Propagate taint from source to derived content.

        The target inherits the source's trust level (or lower).
        """
        source_label = self._labels.get(source_key)
        if source_label is None:
            return None

        new_label = TaintLabel(
            source=source_label.source,
            stage=target_stage,
            trust_level=source_label.trust_level,
            sanitized=False,
            classified=False,
        )
        self._labels[target_key] = new_label
        return new_label

    def untrusted_keys(self) -> list[str]:
        """Get all keys with untrusted content that hasn't been sanitized."""
        return [
            key
            for key, label in self._labels.items()
            if label.trust_level == TrustLevel.UNTRUSTED and not label.sanitized
        ]

    def stats(self) -> dict[str, int]:
        """Taint tracking statistics."""
        counts: dict[str, int] = {
            "trusted": 0,
            "semi_trusted": 0,
            "untrusted": 0,
            "sanitized": 0,
        }
        for label in self._labels.values():
            counts[label.trust_level.value] = counts.get(label.trust_level.value, 0) + 1
            if label.sanitized:
                counts["sanitized"] += 1
        return counts

    def clear(self) -> None:
        """Clear all taint labels."""
        self._labels.clear()

    def __len__(self) -> int:
        return len(self._labels)
