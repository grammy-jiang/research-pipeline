"""Working memory: bounded per-stage context buffer.

Holds recent items (candidates, decisions, summaries) for the current
stage.  Automatically evicts oldest entries when capacity is exceeded.
Resets at stage boundaries.

Supports **segment-level entries**: large values are automatically split
into ≤450-token segments so each buffer slot holds a retrieval-friendly
chunk (per Memory Survey, PVLDB 2026 recommendation).
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from research_pipeline.memory.segmentation import (
    DEFAULT_MAX_TOKENS,
    estimate_tokens,
    segment_text,
)

logger = logging.getLogger(__name__)

DEFAULT_CAPACITY = 50


@dataclass
class MemoryItem:
    """A single item in working memory."""

    key: str
    value: Any
    stage: str
    metadata: dict[str, str] = field(default_factory=dict)


class WorkingMemory:
    """Bounded FIFO buffer for per-stage context.

    Items are evicted oldest-first when capacity is exceeded.
    Call :meth:`reset` at stage boundaries to clear.
    """

    def __init__(self, capacity: int = DEFAULT_CAPACITY) -> None:
        self._capacity = max(1, capacity)
        self._buffer: deque[MemoryItem] = deque(maxlen=self._capacity)
        self._current_stage: str = ""

    @property
    def capacity(self) -> int:
        """Maximum number of items the buffer can hold."""
        return self._capacity

    @property
    def current_stage(self) -> str:
        """The stage that working memory is currently tracking."""
        return self._current_stage

    def __len__(self) -> int:
        return len(self._buffer)

    def add(
        self,
        key: str,
        value: Any,
        stage: str = "",
        metadata: dict[str, str] | None = None,
    ) -> None:
        """Add an item.  Evicts oldest if at capacity."""
        item = MemoryItem(
            key=key,
            value=value,
            stage=stage or self._current_stage,
            metadata=metadata or {},
        )
        self._buffer.append(item)

    def get(self, key: str) -> MemoryItem | None:
        """Get most recent item by key."""
        for item in reversed(self._buffer):
            if item.key == key:
                return item
        return None

    def get_all(self) -> list[MemoryItem]:
        """Get all items in order (oldest first)."""
        return list(self._buffer)

    def get_by_stage(self, stage: str) -> list[MemoryItem]:
        """Get items for a specific stage."""
        return [item for item in self._buffer if item.stage == stage]

    def reset(self, new_stage: str = "") -> list[MemoryItem]:
        """Reset working memory at stage boundary.

        Returns the items that were cleared (for consolidation to episodic).
        """
        cleared = list(self._buffer)
        self._buffer.clear()
        self._current_stage = new_stage
        logger.debug(
            "Working memory reset for stage '%s' (%d items cleared)",
            new_stage,
            len(cleared),
        )
        return cleared

    def summary(self) -> dict[str, Any]:
        """Summary stats for logging/audit."""
        return {
            "capacity": self._capacity,
            "size": len(self._buffer),
            "current_stage": self._current_stage,
            "stages": list({item.stage for item in self._buffer}),
        }

    def add_segmented(
        self,
        key: str,
        value: str,
        stage: str = "",
        metadata: dict[str, str] | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> int:
        """Add a text value, automatically splitting if it exceeds *max_tokens*.

        Each segment is stored as a separate :class:`MemoryItem` with a
        ``segment`` metadata field (e.g. ``"1/3"``).

        Returns the number of items actually added.
        """
        if not isinstance(value, str) or estimate_tokens(value) <= max_tokens:
            self.add(key, value, stage=stage, metadata=metadata)
            return 1

        parts = segment_text(value, max_tokens=max_tokens)
        total = len(parts)
        extra = metadata or {}
        for i, part in enumerate(parts):
            seg_meta = {**extra, "parent_key": key, "segment": f"{i + 1}/{total}"}
            self.add(f"{key}__seg{i}", part, stage=stage, metadata=seg_meta)
        return total
