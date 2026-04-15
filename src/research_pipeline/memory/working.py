"""Working memory: bounded per-stage context buffer.

Holds recent items (candidates, decisions, summaries) for the current
stage.  Automatically evicts oldest entries when capacity is exceeded.
Resets at stage boundaries.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Any

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
