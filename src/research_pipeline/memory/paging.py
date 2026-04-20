"""MemGPT-style tiered memory paging.

MemGPT (report Theme 3 / Recommendation 3, Paper 48-adjacent) treats memory
as an OS-style paging hierarchy: items live in working memory while hot,
are paged out to episodic storage when cold, and can be retrieved back
("faulted in") on demand. This module adds that fault-handling layer on
top of the existing three-tier :class:`MemoryManager`.

We do not modify :class:`MemoryManager` itself (to preserve its test
contract); instead, :class:`PagedMemory` wraps it and provides ``get``,
``page_in``, ``page_out``, and ``faults`` counters.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from research_pipeline.memory.working import MemoryItem

logger = logging.getLogger(__name__)

if TYPE_CHECKING:  # pragma: no cover
    from research_pipeline.memory.manager import MemoryManager


@dataclass
class PagingStats:
    """Counters for paging activity."""

    working_hits: int = 0
    episodic_hits: int = 0
    semantic_hits: int = 0
    misses: int = 0
    page_in_count: int = 0
    page_out_count: int = 0

    @property
    def total_lookups(self) -> int:
        return self.working_hits + self.episodic_hits + self.semantic_hits + self.misses

    @property
    def fault_rate(self) -> float:
        """Fraction of lookups that faulted from a non-working tier."""
        t = self.total_lookups
        if t == 0:
            return 0.0
        return (self.episodic_hits + self.semantic_hits) / t


@dataclass
class _PageOutEntry:
    key: str
    value: Any
    stage: str
    metadata: dict[str, str] = field(default_factory=dict)


class PagedMemory:
    """Tiered paging wrapper over :class:`MemoryManager`.

    Lookup path on :meth:`get`:

    1. Working memory (hot tier) — O(1)-ish scan of the bounded deque.
    2. Episodic backing store (warm tier) — faulted in and re-seated in
       working memory.
    3. Semantic / KG store (cold tier) — faulted in; not automatically
       re-seated in working memory to avoid thrashing.

    When working memory is full, :meth:`page_out` demotes an item to a
    swap buffer that episodic memory can inspect for consolidation.
    """

    def __init__(self, manager: MemoryManager) -> None:
        self._manager = manager
        self._stats = PagingStats()
        self._swap: list[_PageOutEntry] = []

    @property
    def stats(self) -> PagingStats:
        return self._stats

    @property
    def swap(self) -> list[_PageOutEntry]:
        """Items that have been paged out of working memory."""
        return list(self._swap)

    # ── Core paging operations ────────────────────────────────────

    def page_in(self, key: str, value: Any, stage: str = "") -> MemoryItem:
        """Fault an item into working memory.

        If working memory is at capacity, the oldest item is demoted via
        :meth:`page_out` first.
        """
        working = self._manager.working
        if len(working) >= working.capacity:
            oldest = working.get_all()[0] if len(working) > 0 else None
            if oldest is not None:
                self._swap.append(
                    _PageOutEntry(
                        key=oldest.key,
                        value=oldest.value,
                        stage=oldest.stage,
                        metadata=dict(oldest.metadata),
                    )
                )
                self._stats.page_out_count += 1
        working.add(key, value, stage=stage)
        self._stats.page_in_count += 1
        item = working.get(key)
        if item is None:  # pragma: no cover – just added above
            msg = f"page_in: key {key!r} not found after add"
            raise RuntimeError(msg)
        return item

    def page_out(self, key: str) -> _PageOutEntry | None:
        """Evict a specific key from working memory to the swap buffer."""
        working = self._manager.working
        item = working.get(key)
        if item is None:
            return None
        # deque does not support direct removal by key; rebuild
        survivors = [i for i in working.get_all() if i.key != key]
        current_stage = working.current_stage
        working.reset(current_stage)
        for s in survivors:
            working.add(s.key, s.value, stage=s.stage, metadata=dict(s.metadata))
        entry = _PageOutEntry(
            key=item.key,
            value=item.value,
            stage=item.stage,
            metadata=dict(item.metadata),
        )
        self._swap.append(entry)
        self._stats.page_out_count += 1
        return entry

    def get(self, key: str) -> MemoryItem | None:
        """Lookup with tiered fault handling.

        Returns a :class:`MemoryItem` if found at any tier, else ``None``
        (counted as a miss).
        """
        working = self._manager.working
        hit = working.get(key)
        if hit is not None:
            self._stats.working_hits += 1
            return hit

        # Swap buffer: items that were paged out but not yet consolidated
        for entry in reversed(self._swap):
            if entry.key == key:
                self._stats.episodic_hits += 1
                return self.page_in(entry.key, entry.value, stage=entry.stage)

        # Cold tier: semantic store — best-effort lookup
        semantic = self._manager.semantic
        if hasattr(semantic, "get"):
            try:
                value = semantic.get(key)
            except Exception:
                value = None
            if value is not None:
                self._stats.semantic_hits += 1
                return self.page_in(key, value, stage="semantic")

        self._stats.misses += 1
        return None

    def reset_stats(self) -> None:
        """Reset all paging counters (useful for benchmarking)."""
        self._stats = PagingStats()

    def drain_swap(self) -> list[_PageOutEntry]:
        """Return and clear the swap buffer (e.g. for episodic consolidation)."""
        out = list(self._swap)
        self._swap.clear()
        return out
