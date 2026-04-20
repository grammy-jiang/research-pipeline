"""Unit tests for :mod:`research_pipeline.memory.paging`."""

from __future__ import annotations

from research_pipeline.memory.manager import MemoryManager
from research_pipeline.memory.paging import PagedMemory, PagingStats


def _build(tmp_path, capacity: int = 3) -> MemoryManager:
    return MemoryManager(
        working_capacity=capacity,
        episodic_path=tmp_path / "ep.sqlite",
        kg_path=tmp_path / "kg.json",
    )


def test_page_in_hits_working_tier(tmp_path):
    m = _build(tmp_path)
    try:
        paged = PagedMemory(m)
        paged.page_in("k1", "v1", stage="search")
        item = paged.get("k1")
        assert item is not None
        assert item.value == "v1"
        assert paged.stats.working_hits == 1
        assert paged.stats.misses == 0
    finally:
        m.close()


def test_page_in_demotes_oldest_to_swap(tmp_path):
    m = _build(tmp_path, capacity=2)
    try:
        paged = PagedMemory(m)
        paged.page_in("k1", "v1")
        paged.page_in("k2", "v2")
        paged.page_in("k3", "v3")  # evicts k1
        assert paged.stats.page_out_count == 1
        swap_keys = [e.key for e in paged.swap]
        assert "k1" in swap_keys
    finally:
        m.close()


def test_miss_increments_counter(tmp_path):
    m = _build(tmp_path)
    try:
        paged = PagedMemory(m)
        result = paged.get("nope")
        assert result is None
        assert paged.stats.misses == 1
        assert paged.stats.fault_rate == 0.0
    finally:
        m.close()


def test_swap_fault_refetches_and_reseats(tmp_path):
    m = _build(tmp_path, capacity=2)
    try:
        paged = PagedMemory(m)
        paged.page_in("k1", "v1")
        paged.page_in("k2", "v2")
        paged.page_in("k3", "v3")  # k1 → swap
        item = paged.get("k1")  # fault from swap
        assert item is not None
        assert item.value == "v1"
        assert paged.stats.episodic_hits == 1
        assert paged.stats.page_in_count >= 4
    finally:
        m.close()


def test_explicit_page_out_removes_from_working(tmp_path):
    m = _build(tmp_path, capacity=3)
    try:
        paged = PagedMemory(m)
        paged.page_in("k1", "v1")
        paged.page_in("k2", "v2")
        entry = paged.page_out("k1")
        assert entry is not None
        assert entry.key == "k1"
        assert m.working.get("k1") is None
        assert m.working.get("k2") is not None
    finally:
        m.close()


def test_page_out_missing_returns_none(tmp_path):
    m = _build(tmp_path)
    try:
        paged = PagedMemory(m)
        assert paged.page_out("nonexistent") is None
    finally:
        m.close()


def test_drain_swap_clears_buffer(tmp_path):
    m = _build(tmp_path, capacity=1)
    try:
        paged = PagedMemory(m)
        paged.page_in("k1", "v1")
        paged.page_in("k2", "v2")  # k1 → swap
        drained = paged.drain_swap()
        assert len(drained) == 1
        assert paged.swap == []
    finally:
        m.close()


def test_reset_stats_zeroes_counters(tmp_path):
    m = _build(tmp_path)
    try:
        paged = PagedMemory(m)
        paged.page_in("k1", "v1")
        paged.get("k1")
        paged.reset_stats()
        assert paged.stats == PagingStats()
    finally:
        m.close()


def test_fault_rate_computation():
    stats = PagingStats(working_hits=3, episodic_hits=1, semantic_hits=1, misses=5)
    assert stats.total_lookups == 10
    assert stats.fault_rate == 0.2
