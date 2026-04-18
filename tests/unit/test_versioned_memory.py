"""Tests for non-destructive versioned memory (v0.13.24).

Covers:
- VersionedEntry dataclass and is_active property
- VersionedMemory.put() with auto-versioning and supersession
- VersionedMemory.get() for active entries
- VersionedMemory.get_history() for full version chains
- VersionedMemory.get_active_entries() across multiple keys
- VersionedMemory.count() with active_only flag
- VersionedMemory.rollback() to previous version
- Multiple keys with independent version chains
"""

from __future__ import annotations

from pathlib import Path

from research_pipeline.memory.versioned import (
    VersionedEntry,
    VersionedMemory,
)

# ── VersionedEntry ──────────────────────────────────────────────────


class TestVersionedEntry:
    """VersionedEntry dataclass."""

    def test_active_when_no_supersession(self) -> None:
        e = VersionedEntry(entry_id="a:v1", key="a", value="val")
        assert e.is_active is True

    def test_not_active_when_superseded(self) -> None:
        e = VersionedEntry(entry_id="a:v1", key="a", value="val", superseded_by="a:v2")
        assert e.is_active is False

    def test_not_active_when_expired(self) -> None:
        e = VersionedEntry(
            entry_id="a:v1", key="a", value="val", valid_until="2024-01-01"
        )
        assert e.is_active is False

    def test_default_metadata(self) -> None:
        e = VersionedEntry(entry_id="a:v1", key="a", value="val")
        assert e.metadata == {}


# ── VersionedMemory.put() ──────────────────────────────────────────


class TestPut:
    """put() creates entries with auto-versioning."""

    def test_first_put_creates_v1(self, tmp_path: Path) -> None:
        vm = VersionedMemory(tmp_path / "test.db")
        entry = vm.put("topic", "transformers")
        assert entry.version == 1
        assert entry.key == "topic"
        assert entry.value == "transformers"
        assert entry.is_active is True
        vm.close()

    def test_second_put_creates_v2(self, tmp_path: Path) -> None:
        vm = VersionedMemory(tmp_path / "test.db")
        vm.put("topic", "transformers")
        e2 = vm.put("topic", "attention mechanisms")
        assert e2.version == 2
        assert e2.value == "attention mechanisms"
        assert e2.is_active is True
        vm.close()

    def test_old_version_superseded(self, tmp_path: Path) -> None:
        vm = VersionedMemory(tmp_path / "test.db")
        vm.put("topic", "v1")
        e2 = vm.put("topic", "v2")
        # Retrieve history to check e1 was superseded
        history = vm.get_history("topic")
        old = next(h for h in history if h.version == 1)
        assert old.superseded_by == e2.entry_id
        assert old.valid_until != ""
        assert old.is_active is False
        vm.close()

    def test_custom_entry_id(self, tmp_path: Path) -> None:
        vm = VersionedMemory(tmp_path / "test.db")
        entry = vm.put("key", "val", entry_id="custom-id")
        assert entry.entry_id == "custom-id"
        vm.close()

    def test_put_with_source_and_metadata(self, tmp_path: Path) -> None:
        vm = VersionedMemory(tmp_path / "test.db")
        entry = vm.put(
            "key",
            "val",
            source="run-123",
            metadata={"confidence": 0.9},
        )
        assert entry.source == "run-123"
        assert entry.metadata == {"confidence": 0.9}
        vm.close()


# ── VersionedMemory.get() ──────────────────────────────────────────


class TestGet:
    """get() returns only the active entry."""

    def test_get_returns_latest(self, tmp_path: Path) -> None:
        vm = VersionedMemory(tmp_path / "test.db")
        vm.put("k", "first")
        vm.put("k", "second")
        vm.put("k", "third")
        active = vm.get("k")
        assert active is not None
        assert active.value == "third"
        assert active.version == 3
        vm.close()

    def test_get_nonexistent_returns_none(self, tmp_path: Path) -> None:
        vm = VersionedMemory(tmp_path / "test.db")
        assert vm.get("nonexistent") is None
        vm.close()


# ── VersionedMemory.get_history() ──────────────────────────────────


class TestGetHistory:
    """get_history() returns all versions newest-first."""

    def test_history_order(self, tmp_path: Path) -> None:
        vm = VersionedMemory(tmp_path / "test.db")
        vm.put("k", "a")
        vm.put("k", "b")
        vm.put("k", "c")
        history = vm.get_history("k")
        assert len(history) == 3
        assert history[0].version == 3
        assert history[1].version == 2
        assert history[2].version == 1
        vm.close()

    def test_history_empty_key(self, tmp_path: Path) -> None:
        vm = VersionedMemory(tmp_path / "test.db")
        assert vm.get_history("empty") == []
        vm.close()


# ── VersionedMemory.get_active_entries() ───────────────────────────


class TestGetActiveEntries:
    """get_active_entries() returns one active entry per key."""

    def test_multiple_keys(self, tmp_path: Path) -> None:
        vm = VersionedMemory(tmp_path / "test.db")
        vm.put("a", "val-a")
        vm.put("b", "val-b")
        vm.put("a", "val-a2")  # supersedes first
        active = vm.get_active_entries()
        assert len(active) == 2
        keys = {e.key for e in active}
        assert keys == {"a", "b"}
        a_entry = next(e for e in active if e.key == "a")
        assert a_entry.value == "val-a2"
        vm.close()


# ── VersionedMemory.count() ────────────────────────────────────────


class TestCount:
    """count() with active_only flag."""

    def test_active_count(self, tmp_path: Path) -> None:
        vm = VersionedMemory(tmp_path / "test.db")
        vm.put("a", "v1")
        vm.put("a", "v2")
        vm.put("b", "v1")
        assert vm.count(active_only=True) == 2  # a:v2, b:v1
        vm.close()

    def test_total_count(self, tmp_path: Path) -> None:
        vm = VersionedMemory(tmp_path / "test.db")
        vm.put("a", "v1")
        vm.put("a", "v2")
        vm.put("b", "v1")
        assert vm.count(active_only=False) == 3
        vm.close()

    def test_empty_count(self, tmp_path: Path) -> None:
        vm = VersionedMemory(tmp_path / "test.db")
        assert vm.count() == 0
        vm.close()


# ── VersionedMemory.rollback() ─────────────────────────────────────


class TestRollback:
    """rollback() restores previous version."""

    def test_rollback_restores_previous(self, tmp_path: Path) -> None:
        vm = VersionedMemory(tmp_path / "test.db")
        vm.put("k", "old")
        vm.put("k", "new")
        restored = vm.rollback("k")
        assert restored is not None
        assert restored.value == "old"
        assert restored.version == 1
        assert restored.is_active is True
        vm.close()

    def test_rollback_no_history_returns_none(self, tmp_path: Path) -> None:
        vm = VersionedMemory(tmp_path / "test.db")
        vm.put("k", "only")
        assert vm.rollback("k") is None
        vm.close()

    def test_rollback_nonexistent_returns_none(self, tmp_path: Path) -> None:
        vm = VersionedMemory(tmp_path / "test.db")
        assert vm.rollback("nope") is None
        vm.close()

    def test_rollback_then_put_creates_new_version(self, tmp_path: Path) -> None:
        vm = VersionedMemory(tmp_path / "test.db")
        vm.put("k", "v1")
        vm.put("k", "v2")
        vm.rollback("k")
        e3 = vm.put("k", "v3")
        assert e3.version == 3
        assert vm.count(active_only=False) == 3
        vm.close()


# ── Independent key chains ──────────────────────────────────────────


class TestIndependentKeys:
    """Different keys have independent version chains."""

    def test_keys_dont_interfere(self, tmp_path: Path) -> None:
        vm = VersionedMemory(tmp_path / "test.db")
        vm.put("x", "x1")
        vm.put("x", "x2")
        vm.put("y", "y1")
        assert vm.get("x") is not None
        assert vm.get("x").value == "x2"  # type: ignore[union-attr]
        assert vm.get("y") is not None
        assert vm.get("y").value == "y1"  # type: ignore[union-attr]
        assert len(vm.get_history("x")) == 2
        assert len(vm.get_history("y")) == 1
        vm.close()
