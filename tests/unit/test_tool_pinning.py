"""Tests for infra.tool_pinning — hash-pinned tool definitions."""

from __future__ import annotations

import pytest

from research_pipeline.infra.tool_pinning import (
    PinnedTool,
    ToolIntegrityError,
    ToolRegistry,
    compute_definition_hash,
)

# ── Fixtures ─────────────────────────────────────────────────────────


def _sample_definition() -> dict:
    return {
        "name": "search_papers",
        "description": "Search for academic papers",
        "parameters": {
            "query": {"type": "string"},
            "max_results": {"type": "integer", "default": 50},
        },
    }


def _modified_definition() -> dict:
    d = _sample_definition()
    d["description"] = "TAMPERED description"
    return d


# ── compute_definition_hash ──────────────────────────────────────────


class TestComputeDefinitionHash:
    def test_deterministic(self) -> None:
        d = _sample_definition()
        h1 = compute_definition_hash(d)
        h2 = compute_definition_hash(d)
        assert h1 == h2

    def test_different_for_different_input(self) -> None:
        h1 = compute_definition_hash(_sample_definition())
        h2 = compute_definition_hash(_modified_definition())
        assert h1 != h2

    def test_key_order_independent(self) -> None:
        d1 = {"a": 1, "b": 2}
        d2 = {"b": 2, "a": 1}
        assert compute_definition_hash(d1) == compute_definition_hash(d2)

    def test_returns_hex_string(self) -> None:
        h = compute_definition_hash({"x": 1})
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)


# ── PinnedTool ───────────────────────────────────────────────────────


class TestPinnedTool:
    def test_frozen(self) -> None:
        pt = PinnedTool(name="t", definition_hash="abc123")
        with pytest.raises(AttributeError):
            pt.name = "other"  # type: ignore[misc]

    def test_default_version(self) -> None:
        pt = PinnedTool(name="t", definition_hash="abc123")
        assert pt.version == ""


# ── ToolRegistry.register ────────────────────────────────────────────


class TestToolRegistryRegister:
    def test_register_returns_pinned_tool(self) -> None:
        reg = ToolRegistry()
        pt = reg.register("search", _sample_definition())
        assert isinstance(pt, PinnedTool)
        assert pt.name == "search"
        assert len(pt.definition_hash) == 64

    def test_register_with_version(self) -> None:
        reg = ToolRegistry()
        pt = reg.register("search", _sample_definition(), version="1.0")
        assert pt.version == "1.0"

    def test_duplicate_raises(self) -> None:
        reg = ToolRegistry()
        reg.register("search", _sample_definition())
        with pytest.raises(ValueError, match="already registered"):
            reg.register("search", _sample_definition())

    def test_count_increments(self) -> None:
        reg = ToolRegistry()
        assert reg.count == 0
        reg.register("a", {"name": "a"})
        assert reg.count == 1
        reg.register("b", {"name": "b"})
        assert reg.count == 2


# ── ToolRegistry.verify ──────────────────────────────────────────────


class TestToolRegistryVerify:
    def test_verify_unmodified_passes(self) -> None:
        reg = ToolRegistry()
        reg.register("search", _sample_definition())
        assert reg.verify("search", _sample_definition()) is True

    def test_verify_self_check_passes(self) -> None:
        reg = ToolRegistry()
        reg.register("search", _sample_definition())
        # No definition arg → re-hashes stored definition
        assert reg.verify("search") is True

    def test_verify_tampered_strict_raises(self) -> None:
        reg = ToolRegistry(strict=True)
        reg.register("search", _sample_definition())
        with pytest.raises(ToolIntegrityError) as exc_info:
            reg.verify("search", _modified_definition())
        assert exc_info.value.tool_name == "search"

    def test_verify_tampered_nonstrict_returns_false(self) -> None:
        reg = ToolRegistry(strict=False)
        reg.register("search", _sample_definition())
        result = reg.verify("search", _modified_definition())
        assert result is False

    def test_verify_unknown_tool_raises(self) -> None:
        reg = ToolRegistry()
        with pytest.raises(KeyError, match="not registered"):
            reg.verify("nonexistent")


# ── ToolRegistry.verify_all ──────────────────────────────────────────


class TestToolRegistryVerifyAll:
    def test_all_pass(self) -> None:
        reg = ToolRegistry()
        reg.register("a", {"name": "a"})
        reg.register("b", {"name": "b"})
        results = reg.verify_all()
        assert results == {"a": True, "b": True}

    def test_mixed_results_nonstrict(self) -> None:
        reg = ToolRegistry(strict=False)
        reg.register("search", _sample_definition())
        # Tamper the stored definition directly for test
        reg._definitions["search"] = _modified_definition()
        results = reg.verify_all()
        assert results["search"] is False


# ── ToolRegistry.get / list_tools / unregister ───────────────────────


class TestToolRegistryOps:
    def test_get_existing(self) -> None:
        reg = ToolRegistry()
        reg.register("search", _sample_definition())
        pt = reg.get("search")
        assert pt is not None
        assert pt.name == "search"

    def test_get_missing(self) -> None:
        reg = ToolRegistry()
        assert reg.get("nope") is None

    def test_list_tools(self) -> None:
        reg = ToolRegistry()
        reg.register("a", {"name": "a"})
        reg.register("b", {"name": "b"})
        tools = reg.list_tools()
        assert len(tools) == 2
        names = {t.name for t in tools}
        assert names == {"a", "b"}

    def test_unregister(self) -> None:
        reg = ToolRegistry()
        reg.register("search", _sample_definition())
        assert reg.count == 1
        reg.unregister("search")
        assert reg.count == 0
        assert reg.get("search") is None

    def test_unregister_unknown_raises(self) -> None:
        reg = ToolRegistry()
        with pytest.raises(KeyError, match="not registered"):
            reg.unregister("nope")


# ── ToolIntegrityError ───────────────────────────────────────────────


class TestToolIntegrityError:
    def test_attributes(self) -> None:
        err = ToolIntegrityError("search", "expected_hash", "actual_hash")
        assert err.tool_name == "search"
        assert err.expected == "expected_hash"
        assert err.actual == "actual_hash"
        assert "search" in str(err)
