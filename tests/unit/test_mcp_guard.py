"""Tests for MCP zero-trust security guard."""

from __future__ import annotations

import pytest

from research_pipeline.security.mcp_guard import (
    AuditEntry,
    AuthDecision,
    AuthResult,
    CapabilityPolicy,
    McpGuard,
    ToolRegistry,
    ToolSpec,
    TrustDomain,
    compute_args_hash,
    compute_schema_hash,
)

# --- compute_schema_hash ---


class TestComputeSchemaHash:
    """Tests for schema hash computation."""

    def test_deterministic(self) -> None:
        schema = {"query": "str", "limit": "int"}
        h1 = compute_schema_hash(schema)
        h2 = compute_schema_hash(schema)
        assert h1 == h2

    def test_key_order_independent(self) -> None:
        s1 = {"b": "int", "a": "str"}
        s2 = {"a": "str", "b": "int"}
        assert compute_schema_hash(s1) == compute_schema_hash(s2)

    def test_different_schemas_differ(self) -> None:
        s1 = {"query": "str"}
        s2 = {"query": "int"}
        assert compute_schema_hash(s1) != compute_schema_hash(s2)

    def test_empty_schema(self) -> None:
        h = compute_schema_hash({})
        assert len(h) == 64  # SHA-256 hex

    def test_nested_schema(self) -> None:
        schema = {"params": {"query": "str", "options": {"limit": 10}}}
        h = compute_schema_hash(schema)
        assert len(h) == 64

    def test_returns_hex_string(self) -> None:
        h = compute_schema_hash({"a": 1})
        assert all(c in "0123456789abcdef" for c in h)


class TestComputeArgsHash:
    """Tests for args hash computation."""

    def test_deterministic(self) -> None:
        args = {"query": "test", "limit": 10}
        h1 = compute_args_hash(args)
        h2 = compute_args_hash(args)
        assert h1 == h2

    def test_truncated_to_16(self) -> None:
        h = compute_args_hash({"x": 1})
        assert len(h) == 16

    def test_handles_non_serializable(self) -> None:
        """Uses default=str for non-JSON types."""
        from pathlib import Path

        h = compute_args_hash({"path": Path("/tmp/test")})
        assert len(h) == 16


# --- TrustDomain ---


class TestTrustDomain:
    """Tests for TrustDomain enum."""

    def test_values(self) -> None:
        assert TrustDomain.READ.value == "read"
        assert TrustDomain.WRITE.value == "write"
        assert TrustDomain.EXECUTE.value == "execute"
        assert TrustDomain.NETWORK.value == "network"
        assert TrustDomain.SYSTEM.value == "system"

    def test_from_string(self) -> None:
        assert TrustDomain("read") == TrustDomain.READ

    def test_invalid_value(self) -> None:
        with pytest.raises(ValueError):
            TrustDomain("invalid")


# --- AuthDecision ---


class TestAuthDecision:
    """Tests for AuthDecision enum."""

    def test_values(self) -> None:
        assert AuthDecision.ALLOWED.value == "allowed"
        assert AuthDecision.DENIED.value == "denied"
        assert AuthDecision.REQUIRES_APPROVAL.value == "requires_approval"


# --- ToolSpec ---


class TestToolSpec:
    """Tests for ToolSpec dataclass."""

    def test_frozen(self) -> None:
        spec = ToolSpec(
            name="test",
            schema_hash="abc",
            domain=TrustDomain.READ,
            schema={},
        )
        with pytest.raises(AttributeError):
            spec.name = "other"  # type: ignore[misc]

    def test_defaults(self) -> None:
        spec = ToolSpec(
            name="t",
            schema_hash="h",
            domain=TrustDomain.READ,
            schema={},
        )
        assert spec.description == ""
        assert spec.max_calls_per_minute == 60
        assert spec.requires_approval is False


# --- AuthResult ---


class TestAuthResult:
    """Tests for AuthResult dataclass."""

    def test_allowed_property_true(self) -> None:
        r = AuthResult(
            tool_name="t",
            decision=AuthDecision.ALLOWED,
            reason="ok",
        )
        assert r.allowed is True

    def test_allowed_property_false(self) -> None:
        r = AuthResult(
            tool_name="t",
            decision=AuthDecision.DENIED,
            reason="no",
        )
        assert r.allowed is False

    def test_requires_approval_not_allowed(self) -> None:
        r = AuthResult(
            tool_name="t",
            decision=AuthDecision.REQUIRES_APPROVAL,
            reason="needs approval",
        )
        assert r.allowed is False

    def test_defaults(self) -> None:
        r = AuthResult(
            tool_name="t",
            decision=AuthDecision.ALLOWED,
            reason="ok",
        )
        assert r.args == {}
        assert r.caller == ""
        assert r.timestamp == 0.0


# --- ToolRegistry ---


class TestToolRegistry:
    """Tests for ToolRegistry."""

    def test_register_tool(self) -> None:
        reg = ToolRegistry()
        spec = reg.register("search", schema={"query": "str"}, domain="read")
        assert spec.name == "search"
        assert spec.domain == TrustDomain.READ
        assert len(spec.schema_hash) == 64

    def test_register_with_enum_domain(self) -> None:
        reg = ToolRegistry()
        spec = reg.register("dl", schema={}, domain=TrustDomain.NETWORK)
        assert spec.domain == TrustDomain.NETWORK

    def test_get_tool(self) -> None:
        reg = ToolRegistry()
        reg.register("t1", schema={"a": 1})
        assert reg.get_tool("t1") is not None
        assert reg.get_tool("nonexistent") is None

    def test_list_tools(self) -> None:
        reg = ToolRegistry()
        reg.register("a", schema={})
        reg.register("b", schema={})
        assert sorted(reg.list_tools()) == ["a", "b"]

    def test_list_by_domain(self) -> None:
        reg = ToolRegistry()
        reg.register("r1", schema={}, domain="read")
        reg.register("w1", schema={}, domain="write")
        reg.register("r2", schema={}, domain="read")
        assert sorted(reg.list_by_domain(TrustDomain.READ)) == ["r1", "r2"]
        assert reg.list_by_domain(TrustDomain.WRITE) == ["w1"]

    def test_pin_tool(self) -> None:
        reg = ToolRegistry()
        reg.register("t", schema={"x": 1})
        h = reg.pin_tool("t")
        assert len(h) == 64
        assert reg.verify_integrity("t") is True

    def test_pin_nonexistent_raises(self) -> None:
        reg = ToolRegistry()
        with pytest.raises(KeyError, match="Tool not registered"):
            reg.pin_tool("missing")

    def test_pin_all(self) -> None:
        reg = ToolRegistry()
        reg.register("a", schema={"x": 1})
        reg.register("b", schema={"y": 2})
        hashes = reg.pin_all()
        assert set(hashes.keys()) == {"a", "b"}

    def test_verify_integrity_no_pin(self) -> None:
        reg = ToolRegistry()
        reg.register("t", schema={})
        assert reg.verify_integrity("t") is True

    def test_verify_integrity_tampered(self) -> None:
        reg = ToolRegistry()
        reg.register("t", schema={"original": True})
        reg.pin_tool("t")
        # Re-register with different schema
        reg.register("t", schema={"tampered": True})
        assert reg.verify_integrity("t") is False

    def test_verify_integrity_unregistered(self) -> None:
        reg = ToolRegistry()
        reg._pinned_hashes["ghost"] = "abc"
        assert reg.verify_integrity("ghost") is False


# --- CapabilityPolicy ---


class TestCapabilityPolicy:
    """Tests for CapabilityPolicy."""

    def test_grant_and_check(self) -> None:
        p = CapabilityPolicy()
        p.grant("user1", TrustDomain.READ)
        assert p.is_allowed("user1", TrustDomain.READ) is True
        assert p.is_allowed("user1", TrustDomain.WRITE) is False

    def test_grant_all(self) -> None:
        p = CapabilityPolicy()
        p.grant_all("admin")
        for d in TrustDomain:
            assert p.is_allowed("admin", d) is True

    def test_revoke(self) -> None:
        p = CapabilityPolicy()
        p.grant("user", TrustDomain.READ)
        p.grant("user", TrustDomain.WRITE)
        p.revoke("user", TrustDomain.WRITE)
        assert p.is_allowed("user", TrustDomain.READ) is True
        assert p.is_allowed("user", TrustDomain.WRITE) is False

    def test_deny_all(self) -> None:
        p = CapabilityPolicy()
        p.grant_all("blocked")
        p.deny_all("blocked")
        assert p.is_allowed("blocked", TrustDomain.READ) is False

    def test_unknown_caller(self) -> None:
        p = CapabilityPolicy()
        assert p.is_allowed("nobody", TrustDomain.READ) is False

    def test_revoke_nonexistent_caller(self) -> None:
        """Revoke on unknown caller should not raise."""
        p = CapabilityPolicy()
        p.revoke("nobody", TrustDomain.READ)  # No error


# --- McpGuard ---


def _make_guard(
    tools: list[tuple[str, dict, str]] | None = None,
    caller: str = "pipeline",
    grants: list[str] | None = None,
) -> McpGuard:
    """Helper to build guard with tools and policy."""
    reg = ToolRegistry()
    pol = CapabilityPolicy()

    for name, schema, domain in tools or []:
        reg.register(name, schema=schema, domain=domain)

    pol.grant_all(caller)
    for extra in grants or []:
        pol.grant_all(extra)

    reg.pin_all()
    return McpGuard(reg, pol)


class TestMcpGuard:
    """Tests for McpGuard authorization."""

    def test_allow_registered_tool(self) -> None:
        guard = _make_guard([("search", {"q": "str"}, "read")])
        r = guard.authorize("search", {"q": "test"}, caller="pipeline")
        assert r.allowed is True
        assert r.reason == "Authorized"

    def test_deny_unregistered_tool(self) -> None:
        guard = _make_guard([])
        r = guard.authorize("unknown", {}, caller="pipeline")
        assert r.allowed is False
        assert "not registered" in r.reason

    def test_deny_tampered_schema(self) -> None:
        guard = _make_guard([("t", {"original": True}, "read")])
        # Tamper schema
        guard.registry.register("t", schema={"tampered": True}, domain="read")
        r = guard.authorize("t", {}, caller="pipeline")
        assert r.allowed is False
        assert "integrity" in r.reason.lower()

    def test_deny_unauthorized_caller(self) -> None:
        guard = _make_guard([("t", {}, "write")])
        r = guard.authorize("t", {}, caller="stranger")
        assert r.allowed is False
        assert "lacks" in r.reason

    def test_rate_limit(self) -> None:
        reg = ToolRegistry()
        pol = CapabilityPolicy()
        reg.register("t", schema={}, domain="read", max_calls_per_minute=2)
        reg.pin_all()
        pol.grant_all("user")
        guard = McpGuard(reg, pol)

        assert guard.authorize("t", caller="user").allowed is True
        assert guard.authorize("t", caller="user").allowed is True
        assert guard.authorize("t", caller="user").allowed is False

    def test_requires_approval(self) -> None:
        reg = ToolRegistry()
        pol = CapabilityPolicy()
        reg.register(
            "dangerous",
            schema={},
            domain="execute",
            requires_approval=True,
        )
        reg.pin_all()
        pol.grant_all("user")
        guard = McpGuard(reg, pol)

        r = guard.authorize("dangerous", caller="user")
        assert r.decision == AuthDecision.REQUIRES_APPROVAL
        assert r.allowed is False

    def test_audit_trail(self) -> None:
        guard = _make_guard([("t", {}, "read")])
        guard.authorize("t", {"x": 1}, caller="pipeline")
        guard.authorize("missing", {}, caller="pipeline")

        log = guard.audit_log()
        assert len(log) == 2
        assert log[0].decision == AuthDecision.ALLOWED
        assert log[1].decision == AuthDecision.DENIED

    def test_audit_summary(self) -> None:
        guard = _make_guard([("t", {}, "read")])
        guard.authorize("t", caller="pipeline")
        guard.authorize("t", caller="pipeline")
        guard.authorize("missing", caller="pipeline")

        summary = guard.audit_summary()
        assert summary["total"] == 3
        assert summary["allowed"] == 2
        assert summary["denied"] == 1

    def test_audit_trim(self) -> None:
        guard = _make_guard([("t", {}, "read")])
        guard._max_audit = 5
        for _ in range(10):
            guard.authorize("t", caller="pipeline")
        assert len(guard.audit_log()) == 5

    def test_reset_audit(self) -> None:
        guard = _make_guard([("t", {}, "read")])
        guard.authorize("t", caller="pipeline")
        guard.reset_audit()
        assert len(guard.audit_log()) == 0

    def test_reset_rate_limits(self) -> None:
        reg = ToolRegistry()
        pol = CapabilityPolicy()
        reg.register("t", schema={}, domain="read", max_calls_per_minute=1)
        reg.pin_all()
        pol.grant_all("user")
        guard = McpGuard(reg, pol)

        assert guard.authorize("t", caller="user").allowed is True
        assert guard.authorize("t", caller="user").allowed is False

        guard.reset_rate_limits()
        assert guard.authorize("t", caller="user").allowed is True

    def test_args_included_in_result(self) -> None:
        guard = _make_guard([("t", {}, "read")])
        r = guard.authorize("t", {"key": "val"}, caller="pipeline")
        assert r.args == {"key": "val"}

    def test_default_args_empty_dict(self) -> None:
        guard = _make_guard([("t", {}, "read")])
        r = guard.authorize("t", caller="pipeline")
        assert r.args == {}

    def test_properties(self) -> None:
        guard = _make_guard()
        assert isinstance(guard.registry, ToolRegistry)
        assert isinstance(guard.policy, CapabilityPolicy)

    def test_multiple_domains(self) -> None:
        guard = _make_guard(
            [
                ("read_tool", {}, "read"),
                ("write_tool", {}, "write"),
                ("net_tool", {}, "network"),
            ]
        )
        # Pipeline has grant_all, so all pass
        assert guard.authorize("read_tool", caller="pipeline").allowed
        assert guard.authorize("write_tool", caller="pipeline").allowed
        assert guard.authorize("net_tool", caller="pipeline").allowed

    def test_domain_isolation(self) -> None:
        """A caller with only READ cannot use WRITE tools."""
        reg = ToolRegistry()
        pol = CapabilityPolicy()
        reg.register("reader", schema={}, domain="read")
        reg.register("writer", schema={}, domain="write")
        reg.pin_all()
        pol.grant("limited", TrustDomain.READ)

        guard = McpGuard(reg, pol)
        assert guard.authorize("reader", caller="limited").allowed
        assert not guard.authorize("writer", caller="limited").allowed


# --- AuditEntry ---


class TestAuditEntry:
    """Tests for AuditEntry dataclass."""

    def test_creation(self) -> None:
        entry = AuditEntry(
            tool_name="t",
            caller="c",
            decision=AuthDecision.ALLOWED,
            reason="ok",
            args_hash="abc123",
            timestamp=1000.0,
            domain=TrustDomain.READ,
        )
        assert entry.tool_name == "t"
        assert entry.domain == TrustDomain.READ

    def test_domain_optional(self) -> None:
        entry = AuditEntry(
            tool_name="t",
            caller="c",
            decision=AuthDecision.DENIED,
            reason="no",
            args_hash="x",
            timestamp=0.0,
        )
        assert entry.domain is None
