"""MCP zero-trust security: hash-pinned tools, trust domains, capability control.

Implements MCPSHIELD-inspired 4-layer defense for MCP tool interactions:
1. Tool pinning: SHA-256 hash verification of tool schemas
2. Trust domains: classify tools into trust boundaries
3. Capability control: restrict tool actions by domain
4. Audit trail: log every tool invocation with context

Usage::

    registry = ToolRegistry()
    registry.register("search", schema={"query": "str"}, domain="read")
    registry.pin_tool("search")  # computes hash

    guard = McpGuard(registry)
    result = guard.authorize("search", {"query": "transformers"}, caller="pipeline")
    if result.allowed:
        execute_tool(result.tool_name, result.args)
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from enum import StrEnum

logger = logging.getLogger(__name__)


class TrustDomain(StrEnum):
    """Trust domain classification for MCP tools."""

    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"
    NETWORK = "network"
    SYSTEM = "system"


class AuthDecision(StrEnum):
    """Authorization decision result."""

    ALLOWED = "allowed"
    DENIED = "denied"
    REQUIRES_APPROVAL = "requires_approval"


@dataclass(frozen=True)
class ToolSpec:
    """Registered tool specification with security metadata."""

    name: str
    schema_hash: str
    domain: TrustDomain
    schema: dict
    description: str = ""
    max_calls_per_minute: int = 60
    requires_approval: bool = False


@dataclass(frozen=True)
class AuthResult:
    """Result of an authorization check."""

    tool_name: str
    decision: AuthDecision
    reason: str
    args: dict = field(default_factory=dict)
    caller: str = ""
    timestamp: float = 0.0

    @property
    def allowed(self) -> bool:
        """Check if the action is allowed."""
        return self.decision == AuthDecision.ALLOWED


@dataclass
class AuditEntry:
    """Audit trail entry for tool invocations."""

    tool_name: str
    caller: str
    decision: AuthDecision
    reason: str
    args_hash: str
    timestamp: float
    domain: TrustDomain | None = None


def compute_schema_hash(schema: dict) -> str:
    """Compute deterministic SHA-256 hash of a tool schema.

    Args:
        schema: Tool schema dictionary.

    Returns:
        Hex-encoded SHA-256 hash.
    """
    canonical = json.dumps(schema, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def compute_args_hash(args: dict) -> str:
    """Compute SHA-256 hash of tool arguments for audit.

    Args:
        args: Tool arguments dictionary.

    Returns:
        Hex-encoded SHA-256 hash (first 16 chars).
    """
    canonical = json.dumps(args, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


class ToolRegistry:
    """Registry for MCP tools with hash pinning and domain classification.

    Tracks registered tools and their expected schema hashes.
    Detects schema tampering by comparing current vs pinned hashes.
    """

    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}
        self._pinned_hashes: dict[str, str] = {}

    def register(
        self,
        name: str,
        schema: dict,
        domain: str | TrustDomain = TrustDomain.READ,
        description: str = "",
        max_calls_per_minute: int = 60,
        requires_approval: bool = False,
    ) -> ToolSpec:
        """Register a tool with its schema and security metadata.

        Args:
            name: Tool name.
            schema: Tool input schema dictionary.
            domain: Trust domain classification.
            description: Human-readable description.
            max_calls_per_minute: Rate limit.
            requires_approval: Whether invocations need approval.

        Returns:
            The registered ToolSpec.
        """
        if isinstance(domain, str):
            domain = TrustDomain(domain)

        schema_hash = compute_schema_hash(schema)
        spec = ToolSpec(
            name=name,
            schema_hash=schema_hash,
            domain=domain,
            schema=schema,
            description=description,
            max_calls_per_minute=max_calls_per_minute,
            requires_approval=requires_approval,
        )
        self._tools[name] = spec
        return spec

    def pin_tool(self, name: str) -> str:
        """Pin a tool's current schema hash for integrity verification.

        Args:
            name: Tool name (must be registered).

        Returns:
            The pinned hash.

        Raises:
            KeyError: If tool is not registered.
        """
        if name not in self._tools:
            raise KeyError(f"Tool not registered: {name}")
        schema_hash = self._tools[name].schema_hash
        self._pinned_hashes[name] = schema_hash
        logger.debug("Pinned tool %s: %s", name, schema_hash[:16])
        return schema_hash

    def pin_all(self) -> dict[str, str]:
        """Pin all registered tools.

        Returns:
            Dictionary of tool names to their pinned hashes.
        """
        return {name: self.pin_tool(name) for name in self._tools}

    def verify_integrity(self, name: str) -> bool:
        """Verify a tool's schema hasn't been tampered with.

        Args:
            name: Tool name.

        Returns:
            True if hash matches pinned value (or no pin exists).
        """
        if name not in self._pinned_hashes:
            return True  # No pin = no enforcement
        if name not in self._tools:
            return False
        return self._tools[name].schema_hash == self._pinned_hashes[name]

    def get_tool(self, name: str) -> ToolSpec | None:
        """Get a registered tool spec."""
        return self._tools.get(name)

    def list_tools(self) -> list[str]:
        """List all registered tool names."""
        return list(self._tools.keys())

    def list_by_domain(self, domain: TrustDomain) -> list[str]:
        """List tools in a specific trust domain."""
        return [n for n, t in self._tools.items() if t.domain == domain]


class CapabilityPolicy:
    """Capability-based access control policy.

    Defines what callers are allowed to do in which trust domains.
    """

    def __init__(self) -> None:
        self._grants: dict[str, set[TrustDomain]] = {}
        self._deny_all: set[str] = set()

    def grant(self, caller: str, domain: TrustDomain) -> None:
        """Grant a caller access to a trust domain.

        Args:
            caller: Caller identifier.
            domain: Trust domain to grant.
        """
        if caller not in self._grants:
            self._grants[caller] = set()
        self._grants[caller].add(domain)

    def grant_all(self, caller: str) -> None:
        """Grant a caller access to all trust domains.

        Args:
            caller: Caller identifier.
        """
        self._grants[caller] = set(TrustDomain)

    def revoke(self, caller: str, domain: TrustDomain) -> None:
        """Revoke a caller's access to a trust domain.

        Args:
            caller: Caller identifier.
            domain: Trust domain to revoke.
        """
        if caller in self._grants:
            self._grants[caller].discard(domain)

    def deny_all(self, caller: str) -> None:
        """Deny all access for a caller.

        Args:
            caller: Caller identifier.
        """
        self._deny_all.add(caller)

    def is_allowed(self, caller: str, domain: TrustDomain) -> bool:
        """Check if caller has access to the domain.

        Args:
            caller: Caller identifier.
            domain: Trust domain to check.

        Returns:
            True if access is granted.
        """
        if caller in self._deny_all:
            return False
        if caller not in self._grants:
            return False
        return domain in self._grants[caller]


class McpGuard:
    """MCP zero-trust security guard.

    Combines tool registry, capability policy, rate limiting,
    and audit trail into a single authorization layer.
    """

    def __init__(
        self,
        registry: ToolRegistry,
        policy: CapabilityPolicy | None = None,
        max_audit_entries: int = 10000,
    ) -> None:
        self._registry = registry
        self._policy = policy if policy is not None else CapabilityPolicy()
        self._audit: list[AuditEntry] = []
        self._max_audit = max_audit_entries
        self._call_counts: dict[str, list[float]] = {}

    @property
    def registry(self) -> ToolRegistry:
        """Access the tool registry."""
        return self._registry

    @property
    def policy(self) -> CapabilityPolicy:
        """Access the capability policy."""
        return self._policy

    def authorize(
        self,
        tool_name: str,
        args: dict | None = None,
        caller: str = "anonymous",
    ) -> AuthResult:
        """Authorize a tool invocation.

        Performs 4-layer checks:
        1. Tool existence and registration
        2. Schema integrity (hash pin verification)
        3. Capability policy (caller + domain)
        4. Rate limiting

        Args:
            tool_name: Name of the tool to invoke.
            args: Tool arguments.
            caller: Caller identifier.

        Returns:
            AuthResult with decision and reasoning.
        """
        args = args or {}
        now = time.monotonic()

        # Layer 1: Tool registration
        spec = self._registry.get_tool(tool_name)
        if spec is None:
            result = AuthResult(
                tool_name=tool_name,
                decision=AuthDecision.DENIED,
                reason=f"Tool not registered: {tool_name}",
                args=args,
                caller=caller,
                timestamp=now,
            )
            self._record_audit(result, None)
            return result

        # Layer 2: Integrity verification
        if not self._registry.verify_integrity(tool_name):
            result = AuthResult(
                tool_name=tool_name,
                decision=AuthDecision.DENIED,
                reason=f"Schema integrity violation: {tool_name}",
                args=args,
                caller=caller,
                timestamp=now,
            )
            self._record_audit(result, spec.domain)
            return result

        # Layer 3: Capability check
        if not self._policy.is_allowed(caller, spec.domain):
            result = AuthResult(
                tool_name=tool_name,
                decision=AuthDecision.DENIED,
                reason=(
                    f"Caller '{caller}' lacks " f"'{spec.domain.value}' capability"
                ),
                args=args,
                caller=caller,
                timestamp=now,
            )
            self._record_audit(result, spec.domain)
            return result

        # Layer 4: Rate limiting
        if not self._check_rate_limit(tool_name, spec.max_calls_per_minute, now):
            result = AuthResult(
                tool_name=tool_name,
                decision=AuthDecision.DENIED,
                reason=f"Rate limit exceeded: {tool_name}",
                args=args,
                caller=caller,
                timestamp=now,
            )
            self._record_audit(result, spec.domain)
            return result

        # Check if tool requires explicit approval
        if spec.requires_approval:
            result = AuthResult(
                tool_name=tool_name,
                decision=AuthDecision.REQUIRES_APPROVAL,
                reason=f"Tool requires approval: {tool_name}",
                args=args,
                caller=caller,
                timestamp=now,
            )
            self._record_audit(result, spec.domain)
            return result

        # All checks passed
        result = AuthResult(
            tool_name=tool_name,
            decision=AuthDecision.ALLOWED,
            reason="Authorized",
            args=args,
            caller=caller,
            timestamp=now,
        )
        self._record_audit(result, spec.domain)
        return result

    def _check_rate_limit(
        self,
        tool_name: str,
        max_per_minute: int,
        now: float,
    ) -> bool:
        """Check and update rate limit for a tool.

        Args:
            tool_name: Tool name.
            max_per_minute: Maximum calls per minute.
            now: Current monotonic time.

        Returns:
            True if within rate limit.
        """
        if tool_name not in self._call_counts:
            self._call_counts[tool_name] = []

        # Prune old entries (older than 60 seconds)
        window_start = now - 60.0
        self._call_counts[tool_name] = [
            t for t in self._call_counts[tool_name] if t > window_start
        ]

        if len(self._call_counts[tool_name]) >= max_per_minute:
            return False

        self._call_counts[tool_name].append(now)
        return True

    def _record_audit(
        self,
        result: AuthResult,
        domain: TrustDomain | None,
    ) -> None:
        """Record an audit entry.

        Args:
            result: Authorization result.
            domain: Tool's trust domain (if known).
        """
        entry = AuditEntry(
            tool_name=result.tool_name,
            caller=result.caller,
            decision=result.decision,
            reason=result.reason,
            args_hash=compute_args_hash(result.args),
            timestamp=result.timestamp,
            domain=domain,
        )
        self._audit.append(entry)

        # Trim if over limit
        if len(self._audit) > self._max_audit:
            self._audit = self._audit[-self._max_audit :]

        if result.decision == AuthDecision.DENIED:
            logger.warning(
                "MCP DENIED: tool=%s caller=%s reason=%s",
                result.tool_name,
                result.caller,
                result.reason,
            )
        else:
            logger.debug(
                "MCP %s: tool=%s caller=%s",
                result.decision.value,
                result.tool_name,
                result.caller,
            )

    def audit_log(self) -> list[AuditEntry]:
        """Return the full audit trail."""
        return list(self._audit)

    def audit_summary(self) -> dict[str, int]:
        """Summarize audit trail by decision type."""
        summary: dict[str, int] = {
            "total": len(self._audit),
            "allowed": 0,
            "denied": 0,
            "requires_approval": 0,
        }
        for entry in self._audit:
            summary[entry.decision.value] = summary.get(entry.decision.value, 0) + 1
        return summary

    def reset_audit(self) -> None:
        """Clear the audit trail."""
        self._audit.clear()

    def reset_rate_limits(self) -> None:
        """Clear all rate limit counters."""
        self._call_counts.clear()
