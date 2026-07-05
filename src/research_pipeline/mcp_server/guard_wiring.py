"""Wire the zero-trust ``McpGuard`` into FastMCP tool dispatch (#45).

``security/mcp_guard.py`` (4-layer authorize/audit/rate-limit),
``mcp_server/integrity.py`` (hash pinning), and ``infra/tool_pinning.py`` were
imported only by the pipeline orchestrator, never by the ``@mcp.tool()``
dispatch path — the docstrings implied protection that never ran.

This module builds a :class:`ToolRegistry` from the live registered tools and
wraps the tool manager so every ``tools/call`` first passes through
``McpGuard.authorize()`` (tool registration + schema-hash integrity +
capability policy + rate limiting + audit). Calls the guard denies raise, so
they surface as ``isError`` to the client.

The local stdio server trusts its single caller, so the default policy grants
every domain: the value wired here is *active* integrity + rate-limiting +
an audit trail (no longer false assurance), not caller lockout. Domain
classification is retained so a future policy can tighten write/execute tools.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from research_pipeline.security.mcp_guard import (
    AuthDecision,
    CapabilityPolicy,
    McpGuard,
    ToolRegistry,
    TrustDomain,
)

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

logger = logging.getLogger(__name__)

# Generous per-tool ceiling: high enough never to throttle legitimate
# interactive/workflow use, low enough to cap a runaway loop.
_MAX_CALLS_PER_MINUTE = 300
_DEFAULT_CALLER = "anonymous"


def _domain_for(tool: Any) -> TrustDomain:
    """Classify a tool's trust domain from its MCP annotations."""
    ann = getattr(tool, "annotations", None)
    if ann is not None and getattr(ann, "readOnlyHint", False):
        return TrustDomain.READ
    if ann is not None and getattr(ann, "openWorldHint", False):
        return TrustDomain.EXECUTE
    return TrustDomain.WRITE


def build_guard(mcp: FastMCP) -> McpGuard:
    """Build an :class:`McpGuard` registering every tool on *mcp*."""
    registry = ToolRegistry()
    for name, tool in mcp._tool_manager._tools.items():
        registry.register(
            name,
            schema=getattr(tool, "parameters", None) or {},
            domain=_domain_for(tool),
            max_calls_per_minute=_MAX_CALLS_PER_MINUTE,
        )
    policy = CapabilityPolicy()
    policy.grant_all(_DEFAULT_CALLER)
    return McpGuard(registry, policy)


def install_guard(mcp: FastMCP) -> McpGuard:
    """Wrap *mcp*'s tool dispatch so every call is authorized by the guard."""
    guard = build_guard(mcp)
    original = mcp._tool_manager.call_tool

    async def guarded_call_tool(
        name: str, arguments: dict[str, Any], **kwargs: Any
    ) -> Any:
        decision = guard.authorize(name, arguments, caller=_DEFAULT_CALLER)
        if decision.decision == AuthDecision.DENIED:
            logger.warning("MCP guard denied %s: %s", name, decision.reason)
            raise RuntimeError(f"Blocked by MCP guard: {decision.reason}")
        return await original(name, arguments, **kwargs)

    mcp._tool_manager.call_tool = guarded_call_tool  # type: ignore[method-assign]
    return guard
