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

import json
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from research_pipeline.mcp_server.integrity import (
    compute_tool_hashes,
    verify_tool_integrity,
)
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
# Optional committed reference of tool-description hashes; when set, the guard
# verifies live descriptions against it at startup (tool-poisoning defense, #114).
_TOOL_HASHES_ENV = "RESEARCH_PIPELINE_TOOL_HASHES"


def _domain_for(tool: Any) -> TrustDomain:
    """Classify a tool's trust domain from its MCP annotations."""
    ann = getattr(tool, "annotations", None)
    if ann is not None and getattr(ann, "readOnlyHint", False):
        return TrustDomain.READ
    if ann is not None and getattr(ann, "openWorldHint", False):
        return TrustDomain.EXECUTE
    return TrustDomain.WRITE


def build_guard(mcp: FastMCP) -> McpGuard:
    """Build an :class:`McpGuard` registering every tool on *mcp*.

    Activates two integrity controls that were previously inert (#114): each
    tool's schema hash is **pinned** at registration so the per-call
    schema-integrity layer actually enforces (an unpinned tool was a no-op), and
    a SHA-256 of every tool **description** is computed and attached to the guard
    (``description_hashes``) so the tool-poisoning defense is live, not dead code.
    """
    registry = ToolRegistry()
    tool_defs: list[dict[str, str]] = []
    for name, tool in mcp._tool_manager._tools.items():
        description = getattr(tool, "description", "") or ""
        registry.register(
            name,
            schema=getattr(tool, "parameters", None) or {},
            domain=_domain_for(tool),
            description=description,
            max_calls_per_minute=_MAX_CALLS_PER_MINUTE,
        )
        # Pin the freshly-registered schema so verify_integrity has a baseline.
        registry.pin_tool(name)
        tool_defs.append({"name": name, "description": description})
    policy = CapabilityPolicy()
    policy.grant_all(_DEFAULT_CALLER)
    guard = McpGuard(registry, policy)
    guard.description_hashes = compute_tool_hashes(tool_defs)
    return guard


def _verify_description_hashes(guard: McpGuard) -> None:
    """Verify live tool descriptions against a committed reference, if configured.

    When ``RESEARCH_PIPELINE_TOOL_HASHES`` points at a JSON ``{name: hash}``
    reference, warn on any tool whose description no longer matches (a
    tool-poisoning signal). Non-fatal: a legitimate description update should not
    take the server down, so this logs rather than raises.
    """
    ref_path = os.environ.get(_TOOL_HASHES_ENV)
    if not ref_path:
        return
    path = Path(ref_path).expanduser()
    if not path.is_file():
        logger.warning(
            "Tool-hash reference %s not found; skipping integrity check", path
        )
        return
    try:
        reference = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.warning("Could not read tool-hash reference %s: %s", path, exc)
        return
    tampered = verify_tool_integrity(guard.description_hashes, reference)
    if tampered:
        logger.warning(
            "Tool-description integrity check FAILED for %d tool(s): %s",
            len(tampered),
            ", ".join(sorted(tampered)),
        )
    else:
        logger.info("Tool-description integrity verified for %d tools", len(reference))


def install_guard(mcp: FastMCP) -> McpGuard:
    """Wrap *mcp*'s tool dispatch so every call is authorized by the guard."""
    guard = build_guard(mcp)
    _verify_description_hashes(guard)
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
