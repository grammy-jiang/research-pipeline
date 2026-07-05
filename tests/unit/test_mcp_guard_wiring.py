"""Tests for wiring the McpGuard into MCP tool dispatch (#45)."""

from __future__ import annotations

from research_pipeline.mcp_server import server
from research_pipeline.mcp_server.guard_wiring import build_guard
from research_pipeline.mcp_server.server import mcp
from research_pipeline.security.mcp_guard import AuthDecision, TrustDomain


def test_build_guard_registers_all_tools() -> None:
    guard = build_guard(mcp)
    assert len(guard.registry._tools) == len(mcp._tool_manager._tools)


def test_domains_classified_from_annotations() -> None:
    guard = build_guard(mcp)
    # readOnlyHint -> READ
    assert guard.registry.get_tool("tool_get_run_manifest").domain == TrustDomain.READ
    # openWorldHint (network) -> EXECUTE
    assert guard.registry.get_tool("tool_search").domain == TrustDomain.EXECUTE
    # writer (not read-only, not open-world) -> WRITE/EXECUTE, never READ
    assert guard.registry.get_tool("tool_report").domain != TrustDomain.READ


def test_authorize_allows_registered_tool() -> None:
    guard = build_guard(mcp)
    res = guard.authorize("tool_list_backends", {})
    assert res.decision == AuthDecision.ALLOWED


def test_authorize_denies_unregistered_tool() -> None:
    guard = build_guard(mcp)
    res = guard.authorize("tool_does_not_exist", {})
    assert res.decision == AuthDecision.DENIED


def test_guard_installed_on_dispatch() -> None:
    # server import wired a guard and replaced tool_manager.call_tool.
    assert server._guard is not None
    assert mcp._tool_manager.call_tool.__name__ == "guarded_call_tool"


def test_authorize_records_audit() -> None:
    guard = build_guard(mcp)
    guard.authorize("tool_list_backends", {})
    guard.authorize("tool_does_not_exist", {})
    # both the allow and the deny are audited
    assert len(guard._audit) == 2
