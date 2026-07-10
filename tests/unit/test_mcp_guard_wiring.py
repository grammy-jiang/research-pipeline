"""Tests for wiring the McpGuard into MCP tool dispatch (#45)."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from research_pipeline.mcp_server import server
from research_pipeline.mcp_server.guard_wiring import (
    _TOOL_HASHES_ENV,
    _verify_description_hashes,
    build_guard,
)
from research_pipeline.mcp_server.integrity import compute_tool_hashes
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


# --- issue #114: activate the previously-inert integrity controls ---


def test_build_guard_populates_description_hashes() -> None:
    guard = build_guard(mcp)
    assert guard.description_hashes  # no longer dead code
    assert set(guard.description_hashes) == set(mcp._tool_manager._tools)
    expected = compute_tool_hashes(
        [
            {"name": n, "description": getattr(t, "description", "") or ""}
            for n, t in mcp._tool_manager._tools.items()
        ]
    )
    assert guard.description_hashes == expected


def test_build_guard_arms_schema_integrity() -> None:
    guard = build_guard(mcp)
    name = "tool_list_backends"
    # Pinned at registration → integrity is now enforced (was a no-op).
    assert guard.registry.verify_integrity(name) is True
    # Re-register the same tool with a different schema: the new schema_hash no
    # longer matches the pinned baseline, so the (now-armed) check fails.
    guard.registry.register(name, schema={"changed": True}, domain=TrustDomain.READ)
    assert guard.registry.verify_integrity(name) is False


def test_verify_description_hashes_detects_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    guard = build_guard(mcp)
    # A reference where one tool's description hash is wrong → tamper signal.
    reference = dict(guard.description_hashes)
    reference["tool_list_backends"] = "0" * 64
    ref_file = tmp_path / "tool_hashes.json"
    ref_file.write_text(json.dumps(reference))
    monkeypatch.setenv(_TOOL_HASHES_ENV, str(ref_file))
    with caplog.at_level(logging.WARNING):
        _verify_description_hashes(guard)
    assert any("integrity check FAILED" in r.message for r in caplog.records)


def test_verify_description_hashes_noop_without_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(_TOOL_HASHES_ENV, raising=False)
    guard = build_guard(mcp)
    _verify_description_hashes(guard)  # must not raise
