"""Tests for briefing MCP tool governance (G07)."""

from __future__ import annotations

import pytest

from research_pipeline.briefing.tool_governance import (
    BRIEF_TOOL_NAMES,
    ToolPolicy,
    all_policies,
    is_unsupported_source,
    policy_for,
)
from research_pipeline.mcp_server import tools as mcp_tools

EXPECTED_BRIEF_TOOLS = {
    "brief_poll_sources_tool",
    "brief_rank_events_tool",
    "brief_generate_daily_tool",
    "brief_validate_report_tool",
    "brief_run_tool",
    "brief_export_obsidian_tool",
    "brief_record_feedback_tool",
    "brief_generate_dossier_tool",
    "brief_weekly_synthesis_tool",
}


def test_governance_covers_every_brief_tool() -> None:
    assert set(BRIEF_TOOL_NAMES) == EXPECTED_BRIEF_TOOLS
    for name in EXPECTED_BRIEF_TOOLS:
        assert hasattr(mcp_tools, name), f"MCP tool missing: {name}"


def test_policy_for_unknown_tool_raises() -> None:
    with pytest.raises(KeyError):
        policy_for("brief_nonexistent_tool")


def test_network_tools_are_source_allowlisted() -> None:
    for name, pol in all_policies().items():
        if pol.kind == "network":
            assert pol.source_allowlisted, f"{name}: network tool must be allowlisted"


def test_local_tools_are_deterministic() -> None:
    for name, pol in all_policies().items():
        if pol.kind == "local":
            assert pol.deterministic, f"{name}: local tool must be deterministic"


def test_ranking_and_validation_are_local_deterministic() -> None:
    for name in (
        "brief_rank_events_tool",
        "brief_validate_report_tool",
        "brief_generate_daily_tool",
    ):
        pol = policy_for(name)
        assert pol.kind == "local"
        assert pol.deterministic is True
        assert pol.source_allowlisted is False


def test_export_and_feedback_are_write_tools() -> None:
    export = policy_for("brief_export_obsidian_tool")
    feedback = policy_for("brief_record_feedback_tool")
    assert "write" in export.effects
    assert "write" in feedback.effects
    # Writes only to configured paths (vault, local feedback store).
    assert "vault" in export.notes.lower() or "<vault_path>" in export.write_paths[0]
    assert "feedback" in feedback.notes.lower() or "feedback" in feedback.write_paths[0]


def test_polling_is_only_network_tool_for_data_ingest() -> None:
    network = {n for n, p in all_policies().items() if p.kind == "network"}
    # brief_poll_sources_tool and brief_run_tool (which wraps polling).
    assert "brief_poll_sources_tool" in network


def test_tool_policy_is_frozen() -> None:
    pol = policy_for("brief_rank_events_tool")
    with pytest.raises(Exception):  # noqa: B017
        pol.kind = "network"  # type: ignore[misc]


def test_tool_policy_model_shape() -> None:
    for name, pol in all_policies().items():
        assert isinstance(pol, ToolPolicy)
        assert pol.name == name
        assert pol.kind in ("network", "local")
        for eff in pol.effects:
            assert eff in ("read", "write")
        assert isinstance(pol.deterministic, bool)
        assert isinstance(pol.source_allowlisted, bool)


def test_unsupported_source_helper_refuses_unknown() -> None:
    allowlist = ("github.com/", "huggingface.co/")
    assert is_unsupported_source("https://random.example.com/x", allowlist) is True
    assert is_unsupported_source("https://github.com/owner/repo", allowlist) is False
    assert is_unsupported_source("", allowlist) is True


def test_all_policies_returns_copy() -> None:
    a = all_policies()
    a.pop("brief_run_tool", None)
    b = all_policies()
    assert "brief_run_tool" in b
