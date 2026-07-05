"""Tests for capability-domain toolsets (#46)."""

from __future__ import annotations

from types import SimpleNamespace

from research_pipeline.mcp_server import toolsets
from research_pipeline.mcp_server.server import mcp


def _fake_mcp(names: list[str]) -> SimpleNamespace:
    return SimpleNamespace(
        _tool_manager=SimpleNamespace(_tools={n: object() for n in names})
    )


def test_map_has_no_duplicates_and_expected_size() -> None:
    seen = [n for names in toolsets.TOOLSETS.values() for n in names]
    assert len(seen) == len(set(seen)), "a tool appears in multiple domains"
    assert len(set(seen)) == 64


def test_map_covers_every_registered_tool() -> None:
    """A new tool cannot be added without being classified into a domain."""
    mapped = set().union(*toolsets.TOOLSETS.values())
    registered = set(mcp._tool_manager._tools)
    assert registered <= mapped, f"unmapped tools: {registered - mapped}"


def test_default_selection_keeps_all() -> None:
    assert toolsets._requested_domains(None) is None
    assert toolsets._requested_domains("") is None


def test_selects_valid_subset() -> None:
    assert toolsets._requested_domains("pipeline,inspection") == {
        "pipeline",
        "inspection",
    }


def test_unknown_domain_is_ignored() -> None:
    assert toolsets._requested_domains("pipeline,bogus") == {"pipeline"}


def test_all_invalid_falls_back_to_all() -> None:
    assert toolsets._requested_domains("bogus,nope") is None


def test_apply_prunes_to_selected_domains() -> None:
    fake = _fake_mcp(["tool_plan_topic", "tool_kg_query", "brief_run"])
    active = toolsets.apply_toolsets(fake, raw="pipeline")
    assert active == {"pipeline"}
    assert set(fake._tool_manager._tools) == {"tool_plan_topic"}


def test_apply_default_prunes_nothing() -> None:
    names = ["tool_plan_topic", "tool_kg_query", "brief_run"]
    fake = _fake_mcp(names)
    toolsets.apply_toolsets(fake, raw=None)
    assert set(fake._tool_manager._tools) == set(names)
