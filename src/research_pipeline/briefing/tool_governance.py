"""MCP tool-scope governance for daily AI intelligence (Phase G).

Defines capability policies for each ``brief_*`` MCP tool: whether it touches
the network, whether it writes outside the workspace, and which sources are
permitted. This module is the single source of truth queried by tests, by the
skill, and by future MCP server registration to attach annotations.

The module is deterministic and pure — it has no side effects.
"""

from __future__ import annotations

from typing import Final, Literal

from pydantic import BaseModel, ConfigDict

ToolKind = Literal["network", "local"]
ToolEffect = Literal["read", "write"]


class ToolPolicy(BaseModel):
    """Capability policy for a single ``brief_*`` MCP tool."""

    model_config = ConfigDict(frozen=True)

    name: str
    kind: ToolKind
    effects: tuple[ToolEffect, ...]
    deterministic: bool
    source_allowlisted: bool
    write_paths: tuple[str, ...] = ()
    notes: str = ""


_POLICIES: Final[dict[str, ToolPolicy]] = {
    # Polling is the only tool allowed to touch the network. It is strictly
    # source-allowlisted via the registry config supplied by the caller.
    "brief_poll_sources_tool": ToolPolicy(
        name="brief_poll_sources_tool",
        kind="network",
        effects=("read", "write"),
        deterministic=False,
        source_allowlisted=True,
        write_paths=("workspace/briefings/<date>/polled.jsonl",),
        notes="Network polling restricted to registry sources.",
    ),
    "brief_run_tool": ToolPolicy(
        name="brief_run_tool",
        kind="network",
        effects=("read", "write"),
        deterministic=False,
        source_allowlisted=True,
        write_paths=("workspace/briefings/<date>/",),
        notes="Wraps poll + rank + generate + validate.",
    ),
    # Ranking and validation are local and deterministic.
    "brief_rank_events_tool": ToolPolicy(
        name="brief_rank_events_tool",
        kind="local",
        effects=("read", "write"),
        deterministic=True,
        source_allowlisted=False,
        write_paths=("workspace/briefings/<date>/ranked/ranked_clusters.jsonl",),
    ),
    "brief_validate_report_tool": ToolPolicy(
        name="brief_validate_report_tool",
        kind="local",
        effects=("read", "write"),
        deterministic=True,
        source_allowlisted=False,
        write_paths=("workspace/briefings/<date>/validation/validation.json",),
    ),
    "brief_generate_daily_tool": ToolPolicy(
        name="brief_generate_daily_tool",
        kind="local",
        effects=("read", "write"),
        deterministic=True,
        source_allowlisted=False,
        write_paths=("workspace/briefings/<date>/reports/daily.md",),
    ),
    "brief_generate_dossier_tool": ToolPolicy(
        name="brief_generate_dossier_tool",
        kind="local",
        effects=("read", "write"),
        deterministic=True,
        source_allowlisted=False,
        write_paths=("workspace/briefings/<date>/dossiers/",),
    ),
    "brief_weekly_synthesis_tool": ToolPolicy(
        name="brief_weekly_synthesis_tool",
        kind="local",
        effects=("read", "write"),
        deterministic=True,
        source_allowlisted=False,
        write_paths=("workspace/briefings/weekly/<week>.md",),
    ),
    # Obsidian export and feedback are write-tools; they only write to caller-
    # supplied paths (vault / feedback store).
    "brief_export_obsidian_tool": ToolPolicy(
        name="brief_export_obsidian_tool",
        kind="local",
        effects=("read", "write"),
        deterministic=True,
        source_allowlisted=False,
        write_paths=("<vault_path>/",),
        notes="Writes only to configured Obsidian vault path.",
    ),
    "brief_record_feedback_tool": ToolPolicy(
        name="brief_record_feedback_tool",
        kind="local",
        effects=("write",),
        deterministic=True,
        source_allowlisted=False,
        write_paths=("workspace/briefings/feedback.jsonl",),
        notes="Writes only to local feedback store.",
    ),
}


BRIEF_TOOL_NAMES: Final[tuple[str, ...]] = tuple(sorted(_POLICIES))


def policy_for(tool_name: str) -> ToolPolicy:
    """Return the governance policy for a brief_* MCP tool.

    Raises:
        KeyError: If ``tool_name`` has no registered policy.
    """
    if tool_name not in _POLICIES:
        raise KeyError(f"No governance policy for tool: {tool_name}")
    return _POLICIES[tool_name]


def all_policies() -> dict[str, ToolPolicy]:
    """Return a copy of all registered tool policies."""
    return dict(_POLICIES)


def is_unsupported_source(url_or_id: str, allowlist: tuple[str, ...]) -> bool:
    """Return True if ``url_or_id`` is not covered by the allowlist.

    The allowlist is a tuple of substrings that must appear in the URL/ID for
    it to be accepted. This helper lets callers refuse expansion to sources
    that have not been added to the configured registry.
    """
    return not any(token in url_or_id for token in allowlist)
