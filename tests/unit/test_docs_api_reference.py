"""Drift guard for docs/api-reference.md (#115).

The API reference hand-transcribes ~66 CLI commands and 60+ MCP tools with no
tie to the real Typer/Pydantic surface, so a command or tool added (or removed)
without a docs update silently drifts. These tests fail when the reference no
longer covers the registered surface — the cheap, robust half of the
mkdocstrings/CI-diff fix the issue proposed (presence, not full signatures).
"""

from __future__ import annotations

from pathlib import Path

from research_pipeline.cli.app import app
from research_pipeline.mcp_server.server import mcp

_API_REFERENCE = Path(__file__).resolve().parents[2] / "docs" / "api-reference.md"


def _cli_command_names() -> list[str]:
    """Every command + sub-app name Typer exposes (name, else func name)."""
    names: list[str] = []
    for cmd in app.registered_commands:
        name = cmd.name or (
            cmd.callback.__name__.replace("_", "-") if cmd.callback else None
        )
        if name:
            names.append(name)
    for group in app.registered_groups:
        if group.name:
            names.append(group.name)
    return sorted(names)


def test_every_cli_command_is_documented() -> None:
    doc = _API_REFERENCE.read_text(encoding="utf-8")
    missing = [
        name
        for name in _cli_command_names()
        if f"`{name}`" not in doc and f"research-pipeline {name}" not in doc
    ]
    assert not missing, (
        f"CLI commands missing from docs/api-reference.md: {missing}. "
        "Document each (or correct the name) so the reference can't drift."
    )


def test_every_mcp_tool_is_documented() -> None:
    doc = _API_REFERENCE.read_text(encoding="utf-8")
    missing = [name for name in sorted(mcp._tool_manager._tools) if name not in doc]
    assert not missing, (
        f"MCP tools missing from docs/api-reference.md: {missing}. "
        "Add a row to the '12. MCP Server API' tool tables so it can't drift."
    )
