"""In-memory MCP protocol round-trip harness (#47).

Drives the packaged ``mcp`` server through a real client <-> server session so
``tools/call``, ``resources/read``, and ``prompts/get`` are exercised end to
end. The isError / structuredContent / outputSchema guarantees from #36, #38,
#40, and #42 are asserted at the protocol layer, not by introspecting server
internals.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine
from typing import Any

import pytest
from mcp.shared.exceptions import McpError
from mcp.shared.memory import create_connected_server_and_client_session
from pydantic import AnyUrl

from research_pipeline.mcp_server.server import mcp


def _run[T](coro_fn: Callable[[], Coroutine[Any, Any, T]]) -> T:
    return asyncio.run(coro_fn())


def test_list_tools_expose_output_schema() -> None:
    async def _t() -> None:
        async with create_connected_server_and_client_session(mcp) as client:
            resp = await client.list_tools()
            names = {t.name for t in resp.tools}
            assert "tool_plan_topic" in names
            tool = next(t for t in resp.tools if t.name == "tool_plan_topic")
            assert tool.outputSchema is not None
            assert {"success", "message", "artifacts"} <= set(
                tool.outputSchema.get("properties", {})
            )

    _run(_t)


def test_successful_tool_call_has_structured_content() -> None:
    async def _t() -> None:
        async with create_connected_server_and_client_session(mcp) as client:
            result = await client.call_tool("tool_list_backends", {})
            assert result.isError is False
            assert result.structuredContent is not None
            assert result.structuredContent["success"] is True

    _run(_t)


def test_failing_tool_call_sets_iserror() -> None:
    async def _t() -> None:
        async with create_connected_server_and_client_session(mcp) as client:
            # A run_id with traversal is rejected by the CommonParams validator
            # (#40); the failure must surface as isError (#38), not success.
            result = await client.call_tool(
                "tool_plan_topic", {"topic": "x", "run_id": "../../etc"}
            )
            assert result.isError is True

    _run(_t)


def test_resource_read_missing_raises() -> None:
    async def _t() -> None:
        async with create_connected_server_and_client_session(mcp) as client:
            # A missing run must be a JSON-RPC error, not a success blob (#42).
            with pytest.raises(McpError):
                await client.read_resource(AnyUrl("runs://nonexistent-xyz/manifest"))

    _run(_t)


def test_prompt_get_uses_valid_roles() -> None:
    async def _t() -> None:
        async with create_connected_server_and_client_session(mcp) as client:
            # research_topic previously opened with role:"system" and errored
            # every prompts/get (#36).
            resp = await client.get_prompt("research_topic", {"topic": "x"})
            assert resp.messages
            for msg in resp.messages:
                assert msg.role in ("user", "assistant")

    _run(_t)
