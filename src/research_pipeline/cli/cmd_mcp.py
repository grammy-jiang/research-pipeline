"""MCP server CLI helpers."""

from __future__ import annotations

import json
import logging

logger = logging.getLogger(__name__)


def mcp_server_config() -> dict[str, dict[str, dict[str, object]]]:
    """Return a generic MCP client config for the packaged server."""
    return {
        "mcpServers": {
            "research-pipeline": {
                "command": "research-pipeline",
                "args": ["mcp", "serve"],
                "description": "Run the packaged research-pipeline MCP server.",
            }
        }
    }


def run_mcp_serve() -> None:
    """Run the packaged MCP server over stdio."""
    from research_pipeline.mcp_server.server import mcp

    logger.info("Starting research-pipeline MCP server")
    mcp.run()


def render_mcp_config() -> str:
    """Render a reusable MCP client config snippet."""
    return json.dumps(mcp_server_config(), indent=2)
