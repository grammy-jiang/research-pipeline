"""MCP server CLI helpers."""

from __future__ import annotations

import json
import logging

logger = logging.getLogger(__name__)


def mcp_server_config() -> dict[str, dict[str, dict[str, object]]]:
    """Return a generic MCP client config for the packaged server.

    Uses ``"type": "stdio"`` as recommended by GitHub Copilot CLI docs for
    cross-client compatibility (VS Code, Copilot cloud agent, Copilot CLI).
    """
    return {
        "mcpServers": {
            "research-pipeline": {
                "type": "stdio",
                "command": "research-pipeline",
                "args": ["mcp", "serve"],
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
