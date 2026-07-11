"""Packaged MCP server entry point (#109).

Runs the FastMCP server over stdio. Lives in ``mcp_server/`` (not ``cli/``) so the
server package no longer depends on the CLI layer: both the CLI ``mcp serve``
command and the ``python -m research_pipeline.mcp_server`` launcher call this.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def run_mcp_serve() -> None:
    """Run the packaged MCP server over stdio."""
    from research_pipeline.mcp_server.server import mcp

    logger.info("Starting research-pipeline MCP server")
    mcp.run()
