"""MCP adapter for the bounded public-source probe."""

import logging
from functools import partial
from pathlib import Path

import anyio
from mcp.server.fastmcp import Context

from research_pipeline.config.loader import load_config
from research_pipeline.mcp_server.schemas import ProbeSourcesInput, ToolResult
from research_pipeline.sources.source_probe import probe_sources

logger = logging.getLogger(__name__)


async def probe_sources_tool(
    params: ProbeSourcesInput, ctx: Context | None = None
) -> ToolResult:
    """Run blocking network checks off the MCP event loop with awaited progress."""

    def progress(done: int, total: int, message: str) -> None:
        if ctx is not None:
            try:
                anyio.from_thread.run(ctx.report_progress, done, total, message)
            except Exception as exc:
                logger.warning(
                    "Probe progress notification failed (%s)", type(exc).__name__
                )

    try:
        config = load_config(Path(params.config_path) if params.config_path else None)
        report = await anyio.to_thread.run_sync(
            partial(probe_sources, params.source, config=config, progress=progress)
        )
    except Exception as exc:
        logger.error("Source probe failed (%s)", type(exc).__name__)
        return ToolResult(
            success=False, message=f"Source probe failed ({type(exc).__name__})."
        )
    return ToolResult(
        success=True,
        message=(
            "Probe completed. Inspect each status; an HTTP response does not guarantee "
            "usable search and 429 does not establish an IP-specific ban."
        ),
        artifacts={"probe": report.model_dump(mode="json")},
    )
