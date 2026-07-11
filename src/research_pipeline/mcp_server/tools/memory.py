from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from research_pipeline.mcp_server.schemas import (
    CbrLookupInput,
    CbrRetainInput,
    MemoryEpisodesInput,
    MemorySearchInput,
    MemoryStatsInput,
    ToolResult,
)
from research_pipeline.mcp_server.tools._common import _raise_tool_error

if TYPE_CHECKING:
    from mcp.server.fastmcp import Context


def cbr_lookup_tool(
    params: CbrLookupInput,
    ctx: Context | None = None,
) -> ToolResult:
    """Look up similar past cases and recommend a research strategy.

    Uses Case-Based Reasoning (arXiv 2506.18096) to retrieve and adapt
    strategies from successful past runs.
    """
    try:
        from research_pipeline.memory.cbr import cbr_lookup

        ws = Path(params.workspace)
        rec = cbr_lookup(
            params.topic,
            ws,
            max_results=params.max_results,
            min_quality=params.min_quality,
        )

        return ToolResult(
            success=True,
            message=(
                f"CBR recommendation for '{params.topic}': "
                f"confidence={rec.confidence:.2f}, "
                f"sources=[{', '.join(rec.recommended_sources)}], "
                f"profile={rec.recommended_profile}, "
                f"based on {len(rec.basis_cases)} case(s)"
            ),
            artifacts=rec.to_dict(),
        )
    except Exception as exc:
        _raise_tool_error("CBR lookup", exc)


def cbr_retain_tool(
    params: CbrRetainInput,
    ctx: Context | None = None,
) -> ToolResult:
    """Store a completed pipeline run as a CBR case.

    Extracts strategy information from run artifacts and stores it for
    future retrieval and adaptation.
    """
    try:
        from research_pipeline.memory.cbr import cbr_retain

        ws = Path(params.workspace)
        case = cbr_retain(
            params.run_id,
            params.topic,
            ws,
            outcome=params.outcome,
            strategy_notes=params.strategy_notes,
        )

        return ToolResult(
            success=True,
            message=(
                f"Stored CBR case '{case.case_id}': "
                f"quality={case.synthesis_quality:.3f}, "
                f"outcome={case.outcome}, "
                f"papers={case.paper_count}, shortlisted={case.shortlist_count}"
            ),
            artifacts=case.to_dict(),
        )
    except Exception as exc:
        _raise_tool_error("CBR retain", exc)


def memory_stats_tool(
    params: MemoryStatsInput, ctx: Context | None = None
) -> ToolResult:
    """Show memory tier statistics."""
    try:
        from research_pipeline.memory.manager import MemoryManager

        episodic_path = Path(params.episodic_db) if params.episodic_db else None
        kg_path = Path(params.kg_db) if params.kg_db else None

        manager = MemoryManager(episodic_path=episodic_path, kg_path=kg_path)
        try:
            stats = manager.summary()
        finally:
            manager.close()

        return ToolResult(
            success=True,
            message="Memory tier statistics retrieved.",
            artifacts=stats,
        )
    except Exception as exc:
        _raise_tool_error("memory_stats", exc)


def memory_episodes_tool(
    params: MemoryEpisodesInput, ctx: Context | None = None
) -> ToolResult:
    """List recent episodic memories (past runs)."""
    try:
        from research_pipeline.memory.episodic import EpisodicMemory

        episodic_path = Path(params.episodic_db) if params.episodic_db else None
        mem = EpisodicMemory(db_path=episodic_path)
        try:
            episodes = mem.recent_episodes(limit=params.limit)
        finally:
            mem.close()

        episode_list = []
        for ep in episodes:
            episode_list.append(
                {
                    "run_id": ep.run_id,
                    "topic": ep.topic,
                    "paper_count": ep.paper_count,
                    "shortlist_count": ep.shortlist_count,
                    "stages_completed": list(ep.stages_completed),
                    "started_at": str(ep.started_at),
                }
            )

        return ToolResult(
            success=True,
            message=f"Found {len(episode_list)} episode(s).",
            artifacts={"episodes": episode_list},
        )
    except Exception as exc:
        _raise_tool_error("memory_episodes", exc)


def memory_search_tool(
    params: MemorySearchInput, ctx: Context | None = None
) -> ToolResult:
    """Search episodic memory for past runs on a topic."""
    try:
        from research_pipeline.memory.episodic import EpisodicMemory

        episodic_path = Path(params.episodic_db) if params.episodic_db else None
        mem = EpisodicMemory(db_path=episodic_path)
        try:
            episodes = mem.search_by_topic(params.topic, limit=params.limit)
        finally:
            mem.close()

        episode_list = []
        for ep in episodes:
            episode_list.append(
                {
                    "run_id": ep.run_id,
                    "topic": ep.topic,
                    "paper_count": ep.paper_count,
                    "shortlist_count": ep.shortlist_count,
                    "stages_completed": list(ep.stages_completed),
                    "started_at": str(ep.started_at),
                }
            )

        return ToolResult(
            success=True,
            message=(
                f"Found {len(episode_list)} past run(s) matching {params.topic!r}."
            ),
            artifacts={"episodes": episode_list, "query": params.topic},
        )
    except Exception as exc:
        _raise_tool_error("memory_search", exc)
