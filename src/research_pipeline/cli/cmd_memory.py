"""CLI commands for memory inspection."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import typer

logger = logging.getLogger(__name__)


def run_memory_stats(
    episodic_path: Path | None = None,
    kg_path: Path | None = None,
) -> None:
    """Show memory tier statistics."""
    from research_pipeline.memory.manager import MemoryManager

    manager = MemoryManager(episodic_path=episodic_path, kg_path=kg_path)
    try:
        stats = manager.summary()
        typer.echo(json.dumps(stats, indent=2, default=str))
    finally:
        manager.close()


def run_memory_episodes(
    limit: int = 10,
    episodic_path: Path | None = None,
) -> None:
    """List recent episodic memories (past runs)."""
    from research_pipeline.memory.episodic import EpisodicMemory

    mem = EpisodicMemory(db_path=episodic_path)
    try:
        episodes = mem.recent_episodes(limit=limit)
        if not episodes:
            typer.echo("No episodes recorded yet.")
            return
        for ep in episodes:
            typer.echo(
                f"  {ep.run_id}  topic={ep.topic!r}  "
                f"papers={ep.paper_count}  shortlist={ep.shortlist_count}  "
                f"stages={','.join(ep.stages_completed)}  "
                f"started={ep.started_at}"
            )
    finally:
        mem.close()


def run_memory_search(
    topic: str,
    limit: int = 10,
    episodic_path: Path | None = None,
) -> None:
    """Search episodic memory for past runs on a topic."""
    from research_pipeline.memory.episodic import EpisodicMemory

    mem = EpisodicMemory(db_path=episodic_path)
    try:
        episodes = mem.search_by_topic(topic, limit=limit)
        if not episodes:
            typer.echo(f"No past runs found matching {topic!r}.")
            return
        typer.echo(f"Found {len(episodes)} past run(s) matching {topic!r}:")
        for ep in episodes:
            typer.echo(
                f"  {ep.run_id}  topic={ep.topic!r}  "
                f"papers={ep.paper_count}  shortlist={ep.shortlist_count}  "
                f"started={ep.started_at}"
            )
    finally:
        mem.close()
