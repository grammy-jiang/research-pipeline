"""CLI handler for the 'index' command.

Manages the global paper index for incremental runs.
"""

import logging

import typer

from research_pipeline.storage.global_index import GlobalPaperIndex

logger = logging.getLogger(__name__)


def run_index(
    list_papers: bool = False,
    gc: bool = False,
    search: str | None = None,
    search_limit: int = 50,
    db_path: str | None = None,
) -> None:
    """Manage the global paper index.

    Args:
        list_papers: List indexed papers.
        gc: Garbage collect stale entries.
        search: Full-text search query (FTS5).
        search_limit: Maximum number of search results.
        db_path: Path to index database.
    """
    from pathlib import Path

    path = Path(db_path) if db_path else None
    index = GlobalPaperIndex(db_path=path)

    try:
        if gc:
            removed = index.garbage_collect()
            typer.echo(f"Garbage collected {removed} stale entries.")
            return

        if search:
            results = index.search_fulltext(search, limit=search_limit)
            if not results:
                typer.echo(f"No papers matching '{search}'.")
                return
            typer.echo(f"{'arXiv ID':<20} {'Title':<50} {'Run ID'}")
            typer.echo("-" * 100)
            for p in results:
                title = (p.get("title") or "N/A")[:48]
                typer.echo(
                    f"{p.get('arxiv_id', 'N/A'):<20} "
                    f"{title:<50} "
                    f"{p.get('run_id', 'N/A')}"
                )
            typer.echo(f"\n{len(results)} result(s) for '{search}'")
            return

        if list_papers:
            papers = index.list_papers(limit=100)
            if not papers:
                typer.echo("No papers in the global index.")
                return
            typer.echo(f"{'arXiv ID':<20} {'Stage':<12} {'Run ID':<30} {'Indexed'}")
            typer.echo("-" * 80)
            for p in papers:
                typer.echo(
                    f"{p.get('arxiv_id', 'N/A'):<20} "
                    f"{p.get('stage', 'N/A'):<12} "
                    f"{p.get('run_id', 'N/A'):<30} "
                    f"{p.get('indexed_at', 'N/A')}"
                )
            typer.echo(f"\nTotal: {len(papers)} entries")
            return

        typer.echo(
            "Use --list to browse, --search QUERY to search, "
            "or --gc to clean stale entries."
        )
    finally:
        index.close()
