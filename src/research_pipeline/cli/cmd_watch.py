"""CLI handler for the 'watch' command.

Monitors saved queries for new papers on arXiv. Each invocation checks
for papers published since the last check and reports new matches.
Designed to be run periodically (e.g., via cron).
"""

import json
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path

import typer

from research_pipeline.arxiv.client import ArxivClient
from research_pipeline.arxiv.rate_limit import ArxivRateLimiter
from research_pipeline.infra.http import create_session

logger = logging.getLogger(__name__)

# Default watch state file location
DEFAULT_WATCH_DIR = Path.home() / ".cache" / "research-pipeline" / "watch"
DEFAULT_WATCH_STATE = DEFAULT_WATCH_DIR / "watch_state.json"
DEFAULT_QUERIES_FILE = DEFAULT_WATCH_DIR / "watch_queries.json"


def _load_watch_state(state_path: Path) -> dict[str, str]:
    """Load the watch state (last-checked timestamps per query).

    Args:
        state_path: Path to the state JSON file.

    Returns:
        Dict mapping query name to ISO-format last-checked timestamp.
    """
    if state_path.exists():
        return json.loads(state_path.read_text())  # type: ignore[no-any-return]
    return {}


def _save_watch_state(state_path: Path, state: dict[str, str]) -> None:
    """Save watch state to disk.

    Args:
        state_path: Path to the state JSON file.
        state: Dict mapping query name to ISO timestamp.
    """
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, indent=2))


def _load_queries(queries_path: Path) -> list[dict[str, str]]:
    """Load saved watch queries.

    Args:
        queries_path: Path to the queries JSON file.

    Returns:
        List of query dicts with 'name' and 'query' keys.
    """
    if not queries_path.exists():
        return []
    data = json.loads(queries_path.read_text())
    if isinstance(data, list):
        return data
    return []


def watch_command(
    queries_file: Path = DEFAULT_QUERIES_FILE,
    lookback_days: int = 7,
    max_results: int = 20,
    output: Path | None = None,
    config_path: Path = Path("config.toml"),
) -> None:
    """Check for new papers matching saved queries.

    Queries are stored in a JSON file (default:
    ~/.cache/research-pipeline/watch/watch_queries.json).

    Example queries file:
        [
            {"name": "transformers", "query": "transformer architecture"},
            {"name": "rag", "query": "retrieval augmented generation"}
        ]
    """
    queries = _load_queries(queries_file)
    if not queries:
        logger.error(
            "No queries found. Create %s with your watch queries.",
            queries_file,
        )
        logger.info('Example: [{"name": "topic", "query": "search terms"}]')
        raise typer.Exit(code=1)

    state_path = queries_file.parent / "watch_state.json"
    state = _load_watch_state(state_path)

    session = create_session()
    rate_limiter = ArxivRateLimiter()
    client = ArxivClient(session=session, rate_limiter=rate_limiter)

    now = datetime.now(tz=UTC)
    all_new_papers: dict[str, list[dict[str, str]]] = {}
    total_new = 0

    for query_def in queries:
        name = query_def.get("name", "unnamed")
        query_text = query_def.get("query", "")
        if not query_text:
            continue

        last_checked_str = state.get(name)
        if last_checked_str:
            last_checked = datetime.fromisoformat(last_checked_str)
        else:
            last_checked = now - timedelta(days=lookback_days)

        logger.info(
            "Checking '%s': '%s' (since %s)",
            name,
            query_text,
            last_checked.date(),
        )

        try:
            search_result = client.search(
                query=query_text,
                max_results=max_results,
            )
            # ArxivClient.search() returns tuple; extract candidates list
            results = (
                search_result[0] if isinstance(search_result, tuple) else search_result
            )
        except Exception as exc:
            logger.warning("Search failed for '%s': %s", name, exc)
            continue

        new_papers = []
        for paper in results:
            if paper.published >= last_checked:
                new_papers.append(
                    {
                        "arxiv_id": paper.arxiv_id,
                        "title": paper.title,
                        "published": paper.published.isoformat(),
                        "authors": ", ".join(paper.authors[:3]),
                    }
                )

        if new_papers:
            all_new_papers[name] = new_papers
            total_new += len(new_papers)
            logger.info("  → %d new papers for '%s'", len(new_papers), name)
        else:
            logger.info("  → No new papers for '%s'", name)

        state[name] = now.isoformat()

    _save_watch_state(state_path, state)

    if output and all_new_papers:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(all_new_papers, indent=2, ensure_ascii=False))
        logger.info("New papers written to %s", output)

    logger.info(
        "Watch complete: %d new papers across %d queries",
        total_new,
        len(queries),
    )
