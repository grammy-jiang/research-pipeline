"""CLI handler for the 'search' command."""

import json
import logging
from pathlib import Path

import typer

from arxiv_paper_pipeline.arxiv.client import ArxivClient
from arxiv_paper_pipeline.arxiv.dedup import dedup_across_queries
from arxiv_paper_pipeline.arxiv.query_builder import build_query_from_plan
from arxiv_paper_pipeline.arxiv.rate_limit import ArxivRateLimiter
from arxiv_paper_pipeline.config.loader import load_config
from arxiv_paper_pipeline.infra.cache import FileCache
from arxiv_paper_pipeline.infra.clock import date_window
from arxiv_paper_pipeline.infra.http import create_session
from arxiv_paper_pipeline.models.query_plan import QueryPlan
from arxiv_paper_pipeline.storage.manifests import write_jsonl
from arxiv_paper_pipeline.storage.workspace import get_stage_dir, init_run

logger = logging.getLogger(__name__)


def run_search(
    topic: str | None = None,
    resume: bool = False,
    config_path: Path | None = None,
    workspace: Path | None = None,
    run_id: str | None = None,
) -> None:
    """Execute the search stage: query arXiv and collect candidates.

    Args:
        topic: Raw topic (used if plan doesn't exist).
        resume: Skip if already completed.
        config_path: Path to config TOML.
        workspace: Workspace directory.
        run_id: Run ID (required for resume or to use existing plan).
    """
    config = load_config(config_path)
    ws = workspace or Path(config.workspace)
    run_id, run_root = init_run(ws, run_id)

    # Load existing plan or create one
    plan_path = get_stage_dir(run_root, "plan") / "query_plan.json"
    if plan_path.exists():
        plan = QueryPlan.model_validate(
            json.loads(plan_path.read_text(encoding="utf-8"))
        )
    elif topic:
        plan = QueryPlan(
            topic_raw=topic,
            topic_normalized=topic.lower().strip(),
            must_terms=topic.lower().split()[:5],
        )
        plan_path.parent.mkdir(parents=True, exist_ok=True)
        plan_path.write_text(plan.model_dump_json(indent=2), encoding="utf-8")
    else:
        typer.echo(
            "Error: provide a topic or use --run-id with existing plan.", err=True
        )
        raise typer.Exit(1)

    # Setup client
    rate_limiter = ArxivRateLimiter(min_interval=config.arxiv.min_interval_seconds)
    session = create_session(config.contact_email)
    cache: FileCache | None = None
    if config.cache.enabled:
        cache = FileCache(
            Path(config.cache.cache_dir).expanduser(),
            ttl_hours=config.cache.search_snapshot_ttl_hours,
        )
    client = ArxivClient(
        rate_limiter=rate_limiter,
        cache=cache,
        session=session,
        base_url=config.arxiv.base_url,
        request_timeout=config.arxiv.request_timeout_seconds,
    )

    queries = build_query_from_plan(plan)
    date_from, date_to = date_window(plan.primary_months)

    search_dir = get_stage_dir(run_root, "search")
    raw_dir = search_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    all_lists = []
    for q in queries:
        candidates, _ = client.search(
            query=q,
            max_results=config.arxiv.default_page_size,
            date_from=date_from,
            date_to=date_to,
            save_raw_dir=raw_dir,
        )
        all_lists.append(candidates)

    deduped = dedup_across_queries(all_lists)
    candidates_path = search_dir / "candidates.jsonl"
    write_jsonl(candidates_path, [c.model_dump(mode="json") for c in deduped])

    typer.echo(f"Run ID: {run_id}")
    typer.echo(f"Found {len(deduped)} unique candidates")
    typer.echo(f"Saved to: {candidates_path}")
    logger.info("Search stage complete: %d candidates", len(deduped))
