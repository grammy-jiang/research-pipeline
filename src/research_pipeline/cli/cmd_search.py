"""CLI handler for the 'search' command."""

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import typer

from research_pipeline.arxiv.client import ArxivClient
from research_pipeline.arxiv.dedup import dedup_across_queries
from research_pipeline.arxiv.query_builder import build_query_from_plan
from research_pipeline.arxiv.rate_limit import ArxivRateLimiter
from research_pipeline.config.loader import load_config
from research_pipeline.config.models import PipelineConfig
from research_pipeline.infra.cache import FileCache
from research_pipeline.infra.clock import date_window
from research_pipeline.infra.http import create_session
from research_pipeline.models.candidate import CandidateRecord
from research_pipeline.models.query_plan import QueryPlan
from research_pipeline.sources.base import dedup_cross_source
from research_pipeline.storage.manifests import write_jsonl
from research_pipeline.storage.workspace import get_stage_dir, init_run

logger = logging.getLogger(__name__)


def _resolve_sources(source_arg: str | None, config_sources: list[str]) -> list[str]:
    """Resolve which sources to use from CLI arg and config.

    Args:
        source_arg: CLI --source value (e.g. 'arxiv', 'scholar', 'all').
        config_sources: Default sources from config.

    Returns:
        List of source names to query.
    """
    if source_arg:
        if source_arg.lower() == "all":
            return ["arxiv", "scholar", "huggingface"]
        return [s.strip() for s in source_arg.split(",")]
    return config_sources


def _search_arxiv(
    plan: QueryPlan, config: PipelineConfig, search_dir: Path
) -> list[CandidateRecord]:
    """Search arXiv API and return deduplicated candidates."""
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

    raw_dir = search_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    arxiv_lists = []
    for q in queries:
        candidates, _ = client.search(
            query=q,
            max_results=config.arxiv.default_page_size,
            date_from=date_from,
            date_to=date_to,
            save_raw_dir=raw_dir,
        )
        arxiv_lists.append(candidates)

    result = dedup_across_queries(arxiv_lists)
    logger.info("arXiv: %d candidates", len(result))
    return result


def _search_scholar(plan: QueryPlan, config: PipelineConfig) -> list[CandidateRecord]:
    """Search Google Scholar and return candidates."""
    backend = config.sources.scholar_backend
    if backend == "serpapi":
        from research_pipeline.sources.scholar_source import SerpAPISource

        source = SerpAPISource(
            api_key=config.sources.serpapi_key,
            min_interval=config.sources.serpapi_min_interval,
        )
    else:
        from research_pipeline.sources.scholar_source import ScholarlySource

        source = ScholarlySource(  # type: ignore[assignment]
            min_interval=config.sources.scholar_min_interval,
        )

    result = source.search(
        topic=plan.topic_raw,
        must_terms=plan.must_terms,
        nice_terms=plan.nice_terms,
        max_results=min(config.arxiv.default_page_size, 20),
    )
    logger.info("Scholar (%s): %d candidates", backend, len(result))
    return result


def _search_huggingface(
    plan: QueryPlan, config: PipelineConfig
) -> list[CandidateRecord]:
    """Search HuggingFace daily papers and return candidates."""
    from research_pipeline.sources.huggingface_source import HuggingFaceSource

    source = HuggingFaceSource(
        min_interval=config.sources.huggingface_min_interval,
        limit=config.sources.huggingface_limit,
    )
    date_from, date_to = date_window(plan.primary_months)
    result = source.search(
        topic=plan.topic_raw,
        must_terms=plan.must_terms,
        nice_terms=plan.nice_terms,
        max_results=min(config.arxiv.default_page_size, 20),
        date_from=date_from,
        date_to=date_to,
    )
    logger.info("HuggingFace: %d candidates", len(result))
    return result


def run_search(
    topic: str | None = None,
    resume: bool = False,
    config_path: Path | None = None,
    workspace: Path | None = None,
    run_id: str | None = None,
    source: str | None = None,
) -> None:
    """Execute the search stage: query enabled sources and collect candidates.

    When multiple sources are enabled, they are searched in parallel.
    Results are deduplicated across sources by arxiv_id and title.

    Args:
        topic: Raw topic (used if plan doesn't exist).
        resume: Skip if already completed.
        config_path: Path to config TOML.
        workspace: Workspace directory.
        run_id: Run ID (required for resume or to use existing plan).
        source: Source override (arxiv, scholar, huggingface, all).
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
            must_terms=topic.lower().split()[:3],
            nice_terms=topic.lower().split()[3:6],
        )
        plan_path.parent.mkdir(parents=True, exist_ok=True)
        plan_path.write_text(plan.model_dump_json(indent=2), encoding="utf-8")
    else:
        typer.echo(
            f"Error: no query plan found at {plan_path}. "
            "Provide a topic argument or use --run-id with an existing plan.",
            err=True,
        )
        raise typer.Exit(1)

    sources = _resolve_sources(source, config.sources.enabled)
    search_dir = get_stage_dir(run_root, "search")
    all_candidates: list[CandidateRecord] = []

    # Run sources in parallel using ThreadPoolExecutor
    futures = {}
    with ThreadPoolExecutor(max_workers=len(sources)) as executor:
        if "arxiv" in sources:
            futures[executor.submit(_search_arxiv, plan, config, search_dir)] = "arxiv"
        if "scholar" in sources:
            futures[executor.submit(_search_scholar, plan, config)] = "scholar"
        if "huggingface" in sources:
            futures[executor.submit(_search_huggingface, plan, config)] = "huggingface"

        for future in as_completed(futures):
            source_name = futures[future]
            try:
                candidates = future.result()
                all_candidates.extend(candidates)
                typer.echo(f"  {source_name}: {len(candidates)} candidates")
            except ImportError as exc:
                install_hint = (
                    "Install with: pipx inject research-pipeline scholarly"
                    if source_name == "scholar"
                    else str(exc)
                )
                logger.error(
                    "%s search failed (missing dependency): %s",
                    source_name,
                    exc,
                )
                typer.echo(
                    f"  {source_name}: SKIPPED (not installed — {install_hint})",
                    err=True,
                )
            except Exception as exc:
                logger.error("%s search failed: %s", source_name, exc)
                typer.echo(f"  {source_name}: FAILED ({exc})", err=True)

    # Cross-source dedup
    deduped = dedup_cross_source(all_candidates)

    candidates_path = search_dir / "candidates.jsonl"
    write_jsonl(candidates_path, [c.model_dump(mode="json") for c in deduped])

    typer.echo(f"Run ID: {run_id}")
    typer.echo(f"Sources: {', '.join(sources)}")
    before = len(all_candidates)
    typer.echo(f"Found {len(deduped)} unique candidates (before dedup: {before})")
    typer.echo(f"Saved to: {candidates_path}")
    logger.info("Search stage complete: %d candidates from %s", len(deduped), sources)
