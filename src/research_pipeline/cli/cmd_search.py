"""CLI handler for the 'search' command."""

import json
import logging
from pathlib import Path

import typer

from research_pipeline.config.loader import load_config
from research_pipeline.models.query_plan import QueryPlan
from research_pipeline.pipeline.source_search import (
    _resolve_sources,
    _search_arxiv,
    _search_dblp,
    _search_huggingface,
    _search_openalex,
    _search_scholar,
    _search_semantic_scholar,
    execute_search,
    load_search_report,
)
from research_pipeline.storage.manifests import write_jsonl
from research_pipeline.storage.workspace import get_stage_dir, init_run

logger = logging.getLogger(__name__)


def run_search(
    topic: str | None = None,
    resume: bool = False,
    config_path: Path | None = None,
    workspace: Path | None = None,
    run_id: str | None = None,
    source: str | None = None,
    strict_sources: bool = False,
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
        source: Source override (arxiv, scholar, semantic_scholar, openalex,
            dblp, huggingface, all, or comma-separated list).
    """
    config = load_config(config_path)
    ws = workspace or Path(config.workspace)
    run_id, run_root = init_run(ws, run_id)

    search_dir = get_stage_dir(run_root, "search")
    if resume:
        saved = load_search_report(search_dir)
        if saved is not None:
            typer.echo(
                f"Resumed {len(saved.candidates)} candidates; status={saved.status}"
            )
            typer.echo(f"Coverage: {search_dir / 'source_coverage.json'}")
            if saved.status == "failed" or (strict_sources and saved.status != "ok"):
                raise typer.Exit(1)
            return

    # Load existing plan or create one
    plan_path = get_stage_dir(run_root, "plan") / "query_plan.json"
    if plan_path.exists():
        plan = QueryPlan.model_validate(
            json.loads(plan_path.read_text(encoding="utf-8"))
        )
    elif topic:
        from research_pipeline.pipeline.query_planning import build_query_plan

        plan = build_query_plan(topic, config)
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
    handlers = {
        "arxiv": lambda p: _search_arxiv(p, config, search_dir),
        "scholar": lambda p: _search_scholar(p, config),
        "semantic_scholar": lambda p: _search_semantic_scholar(p, config),
        "openalex": lambda p: _search_openalex(p, config),
        "dblp": lambda p: _search_dblp(p, config),
        "huggingface": lambda p: _search_huggingface(p, config),
    }
    report = execute_search(plan, config, sources, search_dir, handlers=handlers)
    search_dir.mkdir(parents=True, exist_ok=True)
    candidates_path = search_dir / "candidates.jsonl"
    write_jsonl(
        candidates_path,
        [candidate.model_dump(mode="json") for candidate in report.candidates],
    )
    coverage = report.model_dump(mode="json", exclude={"candidates"})
    coverage["candidate_count"] = len(report.candidates)
    (search_dir / "source_coverage.json").write_text(json.dumps(coverage, indent=2))

    typer.echo("Per-source results:")
    degraded = []
    for name in sources:
        attempts = [attempt for attempt in report.attempts if attempt.source == name]
        count = sum(attempt.candidate_count for attempt in attempts)
        failures = [
            attempt for attempt in attempts if attempt.status not in ("ok", "empty")
        ]
        label = (
            failures[-1].status.upper()
            if failures
            else f"{count} candidates before dedup"
        )
        typer.echo(f"  {name:<16} {label}")
        if failures or count == 0:
            degraded.append(name)
    if degraded:
        typer.echo(
            f"WARNING: {len(degraded)}/{len(sources)} source(s) contributed nothing "
            f"or incomplete coverage: {', '.join(degraded)}",
            err=True,
        )
    typer.echo(f"Run ID: {run_id}")
    typer.echo(
        f"Found {len(report.candidates)} unique candidates; status={report.status}"
    )
    typer.echo(f"Saved to: {candidates_path}")
    if report.status == "failed" or (strict_sources and degraded):
        raise typer.Exit(1)
