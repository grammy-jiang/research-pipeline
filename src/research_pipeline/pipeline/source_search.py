"""Source acquisition and coverage accounting shared by CLI and MCP."""

import importlib.util
import json
import logging
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Literal

from research_pipeline.arxiv.dedup import dedup_across_queries
from research_pipeline.arxiv.query_builder import build_query_from_plan
from research_pipeline.arxiv.rate_limit import ArxivRateLimiter
from research_pipeline.config.models import PipelineConfig
from research_pipeline.infra.cache import FileCache
from research_pipeline.infra.clock import date_window, provider_date
from research_pipeline.infra.http import create_session
from research_pipeline.infra.request_budget import SourceCooldown
from research_pipeline.infra.retry import safe_exception
from research_pipeline.models.candidate import CandidateRecord
from research_pipeline.models.query_plan import QueryPlan
from research_pipeline.models.source_search import SourceAttempt, SourceSearchReport
from research_pipeline.sources.base import (
    SourceSearchFailure,
    SourceUnavailable,
    dedup_cross_source,
)

logger = logging.getLogger(__name__)


def _checked_source(
    source: Any, result: list[CandidateRecord]
) -> list[CandidateRecord]:
    failure = getattr(source, "last_error", None)
    if isinstance(failure, BaseException):
        raise SourceSearchFailure(failure, result)
    return result


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
            return [
                "arxiv",
                "scholar",
                "semantic_scholar",
                "openalex",
                "dblp",
                "huggingface",
            ]
        return [s.strip() for s in source_arg.split(",")]
    return config_sources


def _search_arxiv(
    plan: QueryPlan, config: PipelineConfig, search_dir: Path
) -> list[CandidateRecord]:
    """Search arXiv API and return deduplicated candidates."""
    from research_pipeline.arxiv.client import ArxivClient

    rate_limiter = ArxivRateLimiter(min_interval=config.arxiv.min_interval_seconds)
    session = create_session(config.contact_email, config.arxiv.min_interval_seconds)
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

    raw_dir = search_dir / "raw" / f"window_{plan.primary_months}"
    raw_dir.mkdir(parents=True, exist_ok=True)

    arxiv_lists = []
    attempted: list[str] = []
    for index, q in enumerate(queries):
        attempted.append(q)
        try:
            candidates, _ = client.search(
                query=q,
                max_results=config.arxiv.default_page_size,
                date_from=date_from,
                date_to=date_to,
                save_raw_dir=raw_dir / f"q{index:02d}",
            )
        except Exception as exc:
            partial = getattr(client, "partial_candidates", [])
            if isinstance(partial, list):
                arxiv_lists.append(partial)
            raise SourceSearchFailure(
                exc, dedup_across_queries(arxiv_lists), attempted
            ) from None
        arxiv_lists.append(candidates)

    result = dedup_across_queries(arxiv_lists)
    logger.info("arXiv: %d candidates", len(result))
    return result


def _search_scholar(plan: QueryPlan, config: PipelineConfig) -> list[CandidateRecord]:
    """Search Google Scholar and return candidates."""
    backend = config.sources.scholar_backend
    module = "serpapi" if backend == "serpapi" else "scholarly"
    if importlib.util.find_spec(module) is None:
        raise SourceUnavailable(f"Missing optional dependency: {module}")
    if backend == "serpapi" and not config.sources.serpapi_key:
        raise SourceUnavailable("SerpAPI credentials are missing")
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
    return _checked_source(source, result)


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
        date_from=provider_date(date_from),
        date_to=provider_date(date_to),
    )
    logger.info("HuggingFace: %d candidates", len(result))
    return _checked_source(source, result)


def _search_semantic_scholar(
    plan: QueryPlan, config: PipelineConfig
) -> list[CandidateRecord]:
    """Search Semantic Scholar and return candidates."""
    from research_pipeline.sources.semantic_scholar_source import SemanticScholarSource

    source = SemanticScholarSource(
        api_key=config.sources.semantic_scholar_api_key,
        min_interval=config.sources.semantic_scholar_min_interval,
    )
    date_from, date_to = date_window(plan.primary_months)
    result = source.search(
        topic=plan.topic_raw,
        must_terms=plan.must_terms,
        nice_terms=plan.nice_terms,
        max_results=min(config.arxiv.default_page_size, 50),
        date_from=date_from,
        date_to=date_to,
    )
    logger.info("Semantic Scholar: %d candidates", len(result))
    return _checked_source(source, result)


def _search_openalex(plan: QueryPlan, config: PipelineConfig) -> list[CandidateRecord]:
    """Search OpenAlex and return candidates."""
    from research_pipeline.sources.openalex_source import OpenAlexSource

    source = OpenAlexSource(
        api_key=config.sources.openalex_api_key,
        min_interval=config.sources.openalex_min_interval,
    )
    date_from, date_to = date_window(plan.primary_months)
    result = source.search(
        topic=plan.topic_raw,
        must_terms=plan.must_terms,
        nice_terms=plan.nice_terms,
        max_results=min(config.arxiv.default_page_size, 50),
        date_from=date_from,
        date_to=date_to,
    )
    logger.info("OpenAlex: %d candidates", len(result))
    return _checked_source(source, result)


def _search_dblp(plan: QueryPlan, config: PipelineConfig) -> list[CandidateRecord]:
    """Search DBLP and return candidates."""
    from research_pipeline.sources.dblp_source import DBLPSource

    source = DBLPSource(
        min_interval=config.sources.dblp_min_interval,
    )
    result = source.search(
        topic=plan.topic_raw,
        must_terms=plan.must_terms,
        nice_terms=plan.nice_terms,
        max_results=min(config.arxiv.default_page_size, 30),
    )
    logger.info("DBLP: %d candidates", len(result))
    return _checked_source(source, result)


def execute_search(
    plan: QueryPlan,
    config: PipelineConfig,
    sources: list[str],
    search_dir: Path,
    handlers: dict[str, Callable[[QueryPlan], list[CandidateRecord]]] | None = None,
) -> SourceSearchReport:
    """Execute supported query plans, retaining partial evidence and real coverage."""
    sources = list(dict.fromkeys(sources))
    if not sources:
        raise ValueError("At least one source must be selected")
    if handlers is None:
        handlers = {
            "arxiv": lambda p: _search_arxiv(p, config, search_dir),
            "scholar": lambda p: _search_scholar(p, config),
            "semantic_scholar": lambda p: _search_semantic_scholar(p, config),
            "openalex": lambda p: _search_openalex(p, config),
            "dblp": lambda p: _search_dblp(p, config),
            "huggingface": lambda p: _search_huggingface(p, config),
        }
    collected: list[CandidateRecord] = []
    attempts: list[SourceAttempt] = []

    def one_source(
        name: str, window: int
    ) -> tuple[list[CandidateRecord], list[SourceAttempt]]:
        records: list[SourceAttempt] = []
        candidates: list[CandidateRecord] = []
        variants = plan.source_queries.get(name, [])[: config.search.max_query_variants]
        if name in ("arxiv", "huggingface") or not variants:
            variants = [""]
        for variant in variants:
            current = plan.model_copy(update={"primary_months": window})
            if variant:
                current = current.model_copy(
                    update={
                        "topic_raw": variant,
                        "must_terms": [],
                        "nice_terms": [],
                    }
                )
            keyword_query = (
                " ".join(current.must_terms[:3] + current.nice_terms[:2])
                or current.topic_raw
            )
            queries = (
                build_query_from_plan(current) if name == "arxiv" else [keyword_query]
            )
            strategy = (
                "arxiv_variants"
                if name == "arxiv"
                else "daily_feed_then_local_filter"
                if name == "huggingface"
                else "explicit_source_query"
                if variant
                else "single_keyword_query"
            )
            start, end = date_window(window)
            record = SourceAttempt(
                source=name,
                window_months=window,
                date_from=provider_date(start)
                if name in ("arxiv", "openalex", "semantic_scholar")
                else None,
                date_to=provider_date(end)
                if name in ("arxiv", "openalex", "semantic_scholar")
                else None,
                query_strategy=strategy,
                planned_queries=queries,
                status="failed",
            )
            try:
                handler = handlers.get(name)
                if handler is None:
                    raise SourceUnavailable("Unknown source")
                result = handler(current)
                record.attempted_queries = queries
                record.candidate_count = len(result)
                record.status = "ok" if result else "empty"
                candidates.extend(result)
            except SourceSearchFailure as exc:
                candidates.extend(exc.candidates)
                record.candidate_count = len(exc.candidates)
                record.attempted_queries = (
                    exc.attempted_queries
                    if exc.attempted_queries is not None
                    else queries[:1]
                )
                record.error_type = type(exc.cause).__name__
                record.detail = safe_exception(exc.cause)
                if isinstance(exc.cause, SourceUnavailable | ImportError):
                    record.status = "unavailable"
                    record.attempted_queries = []
                elif isinstance(exc.cause, SourceCooldown):
                    record.status = "cooldown"
                    record.not_before = exc.cause.not_before
                    if exc.cause.response is None:
                        record.attempted_queries = []
                else:
                    record.status = "partial" if exc.candidates else "failed"
            except Exception as exc:
                record.status = (
                    "unavailable"
                    if isinstance(exc, SourceUnavailable | ImportError)
                    else "failed"
                )
                record.error_type = type(exc).__name__
                record.detail = safe_exception(exc)
            records.append(record)
            if record.status not in ("ok", "empty"):
                break
        return candidates, records

    def run_window(selected: list[str], window: int) -> None:
        with ThreadPoolExecutor(max_workers=len(selected)) as executor:
            futures = {
                executor.submit(one_source, name, window): name for name in selected
            }
            for future in as_completed(futures):
                candidates, records = future.result()
                collected.extend(candidates)
                attempts.extend(records)

    run_window(sources, plan.primary_months)
    primary_count = len(dedup_cross_source(collected))
    if (
        primary_count < plan.sparsity_thresholds.min_candidates
        and plan.fallback_months > plan.primary_months
    ):
        # A source error is not sparse-result evidence. Do not retry failed or
        # cooling-down sources as a date fallback. DBLP/daily feeds have no
        # equivalent historical date-window query.
        eligible = [
            name
            for name in sources
            if name in ("arxiv", "semantic_scholar", "openalex")
            and all(a.status in ("ok", "empty") for a in attempts if a.source == name)
        ]
        if eligible:
            run_window(eligible, plan.fallback_months)
    deduped = dedup_cross_source(collected)
    degraded = any(a.status not in ("ok", "empty") for a in attempts)
    status: Literal["ok", "empty", "partial", "failed"]
    status = (
        ("partial" if degraded else "ok")
        if deduped
        else ("failed" if degraded else "empty")
    )
    return SourceSearchReport(
        status=status, sources=sources, attempts=attempts, candidates=deduped
    )


def load_search_report(search_dir: Path) -> SourceSearchReport | None:
    """Resume saved evidence without issuing requests or erasing coverage."""
    path = search_dir / "candidates.jsonl"
    if not path.exists():
        return None
    candidates = [
        CandidateRecord.model_validate(json.loads(line))
        for line in path.read_text().splitlines()
        if line.strip()
    ]
    coverage_path = search_dir / "source_coverage.json"
    if coverage_path.exists():
        data = json.loads(coverage_path.read_text())
        if data.get("candidate_count") != len(candidates):
            raise ValueError("Saved candidate count does not match source coverage")
        data["candidates"] = candidates
        return SourceSearchReport.model_validate(data)
    # Legacy artifacts carry useful evidence but cannot prove source coverage.
    return SourceSearchReport(
        status="partial" if candidates else "failed",
        sources=sorted({c.source for c in candidates}),
        candidates=candidates,
        attempts=[
            SourceAttempt(
                source="legacy",
                window_months=0,
                query_strategy="saved_artifact",
                status="unavailable",
                detail="Source coverage was not recorded",
            )
        ],
    )
