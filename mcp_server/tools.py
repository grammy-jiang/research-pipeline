"""MCP tool implementations that wrap core pipeline services.

Each function accepts a typed input schema and returns a ToolResult.
These are pure adapter functions — no business logic.
"""

from __future__ import annotations

import contextlib
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mcp.server.fastmcp import Context

from mcp_server.schemas import (
    AdaptiveStoppingInput,
    AnalyzeClaimsInput,
    AnalyzePapersInput,
    BlindingAuditInput,
    CbrLookupInput,
    CbrRetainInput,
    CiteContextInput,
    ClusterInput,
    CoherenceInput,
    CompareRunsInput,
    ConfidenceLayersInput,
    ConsolidationInput,
    ConvertFileInput,
    ConvertFineInput,
    ConvertPdfsInput,
    ConvertRoughInput,
    DownloadPdfsInput,
    DualMetricsInput,
    EnrichInput,
    EvalLogInput,
    EvaluateInput,
    EvaluateQualityInput,
    EvidenceAggregateInput,
    ExpandCitationsInput,
    ExportBibtexInput,
    ExportHtmlInput,
    ExtractContentInput,
    FeedbackInput,
    GateInfoInput,
    GetRunManifestInput,
    KGIngestInput,
    KGQualityInput,
    KGQueryInput,
    KGStatsInput,
    ListBackendsInput,
    ManageIndexInput,
    MemoryEpisodesInput,
    MemorySearchInput,
    MemoryStatsInput,
    ModelRoutingInfoInput,
    PlanTopicInput,
    ReportInput,
    RunPipelineInput,
    ScoreClaimsInput,
    ScreenCandidatesInput,
    SearchInput,
    SummarizePapersInput,
    ToolResult,
    ValidateReportInput,
    VerifyStageInput,
    WatchInput,
)

logger = logging.getLogger(__name__)


def _report_progress(
    ctx: Context | None, progress: float, total: float, message: str = ""
) -> None:
    """Safely report progress via MCP context if available."""
    if ctx is not None:
        with contextlib.suppress(Exception):
            ctx.report_progress(progress, total, message)


def _log_info(ctx: Context | None, message: str) -> None:
    """Log via MCP context if available, always log via Python logger."""
    logger.info(message)
    if ctx is not None:
        with contextlib.suppress(Exception):
            ctx.info(message)


def _backend_kwargs(
    backend_name: str,
    config: object,
) -> dict[str, object]:
    """Build constructor kwargs for a converter backend from config."""
    if backend_name == "docling":
        return {"timeout_seconds": config.conversion.timeout_seconds}  # type: ignore[union-attr]
    if backend_name == "marker":
        mc = config.conversion.marker  # type: ignore[union-attr]
        kwargs: dict[str, object] = {"force_ocr": mc.force_ocr}
        if mc.use_llm:
            kwargs["use_llm"] = True
            if mc.llm_service:
                kwargs["llm_service"] = mc.llm_service
            if mc.llm_api_key:
                kwargs["llm_api_key"] = mc.llm_api_key
        return kwargs
    if backend_name == "mathpix":
        mp = config.conversion.mathpix  # type: ignore[union-attr]
        return {"app_id": mp.app_id, "app_key": mp.app_key}
    if backend_name == "datalab":
        dl = config.conversion.datalab  # type: ignore[union-attr]
        return {"api_key": dl.api_key, "mode": dl.mode}
    if backend_name == "llamaparse":
        lp = config.conversion.llamaparse  # type: ignore[union-attr]
        return {"api_key": lp.api_key}
    if backend_name == "mistral_ocr":
        mo = config.conversion.mistral_ocr  # type: ignore[union-attr]
        return {"api_key": mo.api_key, "model": mo.model}
    if backend_name == "openai_vision":
        ov = config.conversion.openai_vision  # type: ignore[union-attr]
        return {"api_key": ov.api_key, "model": ov.model}
    return {}


def _resolve_workspace(workspace: str) -> Path:
    """Resolve workspace path."""
    return Path(workspace).expanduser().resolve()


def _resolve_run_id(run_id: str) -> str:
    """Generate a run ID if not provided."""
    if run_id:
        return run_id
    from research_pipeline.infra.clock import utc_now

    return utc_now().strftime("%Y%m%dT%H%M%SZ")


def _get_run_root(ws: Path, rid: str) -> Path:
    """Compute the run root directory from workspace and run ID."""
    from research_pipeline.infra.paths import run_dir

    return run_dir(ws, rid)


def plan_topic(params: PlanTopicInput, ctx: Context | None = None) -> ToolResult:
    """Create a query plan from a natural language topic."""
    try:
        from research_pipeline.config.loader import load_config
        from research_pipeline.models.query_plan import QueryPlan
        from research_pipeline.storage.workspace import init_run

        ws = _resolve_workspace(params.workspace)
        rid = _resolve_run_id(params.run_id)
        config = load_config()

        rid, run_root = init_run(ws, rid)
        plan = QueryPlan(
            topic_raw=params.topic,
            topic_normalized=params.topic.lower().strip(),
            must_terms=params.topic.lower().split()[:5],
            nice_terms=[],
            query_variants=[],
            candidate_categories=[],
            negative_terms=[],
            primary_months=config.search.primary_months,
        )
        plan_path = run_root / "plan" / "query_plan.json"
        plan_path.parent.mkdir(parents=True, exist_ok=True)
        plan_path.write_text(plan.model_dump_json(indent=2))

        logger.info("Created query plan for topic: %s", params.topic)
        return ToolResult(
            success=True,
            message=f"Query plan created for topic: {params.topic}",
            artifacts={"query_plan": str(plan_path), "run_id": rid},
        )
    except Exception as exc:
        logger.error("plan_topic failed: %s", exc)
        return ToolResult(success=False, message=f"Failed: {exc}")


def search(params: SearchInput, ctx: Context | None = None) -> ToolResult:
    """Search configured academic paper sources using the query plan."""
    try:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        from research_pipeline.arxiv.client import ArxivClient
        from research_pipeline.arxiv.dedup import dedup_across_queries
        from research_pipeline.arxiv.query_builder import build_query_from_plan
        from research_pipeline.arxiv.rate_limit import ArxivRateLimiter
        from research_pipeline.config.loader import load_config
        from research_pipeline.infra.cache import FileCache
        from research_pipeline.infra.clock import date_window
        from research_pipeline.infra.http import create_session
        from research_pipeline.models.candidate import CandidateRecord
        from research_pipeline.models.query_plan import QueryPlan
        from research_pipeline.sources.base import dedup_cross_source
        from research_pipeline.storage.workspace import get_stage_dir

        config = load_config()
        ws = _resolve_workspace(params.workspace)
        rid = _resolve_run_id(params.run_id)
        run_root = _get_run_root(ws, rid)

        stage_dir = get_stage_dir(run_root, "search")
        candidates_path = stage_dir / "candidates.jsonl"

        if params.resume and candidates_path.exists():
            count = sum(1 for _ in candidates_path.open())
            return ToolResult(
                success=True,
                message=f"Resumed: {count} existing candidates found.",
                artifacts={"candidates": str(candidates_path)},
            )

        # Load or create query plan
        plan_path = get_stage_dir(run_root, "plan") / "query_plan.json"
        if plan_path.exists():
            plan_data = json.loads(plan_path.read_text())
            plan = QueryPlan.model_validate(plan_data)
        elif params.topic:
            plan = QueryPlan(
                topic_raw=params.topic,
                topic_normalized=params.topic.lower().strip(),
                must_terms=params.topic.lower().split()[:3],
                nice_terms=params.topic.lower().split()[3:6],
            )
            plan_path.parent.mkdir(parents=True, exist_ok=True)
            plan_path.write_text(plan.model_dump_json(indent=2))
        else:
            return ToolResult(
                success=False,
                message="No query plan found and no topic provided.",
            )

        # Resolve sources
        if params.source:
            if params.source.lower() == "all":
                sources = ["arxiv", "scholar"]
            else:
                sources = [s.strip() for s in params.source.split(",")]
        else:
            sources = config.sources.enabled

        # Shared cache from config (same as CLI)
        cache: FileCache | None = None
        if config.cache.enabled:
            cache_dir = Path(config.cache.cache_dir).expanduser()
            cache = FileCache(
                cache_dir, ttl_hours=config.cache.search_snapshot_ttl_hours
            )

        all_candidates: list[CandidateRecord] = []
        source_counts: dict[str, int] = {}

        def _do_arxiv() -> list[CandidateRecord]:
            rate_limiter = ArxivRateLimiter(
                min_interval=config.arxiv.min_interval_seconds
            )
            session = create_session(config.contact_email)
            client = ArxivClient(
                session=session,
                rate_limiter=rate_limiter,
                cache=cache,
                base_url=config.arxiv.base_url,
                request_timeout=config.arxiv.request_timeout_seconds,
            )
            queries = build_query_from_plan(plan)
            date_from, date_to = date_window(plan.primary_months)
            arxiv_lists = []
            for q in queries:
                candidates, _ = client.search(
                    query=q,
                    max_results=config.arxiv.default_page_size,
                    date_from=date_from,
                    date_to=date_to,
                )
                arxiv_lists.append(candidates)
            result = dedup_across_queries(arxiv_lists)
            logger.info("arXiv: %d candidates", len(result))
            return result

        def _do_scholar() -> list[CandidateRecord]:
            backend = config.sources.scholar_backend
            if backend == "serpapi":
                from research_pipeline.sources.scholar_source import SerpAPISource

                source_obj = SerpAPISource(
                    api_key=config.sources.serpapi_key,
                    min_interval=config.sources.serpapi_min_interval,
                )
            else:
                from research_pipeline.sources.scholar_source import ScholarlySource

                source_obj = ScholarlySource(  # type: ignore[assignment]
                    min_interval=config.sources.scholar_min_interval,
                )
            result = source_obj.search(
                topic=plan.topic_raw,
                must_terms=plan.must_terms,
                nice_terms=plan.nice_terms,
                max_results=min(config.arxiv.default_page_size, 20),
            )
            logger.info("Scholar (%s): %d candidates", backend, len(result))
            return result

        # Run sources in parallel
        futures = {}
        with ThreadPoolExecutor(max_workers=len(sources)) as executor:
            if "arxiv" in sources:
                futures[executor.submit(_do_arxiv)] = "arxiv"
            if "scholar" in sources:
                futures[executor.submit(_do_scholar)] = "scholar"

            for future in as_completed(futures):
                source_name = futures[future]
                try:
                    candidates = future.result()
                    all_candidates.extend(candidates)
                    source_counts[source_name] = len(candidates)
                except Exception as exc:
                    logger.error("%s search failed: %s", source_name, exc)
                    source_counts[source_name] = -1  # indicates failure

        deduped = dedup_cross_source(all_candidates)

        stage_dir.mkdir(parents=True, exist_ok=True)
        with candidates_path.open("w") as fh:
            for c in deduped:
                fh.write(c.model_dump_json() + "\n")

        source_summary = ", ".join(
            f"{k}: {v}" if v >= 0 else f"{k}: FAILED" for k, v in source_counts.items()
        )
        logger.info("Search returned %d unique candidates", len(deduped))
        return ToolResult(
            success=True,
            message=(f"Found {len(deduped)} unique candidates ({source_summary})."),
            artifacts={
                "candidates": str(candidates_path),
                "count": len(deduped),
                "sources": source_counts,
            },
        )
    except Exception as exc:
        logger.error("search failed: %s", exc)
        return ToolResult(success=False, message=f"Failed: {exc}")


def screen_candidates(
    params: ScreenCandidatesInput, ctx: Context | None = None
) -> ToolResult:
    """Two-stage relevance screening of candidates."""
    try:
        from research_pipeline.config.loader import load_config
        from research_pipeline.models.candidate import CandidateRecord
        from research_pipeline.screening.heuristic import (
            score_candidates,
            select_topk,
        )
        from research_pipeline.storage.workspace import get_stage_dir

        ws = _resolve_workspace(params.workspace)
        rid = _resolve_run_id(params.run_id)
        config = load_config()
        run_root = _get_run_root(ws, rid)

        search_dir = get_stage_dir(run_root, "search")
        candidates_path = search_dir / "candidates.jsonl"
        if not candidates_path.exists():
            return ToolResult(
                success=False,
                message="No candidates found. Run search_arxiv first.",
            )

        screen_dir = get_stage_dir(run_root, "screen")
        screen_dir.mkdir(parents=True, exist_ok=True)

        # Load candidates
        candidates = []
        for line in candidates_path.read_text().strip().split("\n"):
            if line:
                candidates.append(CandidateRecord.model_validate_json(line))

        # Load query plan for terms
        plan_path = get_stage_dir(run_root, "plan") / "query_plan.json"
        plan_data = json.loads(plan_path.read_text())

        scored = score_candidates(
            candidates,
            must_terms=plan_data.get("must_terms", []),
            nice_terms=plan_data.get("nice_terms", []),
            negative_terms=plan_data.get("negative_terms", []),
            target_categories=plan_data.get("candidate_categories", []),
        )
        shortlist = select_topk(candidates, scored, top_k=config.screen.cheap_top_k)

        scores_path = screen_dir / "cheap_scores.jsonl"
        with scores_path.open("w") as fh:
            for s in scored:
                fh.write(s.model_dump_json() + "\n")

        shortlist_path = screen_dir / "shortlist.json"
        shortlist_data = [
            {"candidate": c.model_dump(), "score": s.model_dump()} for c, s in shortlist
        ]
        shortlist_path.write_text(json.dumps(shortlist_data, indent=2))

        logger.info(
            "Screening: %d candidates → %d shortlisted",
            len(candidates),
            len(shortlist),
        )
        return ToolResult(
            success=True,
            message=f"Screened {len(candidates)} → {len(shortlist)} shortlisted.",
            artifacts={
                "scores": str(scores_path),
                "shortlist": str(shortlist_path),
                "shortlist_count": len(shortlist),
            },
        )
    except Exception as exc:
        logger.error("screen_candidates failed: %s", exc)
        return ToolResult(success=False, message=f"Failed: {exc}")


def download_pdfs(params: DownloadPdfsInput, ctx: Context | None = None) -> ToolResult:
    """Download shortlisted PDFs from arXiv."""
    try:
        from research_pipeline.arxiv.rate_limit import ArxivRateLimiter
        from research_pipeline.config.loader import load_config
        from research_pipeline.download.pdf import download_batch
        from research_pipeline.infra.http import create_session
        from research_pipeline.storage.workspace import get_stage_dir

        ws = _resolve_workspace(params.workspace)
        rid = _resolve_run_id(params.run_id)
        config = load_config()
        run_root = _get_run_root(ws, rid)

        screen_dir = get_stage_dir(run_root, "screen")
        shortlist_path = screen_dir / "shortlist.json"
        if not shortlist_path.exists():
            return ToolResult(
                success=False,
                message="No shortlist found. Run screen_candidates first.",
            )

        download_dir = get_stage_dir(run_root, "download")
        download_dir.mkdir(parents=True, exist_ok=True)

        shortlist = json.loads(shortlist_path.read_text())
        session = create_session()
        rate_limiter = ArxivRateLimiter()

        results = download_batch(
            shortlist,
            download_dir,
            session,
            rate_limiter,
            max_downloads=config.download.max_per_run,
        )

        manifest_path = download_dir / "download_manifest.jsonl"
        with manifest_path.open("w") as fh:
            for r in results:
                fh.write(r.model_dump_json() + "\n")

        success_count = sum(1 for r in results if r.status == "downloaded")
        logger.info("Downloaded %d/%d PDFs", success_count, len(results))
        return ToolResult(
            success=True,
            message=f"Downloaded {success_count}/{len(results)} PDFs.",
            artifacts={
                "manifest": str(manifest_path),
                "download_dir": str(download_dir),
                "downloaded": success_count,
            },
        )
    except Exception as exc:
        logger.error("download_pdfs failed: %s", exc)
        return ToolResult(success=False, message=f"Failed: {exc}")


def convert_pdfs(params: ConvertPdfsInput, ctx: Context | None = None) -> ToolResult:
    """Convert downloaded PDFs to Markdown."""
    try:
        from research_pipeline.config.loader import load_config
        from research_pipeline.conversion.registry import (
            _ensure_builtins_registered,
            get_backend,
        )
        from research_pipeline.models.download import DownloadManifestEntry
        from research_pipeline.storage.workspace import get_stage_dir

        _ensure_builtins_registered()

        ws = _resolve_workspace(params.workspace)
        rid = _resolve_run_id(params.run_id)
        run_root = _get_run_root(ws, rid)
        config = load_config()

        download_dir = get_stage_dir(run_root, "download")
        manifest_path = download_dir / "download_manifest.jsonl"
        if not manifest_path.exists():
            return ToolResult(
                success=False,
                message="No download manifest found. Run download_pdfs first.",
            )

        convert_dir = get_stage_dir(run_root, "convert")
        convert_dir.mkdir(parents=True, exist_ok=True)

        backend_name = params.backend or config.conversion.backend
        kwargs = _backend_kwargs(backend_name, config)
        backend = get_backend(backend_name, **kwargs)

        entries = []
        for line in manifest_path.read_text().strip().split("\n"):
            if line:
                entries.append(DownloadManifestEntry.model_validate_json(line))

        results = []
        for entry in entries:
            if entry.status != "downloaded":
                continue
            pdf_path = Path(entry.local_path)
            if pdf_path.exists():
                result = backend.convert(pdf_path, convert_dir, force=params.force)
                results.append(result)

        conv_manifest = convert_dir / "convert_manifest.jsonl"
        with conv_manifest.open("w") as fh:
            for r in results:
                fh.write(r.model_dump_json() + "\n")

        success_count = sum(1 for r in results if r.status == "converted")
        logger.info("Converted %d/%d PDFs", success_count, len(results))
        return ToolResult(
            success=True,
            message=(
                f"Converted {success_count}/{len(results)} PDFs to Markdown "
                f"(backend={backend_name})."
            ),
            artifacts={
                "manifest": str(conv_manifest),
                "convert_dir": str(convert_dir),
                "converted": success_count,
                "backend": backend_name,
            },
        )
    except ImportError as exc:
        return ToolResult(
            success=False,
            message=(
                f"Converter backend is not installed: {exc}. "
                "Install the corresponding extra."
            ),
        )
    except Exception as exc:
        logger.error("convert_pdfs failed: %s", exc)
        return ToolResult(success=False, message=f"Failed: {exc}")


def extract_content(
    params: ExtractContentInput, ctx: Context | None = None
) -> ToolResult:
    """Extract structured content from converted Markdown."""
    try:
        from research_pipeline.extraction.extractor import extract_from_markdown
        from research_pipeline.models.download import DownloadManifestEntry
        from research_pipeline.storage.workspace import get_stage_dir

        ws = _resolve_workspace(params.workspace)
        rid = _resolve_run_id(params.run_id)
        run_root = _get_run_root(ws, rid)

        convert_dir = get_stage_dir(run_root, "convert")
        extract_dir = get_stage_dir(run_root, "extract")
        extract_dir.mkdir(parents=True, exist_ok=True)

        md_files = list(convert_dir.glob("*.md"))
        if not md_files:
            return ToolResult(
                success=False,
                message="No Markdown files found. Run convert_pdfs first.",
            )

        # Load download manifest to get arxiv_id/version per file
        download_dir = get_stage_dir(run_root, "download")
        manifest_path = download_dir / "download_manifest.jsonl"
        id_map: dict[str, DownloadManifestEntry] = {}
        if manifest_path.exists():
            for line in manifest_path.read_text().strip().split("\n"):
                if line:
                    entry = DownloadManifestEntry.model_validate_json(line)
                    stem = Path(entry.local_path).stem
                    id_map[stem] = entry

        results = []
        for md_path in md_files:
            entry = id_map.get(md_path.stem)
            arxiv_id = entry.arxiv_id if entry else md_path.stem
            version = entry.version if entry else "v1"
            result = extract_from_markdown(md_path, arxiv_id, version)
            results.append(result)

        logger.info("Extracted content from %d files", len(results))
        return ToolResult(
            success=True,
            message=f"Extracted content from {len(results)} Markdown files.",
            artifacts={
                "extract_dir": str(extract_dir),
                "file_count": len(results),
            },
        )
    except Exception as exc:
        logger.error("extract_content failed: %s", exc)
        return ToolResult(success=False, message=f"Failed: {exc}")


def summarize_papers(
    params: SummarizePapersInput, ctx: Context | None = None
) -> ToolResult:
    """Generate per-paper summaries and cross-paper synthesis."""
    try:
        from research_pipeline.models.download import DownloadManifestEntry
        from research_pipeline.storage.workspace import get_stage_dir
        from research_pipeline.summarization.per_paper import (
            extract_paper,
            project_extraction_to_summary,
            render_extraction_markdown,
        )
        from research_pipeline.summarization.synthesis import (
            project_structured_synthesis_to_report,
            render_structured_synthesis_markdown,
            synthesize_extractions,
        )

        ws = _resolve_workspace(params.workspace)
        rid = _resolve_run_id(params.run_id)
        run_root = _get_run_root(ws, rid)

        convert_dir = get_stage_dir(run_root, "convert")
        summarize_dir = get_stage_dir(run_root, "summarize")
        summarize_dir.mkdir(parents=True, exist_ok=True)
        extraction_dir = summarize_dir / "extractions"
        extraction_dir.mkdir(parents=True, exist_ok=True)

        md_files = list(convert_dir.glob("*.md"))
        if not md_files:
            return ToolResult(
                success=False,
                message="No Markdown files found. Run convert_pdfs first.",
            )

        # Load plan for topic terms
        plan_path = get_stage_dir(run_root, "plan") / "query_plan.json"
        plan_data = json.loads(plan_path.read_text())
        topic_terms = plan_data.get("must_terms", []) + plan_data.get("nice_terms", [])
        topic = plan_data.get("topic_raw", "")

        # Load download manifest for arxiv_id/version/title
        download_dir = get_stage_dir(run_root, "download")
        manifest_path = download_dir / "download_manifest.jsonl"
        id_map: dict[str, DownloadManifestEntry] = {}
        if manifest_path.exists():
            for line in manifest_path.read_text().strip().split("\n"):
                if line:
                    entry = DownloadManifestEntry.model_validate_json(line)
                    stem = Path(entry.local_path).stem
                    id_map[stem] = entry

        summaries = []
        extraction_records = []
        for md_path in md_files:
            entry = id_map.get(md_path.stem)
            arxiv_id = entry.arxiv_id if entry else md_path.stem
            version = entry.version if entry else "v1"
            title = md_path.stem  # best-effort title from filename
            extraction = extract_paper(md_path, arxiv_id, version, title, topic_terms)
            extraction_records.append(extraction)
            base_name = f"{arxiv_id}{version}"
            (extraction_dir / f"{base_name}.extraction.json").write_text(
                extraction.model_dump_json(indent=2),
                encoding="utf-8",
            )
            (extraction_dir / f"{base_name}.extraction.md").write_text(
                render_extraction_markdown(extraction),
                encoding="utf-8",
            )
            summary = project_extraction_to_summary(extraction)
            summaries.append(summary)
            (summarize_dir / f"{base_name}.summary.json").write_text(
                summary.model_dump_json(indent=2),
                encoding="utf-8",
            )

        structured = synthesize_extractions(extraction_records, topic)
        synthesis = project_structured_synthesis_to_report(structured, summaries)
        structured_md = render_structured_synthesis_markdown(structured)
        synthesis_json = summarize_dir / "synthesis.json"
        structured_json = summarize_dir / "synthesis_report.json"
        structured_md_path = summarize_dir / "synthesis_report.md"
        synthesis_md_path = summarize_dir / "synthesis.md"
        traceability_path = summarize_dir / "synthesis_traceability.json"
        quality_path = summarize_dir / "synthesis_quality.json"
        synthesis_json.write_text(synthesis.model_dump_json(indent=2), encoding="utf-8")
        structured_json.write_text(
            structured.model_dump_json(indent=2),
            encoding="utf-8",
        )
        structured_md_path.write_text(structured_md, encoding="utf-8")
        synthesis_md_path.write_text(structured_md, encoding="utf-8")
        traceability_path.write_text(
            json.dumps(structured.traceability_appendix, indent=2),
            encoding="utf-8",
        )
        quality_path.write_text(
            structured.quality.model_dump_json(indent=2),
            encoding="utf-8",
        )

        logger.info("Summarized %d papers, synthesis written", len(summaries))
        return ToolResult(
            success=True,
            message=f"Summarized {len(summaries)} papers with cross-paper synthesis.",
            artifacts={
                "summarize_dir": str(summarize_dir),
                "extraction_dir": str(extraction_dir),
                "summary_count": len(summaries),
                "synthesis": str(synthesis_json),
                "structured_synthesis": str(structured_json),
                "synthesis_report": str(structured_md_path),
                "traceability": str(traceability_path),
                "quality": str(quality_path),
            },
        )
    except Exception as exc:
        logger.error("summarize_papers failed: %s", exc)
        return ToolResult(success=False, message=f"Failed: {exc}")


def run_pipeline(params: RunPipelineInput, ctx: Context | None = None) -> ToolResult:
    """Run the full pipeline end-to-end."""
    try:
        from research_pipeline.pipeline.orchestrator import run_pipeline as _run

        ws = _resolve_workspace(params.workspace)
        rid = _resolve_run_id(params.run_id)

        manifest = _run(
            topic=params.topic,
            run_id=rid,
            resume=params.resume,
            workspace=ws,
        )

        logger.info("Pipeline completed: run_id=%s", rid)
        return ToolResult(
            success=True,
            message=f"Pipeline completed for topic: {params.topic}",
            artifacts={
                "run_id": rid,
                "manifest": manifest.model_dump() if manifest else {},
            },
        )
    except Exception as exc:
        logger.error("run_pipeline failed: %s", exc)
        return ToolResult(success=False, message=f"Failed: {exc}")


def get_run_manifest(
    params: GetRunManifestInput, ctx: Context | None = None
) -> ToolResult:
    """Inspect a run's manifest and artifacts."""
    try:
        from research_pipeline.storage.manifests import load_manifest

        ws = _resolve_workspace(params.workspace)
        rid = params.run_id
        run_root = _get_run_root(ws, rid)

        manifest = load_manifest(run_root)
        if manifest is None:
            return ToolResult(
                success=False,
                message=f"No manifest found for run_id={rid}.",
            )

        return ToolResult(
            success=True,
            message=f"Manifest loaded for run_id={rid}.",
            artifacts={"manifest": manifest.model_dump()},
        )
    except Exception as exc:
        logger.error("get_run_manifest failed: %s", exc)
        return ToolResult(success=False, message=f"Failed: {exc}")


def convert_file(params: ConvertFileInput, ctx: Context | None = None) -> ToolResult:
    """Convert a single PDF file to Markdown (standalone, no workspace needed)."""
    try:
        from research_pipeline.config.loader import load_config
        from research_pipeline.conversion.registry import (
            _ensure_builtins_registered,
            get_backend,
        )

        _ensure_builtins_registered()

        pdf_path = Path(params.pdf_path).expanduser().resolve()
        if not pdf_path.exists():
            return ToolResult(
                success=False,
                message=f"PDF file not found: {pdf_path}",
            )

        output_dir = Path(params.output_dir) if params.output_dir else pdf_path.parent
        output_dir = output_dir.expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        config = load_config()
        backend_name = params.backend or config.conversion.backend
        kwargs = _backend_kwargs(backend_name, config)
        backend = get_backend(backend_name, **kwargs)

        result = backend.convert(pdf_path, output_dir)

        md_path = output_dir / f"{pdf_path.stem}.md"
        logger.info(
            "Converted %s → %s (status=%s, backend=%s)",
            pdf_path,
            md_path,
            result.status,
            backend_name,
        )
        return ToolResult(
            success=result.status in ("converted", "skipped_exists"),
            message=(
                f"Conversion {result.status}: {pdf_path.name} → {md_path.name} "
                f"(backend={backend_name})"
            ),
            artifacts={
                "markdown_path": str(md_path) if md_path.exists() else "",
                "status": result.status,
                "backend": backend_name,
            },
        )
    except ImportError as exc:
        return ToolResult(
            success=False,
            message=(
                f"Converter backend is not installed: {exc}. "
                "Install the corresponding extra."
            ),
        )
    except Exception as exc:
        logger.error("convert_file failed: %s", exc)
        return ToolResult(success=False, message=f"Failed: {exc}")


def list_backends(params: ListBackendsInput, ctx: Context | None = None) -> ToolResult:
    """List available converter backends."""
    try:
        from research_pipeline.conversion.registry import (
            _ensure_builtins_registered,
        )
        from research_pipeline.conversion.registry import list_backends as _list

        _ensure_builtins_registered()
        backends = _list()
        return ToolResult(
            success=True,
            message=f"Available backends: {', '.join(backends)}",
            artifacts={"backends": backends},
        )
    except Exception as exc:
        logger.error("list_backends failed: %s", exc)
        return ToolResult(success=False, message=f"Failed: {exc}")


def expand_citations(
    params: ExpandCitationsInput, ctx: Context | None = None
) -> ToolResult:
    """Expand citation graph for specified papers via Semantic Scholar."""
    try:
        import json

        from research_pipeline.config.loader import load_config
        from research_pipeline.infra.rate_limit import RateLimiter
        from research_pipeline.sources.citation_graph import CitationGraphClient
        from research_pipeline.storage.manifests import write_jsonl
        from research_pipeline.storage.workspace import get_stage_dir, init_run

        if not params.paper_ids:
            return ToolResult(success=False, message="No paper IDs provided.")

        config = load_config()
        ws = _resolve_workspace(params.workspace)
        rid = _resolve_run_id(params.run_id)
        _rid, run_root = init_run(ws, rid)

        expand_dir = get_stage_dir(run_root, "expand")
        expand_dir.mkdir(parents=True, exist_ok=True)

        s2_api_key = config.sources.semantic_scholar_api_key
        rate_limiter = RateLimiter(
            min_interval=config.sources.semantic_scholar_min_interval,
            name="s2_expand",
        )
        client = CitationGraphClient(
            api_key=s2_api_key,
            rate_limiter=rate_limiter,
        )

        if params.snowball:
            from research_pipeline.models.snowball import SnowballBudget
            from research_pipeline.sources.snowball import (
                format_snowball_report,
                snowball_expand,
            )

            budget = SnowballBudget(
                max_rounds=params.snowball_max_rounds,
                max_total_papers=params.snowball_max_papers,
                limit_per_paper=params.limit,
                direction=params.direction,
            )

            _log_info(
                ctx,
                f"Snowball expansion: {len(params.paper_ids)} seeds, "
                f"max_rounds={params.snowball_max_rounds}",
            )

            candidates, result = snowball_expand(
                client=client,
                seed_ids=params.paper_ids,
                query_terms=params.query_terms,
                budget=budget,
            )

            report_path = expand_dir / "snowball_report.md"
            report_path.write_text(format_snowball_report(result), encoding="utf-8")
            stats_path = expand_dir / "snowball_stats.json"
            stats_path.write_text(
                json.dumps(result.model_dump(mode="json"), indent=2),
                encoding="utf-8",
            )

            stop_reason = result.stop_reason.value
        else:
            _log_info(
                ctx,
                f"Expanding citations for {len(params.paper_ids)} papers "
                f"(direction={params.direction})",
            )

            candidates = client.fetch_related(
                paper_ids=params.paper_ids,
                direction=params.direction,
                limit_per_paper=params.limit,
            )
            stop_reason = "single_hop"

        _report_progress(
            ctx,
            len(params.paper_ids),
            len(params.paper_ids),
            "Expansion complete",
        )

        output_path = expand_dir / "expanded_candidates.jsonl"
        records = [c.model_dump(mode="json") for c in candidates]
        write_jsonl(records, output_path)

        logger.info("Expansion complete: %d related papers", len(candidates))
        artifacts: dict[str, object] = {
            "expanded_candidates": str(output_path),
            "run_id": _rid,
            "count": len(candidates),
            "mode": "snowball" if params.snowball else "single_hop",
        }
        if params.snowball:
            artifacts["stop_reason"] = stop_reason
            artifacts["snowball_report"] = str(expand_dir / "snowball_report.md")

        return ToolResult(
            success=True,
            message=(
                f"Expanded {len(params.paper_ids)} seed papers → "
                f"{len(candidates)} related papers"
                f" (mode={'snowball' if params.snowball else 'single_hop'}"
                f"{', stop=' + stop_reason if params.snowball else ''})."
            ),
            artifacts=artifacts,
        )
    except Exception as exc:
        logger.error("expand_citations failed: %s", exc)
        return ToolResult(success=False, message=f"Failed: {exc}")


def evaluate_quality(
    params: EvaluateQualityInput, ctx: Context | None = None
) -> ToolResult:
    """Compute composite quality scores for candidate papers."""
    try:
        from research_pipeline.config.loader import load_config
        from research_pipeline.models.candidate import CandidateRecord
        from research_pipeline.quality.composite import compute_quality_score
        from research_pipeline.storage.manifests import read_jsonl, write_jsonl
        from research_pipeline.storage.workspace import get_stage_dir, init_run

        config = load_config()
        ws = _resolve_workspace(params.workspace)
        rid = _resolve_run_id(params.run_id)
        _rid, run_root = init_run(ws, rid)

        qc = config.quality
        weights = {
            "citation_weight": qc.citation_weight,
            "venue_weight": qc.venue_weight,
            "author_weight": qc.author_weight,
            "recency_weight": qc.recency_weight,
        }

        # Try screen shortlist first, then search candidates
        screen_dir = get_stage_dir(run_root, "screen")
        search_dir = get_stage_dir(run_root, "search")

        candidates_path = screen_dir / "shortlist.jsonl"
        if not candidates_path.exists():
            candidates_path = search_dir / "candidates.jsonl"

        if not candidates_path.exists():
            return ToolResult(
                success=False,
                message="No candidates found. Run search or screen first.",
            )

        raw_records = read_jsonl(candidates_path)
        candidates = [CandidateRecord(**r) for r in raw_records]

        quality_dir = get_stage_dir(run_root, "quality")
        quality_dir.mkdir(parents=True, exist_ok=True)

        _log_info(ctx, f"Scoring quality for {len(candidates)} candidates")

        scores = []
        total = len(candidates)
        for i, candidate in enumerate(candidates):
            qs = compute_quality_score(
                candidate,
                weights=weights,
                venue_data_path=qc.venue_data_path,
            )
            scores.append(qs.model_dump(mode="json"))
            if (i + 1) % 10 == 0 or i == total - 1:
                _report_progress(ctx, i + 1, total, "Scoring papers")

        output_path = quality_dir / "quality_scores.jsonl"
        write_jsonl(scores, output_path)

        logger.info("Quality scoring complete: %d scores", len(scores))
        return ToolResult(
            success=True,
            message=f"Quality scores computed for {len(scores)} candidates.",
            artifacts={
                "quality_scores": str(output_path),
                "run_id": _rid,
                "count": len(scores),
            },
        )
    except Exception as exc:
        logger.error("evaluate_quality failed: %s", exc)
        return ToolResult(success=False, message=f"Failed: {exc}")


def convert_rough(params: ConvertRoughInput, ctx: Context | None = None) -> ToolResult:
    """Fast Tier 2 conversion of all downloaded PDFs via pymupdf4llm."""
    try:
        from research_pipeline.conversion.registry import (
            _ensure_builtins_registered,
            get_backend,
        )
        from research_pipeline.models.download import DownloadManifestEntry
        from research_pipeline.storage.manifests import read_jsonl, write_jsonl
        from research_pipeline.storage.workspace import get_stage_dir, init_run

        ws = _resolve_workspace(params.workspace)
        rid = _resolve_run_id(params.run_id)
        _rid, run_root = init_run(ws, rid)

        dl_manifest_path = (
            get_stage_dir(run_root, "download_root") / "download_manifest.jsonl"
        )
        if not dl_manifest_path.exists():
            return ToolResult(
                success=False,
                message="No download manifest found. Run download first.",
            )

        raw = read_jsonl(dl_manifest_path)
        entries = [DownloadManifestEntry.model_validate(d) for d in raw]

        _ensure_builtins_registered()
        converter = get_backend("pymupdf4llm")

        rough_dir = get_stage_dir(run_root, "convert_rough")
        rough_dir.mkdir(parents=True, exist_ok=True)

        _log_info(ctx, f"Starting rough conversion of {len(entries)} entries")

        results = []
        eligible = [e for e in entries if e.status in ("downloaded", "skipped_exists")]
        total = len(eligible)
        for i, entry in enumerate(eligible):
            pdf_path = Path(entry.local_path)
            if not pdf_path.exists():
                logger.warning("PDF not found: %s", pdf_path)
                continue
            _report_progress(ctx, i, total, f"Converting {pdf_path.name}")
            result = converter.convert(pdf_path, rough_dir, force=params.force)
            results.append(result)
        _report_progress(ctx, total, total, "Rough conversion complete")

        manifest_path = rough_dir / "convert_rough_manifest.jsonl"
        records = [r.model_dump(mode="json") for r in results]
        for rec in records:
            rec["tier"] = "rough"
        write_jsonl(manifest_path, records)

        converted = sum(1 for r in results if r.status == "converted")
        skipped = sum(1 for r in results if r.status == "skipped_exists")
        failed = sum(1 for r in results if r.status == "failed")

        logger.info(
            "Rough conversion: %d converted, %d skipped, %d failed",
            converted,
            skipped,
            failed,
        )
        return ToolResult(
            success=True,
            message=(
                f"Rough conversion: {converted} converted, "
                f"{skipped} skipped, {failed} failed."
            ),
            artifacts={
                "manifest": str(manifest_path),
                "run_id": _rid,
                "converted": converted,
                "skipped": skipped,
                "failed": failed,
            },
        )
    except Exception as exc:
        logger.error("convert_rough failed: %s", exc)
        return ToolResult(success=False, message=f"Failed: {exc}")


def convert_fine(params: ConvertFineInput, ctx: Context | None = None) -> ToolResult:
    """High-quality Tier 3 conversion of selected PDFs."""
    try:
        from research_pipeline.cli.cmd_convert import _create_converter
        from research_pipeline.config.loader import load_config
        from research_pipeline.models.download import DownloadManifestEntry
        from research_pipeline.storage.manifests import read_jsonl, write_jsonl
        from research_pipeline.storage.workspace import get_stage_dir, init_run

        if not params.paper_ids:
            return ToolResult(success=False, message="No paper IDs provided.")

        config = load_config()
        if params.backend:
            config.conversion.backend = params.backend
        ws = _resolve_workspace(params.workspace)
        rid = _resolve_run_id(params.run_id)
        _rid, run_root = init_run(ws, rid)

        dl_manifest_path = (
            get_stage_dir(run_root, "download_root") / "download_manifest.jsonl"
        )
        if not dl_manifest_path.exists():
            return ToolResult(
                success=False,
                message="No download manifest found. Run download first.",
            )

        raw = read_jsonl(dl_manifest_path)
        entries = [DownloadManifestEntry.model_validate(d) for d in raw]

        id_set = set(params.paper_ids)
        selected = [
            e
            for e in entries
            if e.arxiv_id in id_set and e.status in ("downloaded", "skipped_exists")
        ]

        if not selected:
            return ToolResult(
                success=False,
                message="No matching downloaded papers found for given IDs.",
            )

        converter = _create_converter(config)

        fine_dir = get_stage_dir(run_root, "convert_fine")
        fine_dir.mkdir(parents=True, exist_ok=True)

        _log_info(ctx, f"Starting fine conversion of {len(selected)} papers")

        results = []
        total = len(selected)
        for i, entry in enumerate(selected):
            pdf_path = Path(entry.local_path)
            if not pdf_path.exists():
                logger.warning("PDF not found: %s", pdf_path)
                continue
            _report_progress(ctx, i, total, f"Converting {pdf_path.name}")
            result = converter.convert(pdf_path, fine_dir, force=params.force)
            results.append(result)
        _report_progress(ctx, total, total, "Fine conversion complete")

        manifest_path = fine_dir / "convert_fine_manifest.jsonl"
        records = [r.model_dump(mode="json") for r in results]
        for rec in records:
            rec["tier"] = "fine"
        write_jsonl(manifest_path, records)

        converted = sum(1 for r in results if r.status == "converted")
        skipped = sum(1 for r in results if r.status == "skipped_exists")
        failed = sum(1 for r in results if r.status == "failed")

        logger.info(
            "Fine conversion: %d converted, %d skipped, %d failed",
            converted,
            skipped,
            failed,
        )
        return ToolResult(
            success=True,
            message=(
                f"Fine conversion: {converted} converted, "
                f"{skipped} skipped, {failed} failed."
            ),
            artifacts={
                "manifest": str(manifest_path),
                "run_id": _rid,
                "converted": converted,
                "skipped": skipped,
                "failed": failed,
            },
        )
    except Exception as exc:
        logger.error("convert_fine failed: %s", exc)
        return ToolResult(success=False, message=f"Failed: {exc}")


def manage_index(params: ManageIndexInput, ctx: Context | None = None) -> ToolResult:
    """Manage the global paper index for incremental runs."""
    try:
        from research_pipeline.storage.global_index import GlobalPaperIndex

        db_path_val = Path(params.db_path) if params.db_path else None
        index = GlobalPaperIndex(db_path=db_path_val)

        try:
            if params.gc:
                removed = index.garbage_collect()
                return ToolResult(
                    success=True,
                    message=f"Garbage collected {removed} stale entries.",
                    artifacts={"removed": removed},
                )

            if params.list_papers:
                papers = index.list_papers(limit=100)
                return ToolResult(
                    success=True,
                    message=f"Found {len(papers)} indexed papers.",
                    artifacts={"papers": papers, "count": len(papers)},
                )

            return ToolResult(
                success=True,
                message=(
                    "Use list_papers=true to browse or gc=true to clean stale entries."
                ),
            )
        finally:
            index.close()
    except Exception as exc:
        logger.error("manage_index failed: %s", exc)
        return ToolResult(success=False, message=f"Failed: {exc}")


def analyze_papers(
    params: AnalyzePapersInput, ctx: Context | None = None
) -> ToolResult:
    """Prepare per-paper analysis tasks or validate collected analysis results."""
    try:
        from research_pipeline.cli.cmd_analyze import (
            _discover_papers,
            _generate_prompt,
            _load_research_topic,
        )
        from research_pipeline.storage.workspace import get_stage_dir, init_run

        ws = _resolve_workspace(params.workspace)
        rid = _resolve_run_id(params.run_id)
        _rid, run_root = init_run(ws, rid)

        analysis_dir = get_stage_dir(run_root, "analysis")
        analysis_dir.mkdir(parents=True, exist_ok=True)

        if params.collect:
            json_files = sorted(analysis_dir.glob("*_analysis.json"))
            if not json_files:
                return ToolResult(
                    success=False,
                    message="No analysis JSON files found. Run paper-analyzer first.",
                )
            from research_pipeline.cli.cmd_analyze import _validate_analysis_json

            valid = 0
            total_errs = 0
            results_list = []
            for jf in json_files:
                errs = _validate_analysis_json(jf)
                results_list.append(
                    {"file": jf.name, "valid": not errs, "errors": errs}
                )
                if not errs:
                    valid += 1
                total_errs += len(errs)

            report_path = analysis_dir / "validation_report.json"
            import json as _json

            report_path.write_text(
                _json.dumps(
                    {
                        "total_files": len(json_files),
                        "valid": valid,
                        "invalid": len(json_files) - valid,
                        "total_errors": total_errs,
                        "results": results_list,
                    },
                    indent=2,
                )
            )
            return ToolResult(
                success=True,
                message=(
                    f"Validated {len(json_files)} analyses: "
                    f"{valid} valid, {len(json_files) - valid} invalid."
                ),
                artifacts={
                    "validation_report": str(report_path),
                    "valid_count": valid,
                    "invalid_count": len(json_files) - valid,
                },
            )

        papers = _discover_papers(run_root)
        if params.paper_ids:
            papers = [p for p in papers if p["arxiv_id"] in params.paper_ids]
        if not papers:
            return ToolResult(
                success=False,
                message="No converted papers found. Run convert first.",
            )

        topic = _load_research_topic(run_root)
        prompts = [_generate_prompt(p, topic, run_root) for p in papers]

        prompts_path = analysis_dir / "analysis_tasks.json"
        prompts_path.write_text(json.dumps(prompts, indent=2))

        logger.info("Prepared %d analysis tasks", len(prompts))
        return ToolResult(
            success=True,
            message=(
                f"Prepared {len(prompts)} analysis tasks for topic: '{topic}'. "
                "Launch paper-analyzer sub-agents, then call with collect=True."
            ),
            artifacts={
                "tasks_file": str(prompts_path),
                "paper_count": len(prompts),
                "run_id": _rid,
            },
        )
    except Exception as exc:
        logger.error("analyze_papers failed: %s", exc)
        return ToolResult(success=False, message=f"Failed: {exc}")


def validate_report(
    params: ValidateReportInput, ctx: Context | None = None
) -> ToolResult:
    """Validate a research report for completeness and quality."""
    try:
        from research_pipeline.cli.cmd_validate import validate_report as _validate

        report_path: Path | None = None
        if params.report_path:
            report_path = Path(params.report_path).expanduser().resolve()
        elif params.run_id:
            from research_pipeline.storage.workspace import get_stage_dir, init_run

            ws = _resolve_workspace(params.workspace)
            rid = _resolve_run_id(params.run_id) if params.run_id else ""
            _, run_root = init_run(ws, rid)
            synth_dir = get_stage_dir(run_root, "summarize")
            for candidate in [
                synth_dir / "synthesis_report.md",
                run_root / "synthesis" / "synthesis_report.md",
            ]:
                if candidate.exists():
                    report_path = candidate
                    break

        if report_path is None or not report_path.exists():
            return ToolResult(
                success=False,
                message="No report found. Provide report_path or run_id.",
            )

        result = _validate(report_path)
        verdict = result["verdict"]
        score = result["overall_score"]

        return ToolResult(
            success=verdict == "PASS",
            message=f"Report validation: {verdict} (score: {score})",
            artifacts=result,
        )
    except Exception as exc:
        logger.error("validate_report failed: %s", exc)
        return ToolResult(success=False, message=f"Failed: {exc}")


def compare_runs(params: CompareRunsInput, ctx: Context | None = None) -> ToolResult:
    """Compare two pipeline runs and produce a structured diff."""
    try:
        from research_pipeline.cli.cmd_compare import compare_runs as _compare
        from research_pipeline.storage.workspace import init_run

        ws = _resolve_workspace(params.workspace)
        _, run_root_a = init_run(ws, params.run_id_a)
        _, run_root_b = init_run(ws, params.run_id_b)

        result = _compare(run_root_a, run_root_b, params.run_id_a, params.run_id_b)

        pd = result["paper_diff"]
        ga = result["gap_analysis"]
        return ToolResult(
            success=True,
            message=(
                f"Compared {params.run_id_a} vs {params.run_id_b}: "
                f"{pd['count_a']}→{pd['count_b']} papers, "
                f"{ga['resolved_count']} gaps resolved, "
                f"{ga['new_count']} new gaps"
            ),
            artifacts=result,
        )
    except Exception as exc:
        logger.error("compare_runs failed: %s", exc)
        return ToolResult(success=False, message=f"Failed: {exc}")


def verify_stage(params: VerifyStageInput, ctx: Context | None = None) -> ToolResult:
    """Verify structural completeness of a pipeline stage output."""
    try:
        from research_pipeline.pipeline.orchestrator import verify_stage as _verify
        from research_pipeline.storage.workspace import init_run

        ws = _resolve_workspace(params.workspace)
        rid = _resolve_run_id(params.run_id)
        _, run_root = init_run(ws, rid)

        errors = _verify(run_root, params.stage)

        if errors:
            return ToolResult(
                success=False,
                message=(
                    f"Stage '{params.stage}' failed verification: "
                    f"{len(errors)} error(s)"
                ),
                artifacts={"stage": params.stage, "errors": errors},
            )
        return ToolResult(
            success=True,
            message=f"Stage '{params.stage}' verified OK.",
            artifacts={"stage": params.stage, "errors": []},
        )
    except Exception as exc:
        logger.error("verify_stage failed: %s", exc)
        return ToolResult(success=False, message=f"Failed: {exc}")


def record_feedback(params: FeedbackInput, ctx: Context | None = None) -> ToolResult:
    """Record user accept/reject feedback on screened papers.

    Accumulated feedback adjusts BM25 screening weights via ELO-style
    learning. Use --adjust to recompute weights after recording.
    """
    try:
        from research_pipeline.feedback.models import FeedbackDecision, FeedbackRecord
        from research_pipeline.feedback.store import FeedbackStore

        store = FeedbackStore()
        recorded = 0
        run_id = _resolve_run_id(params.run_id)

        for pid in params.accept:
            store.record(
                FeedbackRecord(
                    paper_id=pid,
                    run_id=run_id,
                    decision=FeedbackDecision.ACCEPT,
                    reason=params.reason,
                )
            )
            recorded += 1

        for pid in params.reject:
            store.record(
                FeedbackRecord(
                    paper_id=pid,
                    run_id=run_id,
                    decision=FeedbackDecision.REJECT,
                    reason=params.reason,
                )
            )
            recorded += 1

        result: dict = {"recorded": recorded, "run_id": run_id}

        if params.show:
            result["counts"] = store.count(run_id=run_id)
            result["all_time_counts"] = store.count()
            latest = store.get_latest_weights()
            if latest is not None:
                result["latest_weights"] = latest.to_weight_dict()

        if params.adjust:
            adjusted = store.compute_adjusted_weights()
            result["adjusted_weights"] = adjusted.to_weight_dict()
            result["feedback_count"] = adjusted.feedback_count

        store.close()
        return ToolResult(
            success=True,
            message=f"Recorded {recorded} feedback entries for run {run_id}",
            artifacts=result,
        )
    except Exception as exc:
        logger.error("record_feedback failed: %s", exc)
        return ToolResult(success=False, message=f"Failed: {exc}")


def query_eval_log(params: EvalLogInput, ctx: Context | None = None) -> ToolResult:
    """Query three-channel evaluation logs for a run.

    Channels: traces (JSONL), audit (SQLite), snapshots (filesystem).
    """
    try:
        from research_pipeline.infra.eval_logging import EvalLogger
        from research_pipeline.storage.workspace import resolve_workspace

        ws = resolve_workspace(Path(params.workspace) if params.workspace else None)
        run_root = ws / params.run_id
        if not run_root.exists():
            return ToolResult(
                success=False,
                message=f"Run not found: {params.run_id}",
            )

        eval_log = EvalLogger(run_root)
        result: dict = {"run_id": params.run_id}

        if params.channel in ("traces", "all"):
            traces = eval_log.tracer.read_traces(stage=params.stage)
            result["traces"] = traces[-params.limit :]
            result["trace_count"] = len(traces)

        if params.channel in ("audit", "all"):
            records = eval_log.audit.query(stage=params.stage, limit=params.limit)
            result["audit_records"] = records
            result["audit_total"] = eval_log.audit.count(stage=params.stage)

        if params.channel in ("snapshots", "all"):
            result["snapshots"] = eval_log.snapshots.list_snapshots()

        if params.channel == "summary":
            result["summary"] = eval_log.summary()

        eval_log.close()
        return ToolResult(
            success=True,
            message="Eval log query complete",
            artifacts=result,
        )
    except Exception as exc:
        logger.error("query_eval_log failed: %s", exc)
        return ToolResult(success=False, message=f"Failed: {exc}")


def aggregate_evidence_tool(
    params: EvidenceAggregateInput,
    ctx: Context | None = None,
) -> ToolResult:
    """Aggregate evidence from synthesis, stripping rhetoric.

    Processes synthesis report through evidence-only aggregation:
    strip rhetoric, normalize length, extract evidence pointers,
    merge duplicates, and filter by evidence requirements.
    """
    try:
        from research_pipeline.models.summary import (
            SynthesisReport,
        )
        from research_pipeline.storage.workspace import resolve_workspace
        from research_pipeline.summarization.evidence_aggregation import (
            aggregate_evidence,
            format_aggregation_text,
        )

        ws = resolve_workspace(Path(params.workspace) if params.workspace else None)
        run_root = ws / params.run_id

        _report_progress(ctx, 0, 3, "Loading synthesis report")

        # Load synthesis report
        from research_pipeline.storage.workspace import get_stage_dir

        sum_dir = get_stage_dir(run_root, "summarize")
        report_path = sum_dir / "synthesis.json"
        if not report_path.exists():
            report_path = sum_dir / "synthesis_report.json"
        if not report_path.exists():
            return ToolResult(
                success=False,
                message="No synthesis report found",
            )

        raw = json.loads(report_path.read_text(encoding="utf-8"))
        report = SynthesisReport.model_validate(raw)

        _report_progress(ctx, 1, 3, "Running evidence aggregation")

        result = aggregate_evidence(
            report,
            min_pointers=params.min_pointers,
            max_words=params.max_words,
            similarity_threshold=params.similarity_threshold,
            strip_rhetoric_enabled=params.strip_rhetoric,
        )

        _report_progress(ctx, 2, 3, "Saving results")

        # Save outputs
        agg_json = sum_dir / "evidence_aggregation.json"
        agg_json.write_text(
            result.model_dump_json(indent=2),
            encoding="utf-8",
        )
        agg_text = sum_dir / "evidence_aggregation.md"
        agg_text.write_text(
            format_aggregation_text(result),
            encoding="utf-8",
        )

        _report_progress(ctx, 3, 3, "Complete")

        if params.output_format == "json":
            content = result.model_dump()
        else:
            content = {
                "text": format_aggregation_text(result),
                "stats": result.stats.model_dump(),
            }

        return ToolResult(
            success=True,
            message=(
                f"Aggregated {result.stats.input_statements} → "
                f"{result.stats.output_statements} evidence statements"
            ),
            artifacts=content,
        )
    except Exception as exc:
        logger.error("aggregate_evidence failed: %s", exc)
        return ToolResult(success=False, message=f"Failed: {exc}")


def export_html_tool(
    params: ExportHtmlInput,
    ctx: Context | None = None,
) -> ToolResult:
    """Export synthesis report as self-contained HTML.

    Supports two modes:
    - run_id: Renders structured SynthesisReport as rich HTML.
    - markdown_file: Converts Markdown to styled HTML.
    """
    try:
        from research_pipeline.summarization.html_export import (
            render_html_from_markdown,
            render_html_report,
        )

        _report_progress(ctx, 0, 3, "Preparing HTML export")

        if params.markdown_file:
            md_path = Path(params.markdown_file)
            if not md_path.exists():
                return ToolResult(
                    success=False,
                    message=f"Markdown file not found: {md_path}",
                )
            out_path = (
                Path(params.output) if params.output else md_path.with_suffix(".html")
            )

            _report_progress(ctx, 1, 3, "Converting Markdown to HTML")
            html_str = render_html_from_markdown(md_path, out_path, title=params.title)

            _report_progress(ctx, 3, 3, "Complete")
            return ToolResult(
                success=True,
                message=f"HTML report written to {out_path}",
                artifacts={"path": str(out_path), "size_bytes": len(html_str)},
            )

        if not params.run_id:
            return ToolResult(
                success=False,
                message="Provide either run_id or markdown_file",
            )

        from research_pipeline.models.summary import (
            SynthesisReport,
        )
        from research_pipeline.storage.workspace import get_stage_dir, resolve_workspace

        ws = resolve_workspace(Path(params.workspace) if params.workspace else None)
        run_root = ws / params.run_id
        sum_dir = get_stage_dir(run_root, "summarize")
        report_path = sum_dir / "synthesis_report.json"

        if not report_path.exists():
            report_path = sum_dir / "synthesis.json"
        if not report_path.exists():
            return ToolResult(
                success=False,
                message="No synthesis report found",
            )

        _report_progress(ctx, 1, 3, "Loading synthesis report")
        raw = json.loads(report_path.read_text(encoding="utf-8"))
        report = SynthesisReport.model_validate(raw)

        out_path = (
            Path(params.output) if params.output else sum_dir / "synthesis_report.html"
        )

        _report_progress(ctx, 2, 3, "Rendering HTML")
        html_str = render_html_report(report, out_path)

        _report_progress(ctx, 3, 3, "Complete")
        return ToolResult(
            success=True,
            message=f"HTML report ({report.paper_count} papers) → {out_path}",
            artifacts={"path": str(out_path), "size_bytes": len(html_str)},
        )
    except Exception as exc:
        logger.error("export_html failed: %s", exc)
        return ToolResult(success=False, message=f"Failed: {exc}")


def model_routing_info_tool(
    params: ModelRoutingInfoInput,
    ctx: Context | None = None,
) -> ToolResult:
    """Return the current model routing configuration.

    Shows which LLM provider is assigned to each phase tier
    (mechanical, intelligent, critical_safety) and the stage→tier mapping.
    """
    try:
        from research_pipeline.config.loader import load_config
        from research_pipeline.llm.routing import (
            create_model_router,
        )

        config_path = (
            Path(params.config_path) if params.config_path else Path("config.toml")
        )
        cfg = load_config(config_path)
        router = create_model_router(cfg)

        summary = router.summary()
        stage_map = {stage: tier.value for stage, tier in router.stage_map.items()}

        return ToolResult(
            success=True,
            message=(
                f"Model routing: mechanical={summary['mechanical'] or 'none'}, "
                f"intelligent={summary['intelligent'] or 'none'}, "
                f"critical_safety={summary['critical_safety'] or 'none'}"
            ),
            artifacts={
                "provider_summary": summary,
                "stage_tier_map": stage_map,
                "routing_enabled": getattr(
                    getattr(cfg, "llm_routing", None), "enabled", False
                ),
            },
        )
    except Exception as exc:
        logger.error("model_routing_info failed: %s", exc)
        return ToolResult(success=False, message=f"Failed: {exc}")


def gate_info_tool(
    params: GateInfoInput,
    ctx: Context | None = None,
) -> ToolResult:
    """Return the current HITL gate configuration.

    Shows which stages have approval gates and whether
    gates are in auto-approve or interactive mode.
    """
    try:
        from research_pipeline.config.loader import load_config
        from research_pipeline.pipeline.gates import DEFAULT_GATE_STAGES

        config_path = (
            Path(params.config_path) if params.config_path else Path("config.toml")
        )
        cfg = load_config(config_path)
        gate_cfg = cfg.gates

        return ToolResult(
            success=True,
            message=(
                f"Gates: enabled={gate_cfg.enabled}, "
                f"auto_approve={gate_cfg.auto_approve}, "
                f"stages={gate_cfg.gate_after}"
            ),
            artifacts={
                "enabled": gate_cfg.enabled,
                "auto_approve": gate_cfg.auto_approve,
                "gate_after": gate_cfg.gate_after,
                "default_gate_stages": DEFAULT_GATE_STAGES,
            },
        )
    except Exception as exc:
        logger.error("gate_info failed: %s", exc)
        return ToolResult(success=False, message=f"Failed: {exc}")


def coherence_tool(
    params: CoherenceInput,
    ctx: Context | None = None,
) -> ToolResult:
    """Evaluate multi-session coherence across pipeline runs.

    Computes factual consistency, temporal ordering, knowledge update
    fidelity, and contradiction detection across 2+ runs.
    """
    try:
        from research_pipeline.pipeline.coherence import run_coherence

        ws = Path(params.workspace)
        report = run_coherence(
            run_ids=params.run_ids,
            workspace=ws,
        )

        return ToolResult(
            success=True,
            message=(
                f"Coherence evaluated across {len(params.run_ids)} runs: "
                f"overall={report.score.overall:.2f}, "
                f"contradictions={len(report.contradictions)}"
            ),
            artifacts=report.to_dict(),
        )
    except Exception as exc:
        logger.error("coherence evaluation failed: %s", exc)
        return ToolResult(success=False, message=f"Failed: {exc}")


def consolidation_tool(
    params: ConsolidationInput,
    ctx: Context | None = None,
) -> ToolResult:
    """Consolidate cross-run memory: compress episodes, promote rules, prune stale.

    Implements episodic → semantic consolidation following SEA/MLMF
    three-tier memory architecture.
    """
    try:
        from dataclasses import asdict

        from research_pipeline.pipeline.consolidation import run_consolidation

        ws = Path(params.workspace)
        result = run_consolidation(
            workspace=ws,
            run_ids=params.run_ids,
            capacity=params.capacity,
            threshold=params.threshold,
            min_support=params.min_support,
            dry_run=params.dry_run,
        )

        return ToolResult(
            success=True,
            message=(
                f"Consolidation complete: "
                f"{result.episodes_before}→{result.episodes_after} episodes, "
                f"{result.rules_created} new rules, "
                f"{result.entries_pruned} pruned"
            ),
            artifacts=asdict(result),
        )
    except Exception as exc:
        logger.error("consolidation failed: %s", exc)
        return ToolResult(success=False, message=f"Failed: {exc}")


def blinding_audit_tool(
    params: BlindingAuditInput,
    ctx: Context | None = None,
) -> ToolResult:
    """Run epistemic blinding audit to detect LLM prior contamination.

    Implements A/B blinding protocol from arXiv 2604.06013: scans analysis
    outputs for identifying feature references and scores contamination.
    """
    try:
        from research_pipeline.evaluation.blinding import (
            run_blinding_audit_for_workspace,
        )

        ws = Path(params.workspace)
        result = run_blinding_audit_for_workspace(
            ws,
            run_id=params.run_id or None,
            contamination_threshold=params.threshold,
            store_results=params.store_results,
        )

        return ToolResult(
            success=True,
            message=(
                f"Blinding audit complete for run {result.run_id}: "
                f"score={result.aggregate_score:.3f}, "
                f"{len(result.high_contamination_papers)} flagged papers. "
                f"{result.recommendation}"
            ),
            artifacts=result.to_dict(),
        )
    except Exception as exc:
        logger.error("blinding audit failed: %s", exc)
        return ToolResult(success=False, message=f"Failed: {exc}")


def dual_metrics_tool(
    params: DualMetricsInput,
    ctx: Context | None = None,
) -> ToolResult:
    """Evaluate pipeline runs using Pass@k + Pass[k] dual metrics.

    Implements Claw-Eval framework (arXiv 2604.06132): computes capability
    ceiling (Pass@k) and reliability floor (Pass[k]) with safety gates.
    """
    try:
        from research_pipeline.evaluation.dual_metrics import evaluate_runs

        ws = Path(params.workspace)
        run_ids = params.run_ids if params.run_ids else None
        result = evaluate_runs(
            ws,
            params.query,
            run_ids=run_ids,
            k=params.k,
            store_results=params.store_results,
        )

        return ToolResult(
            success=True,
            message=(
                f"Dual metrics for '{result.query}': "
                f"Pass@{result.k}={result.gated_pass_at_k:.3f}, "
                f"Pass[{result.k}]={result.gated_pass_bracket_k:.3f}, "
                f"gap={result.pass_at_k - result.pass_bracket_k:.3f}, "
                f"safety={result.safety_gate:.1f}, "
                f"n={result.n}, c={result.c}"
            ),
            artifacts=result.to_dict(),
        )
    except Exception as exc:
        logger.error("dual metrics failed: %s", exc)
        return ToolResult(success=False, message=f"Failed: {exc}")


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
        logger.error("CBR lookup failed: %s", exc)
        return ToolResult(success=False, message=f"Failed: {exc}")


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
        logger.error("CBR retain failed: %s", exc)
        return ToolResult(success=False, message=f"Failed: {exc}")


def kg_quality_tool(
    params: KGQualityInput,
    ctx: Context | None = None,
) -> ToolResult:
    """Evaluate knowledge graph quality across 5 dimensions.

    Uses the three-layer composable architecture (structural metrics,
    IC+EC consistency, TWCS sampling) to produce a composite score.
    """
    try:
        import sqlite3

        from research_pipeline.quality.kg_quality import (
            evaluate_kg_quality,
            sample_triples_twcs,
        )
        from research_pipeline.storage.knowledge_graph import DEFAULT_KG_PATH

        db_path = Path(params.db_path) if params.db_path else DEFAULT_KG_PATH
        if not db_path.exists():
            return ToolResult(
                success=False,
                message=f"KG database not found: {db_path}",
            )

        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row

        try:
            score = evaluate_kg_quality(conn, staleness_days=params.staleness_days)

            result: dict = score.to_dict()

            if params.sample_size > 0:
                sample = sample_triples_twcs(conn, sample_size=params.sample_size)
                result["twcs_sample"] = sample

            return ToolResult(
                success=True,
                message=(
                    f"KG quality: composite={score.composite:.4f}, "
                    f"accuracy={score.accuracy:.4f}, "
                    f"consistency={score.consistency:.4f}, "
                    f"completeness={score.completeness:.4f}"
                ),
                artifacts=result,
            )
        finally:
            conn.close()

    except Exception as exc:
        logger.error("KG quality evaluation failed: %s", exc)
        return ToolResult(success=False, message=f"Failed: {exc}")


def adaptive_stopping_tool(
    params: AdaptiveStoppingInput,
    ctx: Context | None = None,
) -> ToolResult:
    """Evaluate query-adaptive retrieval stopping criteria.

    Three strategies based on query type (HingeMem WWW '26):
    - recall: knee detection on cumulative relevance
    - precision: top-k saturation check
    - judgment: top-1 stability across batches
    Plus score plateau backstop and budget limits.
    """
    try:
        from research_pipeline.screening.adaptive_stopping import (
            BatchScores,
            QueryType,
            StoppingState,
            evaluate_stopping,
        )

        try:
            qtype = QueryType(params.query_type.lower())
        except ValueError:
            return ToolResult(
                success=False,
                message=f"Invalid query_type: {params.query_type}",
            )

        state = StoppingState(
            query_type=qtype,
            max_budget=params.max_budget,
            min_results=params.min_results,
            relevance_threshold=params.relevance_threshold,
        )
        for i, batch in enumerate(params.batch_scores):
            state.batches.append(BatchScores(i, [float(s) for s in batch]))

        decision = evaluate_stopping(state, query=params.query or None)

        return ToolResult(
            success=True,
            message=(
                f"Stopping: {'STOP' if decision.should_stop else 'CONTINUE'} "
                f"({decision.reason.value}) — {decision.details}"
            ),
            artifacts={
                "should_stop": decision.should_stop,
                "reason": decision.reason.value,
                "details": decision.details,
                "batches_processed": decision.batches_processed,
                "total_results": decision.total_results,
                "current_score": decision.current_score,
            },
        )

    except Exception as exc:
        logger.error("Adaptive stopping evaluation failed: %s", exc)
        return ToolResult(success=False, message=f"Failed: {exc}")


def confidence_layers_tool(
    params: ConfidenceLayersInput,
    ctx: Context | None = None,
) -> ToolResult:
    """Score claims through the 4-layer confidence architecture.

    L1 (fast signal) → L2 (adaptive granularity) → L3 (DINCO calibration)
    → L4 (selective verification). Based on Atomic Calibration, AGSC,
    DINCO, and LoVeC research.
    """
    try:
        from pathlib import Path

        from research_pipeline.cli.cmd_confidence_layers import (
            run_confidence_layers,
        )

        run_confidence_layers(
            config_path=Path(params.config_path) if params.config_path else None,
            workspace=Path(params.workspace) if params.workspace else None,
            run_id=params.run_id,
            l4_threshold=params.l4_threshold,
            damping=params.damping,
            calibrate=params.calibrate,
        )

        return ToolResult(
            success=True,
            message=(
                f"4-layer confidence scoring completed for run {params.run_id}. "
                f"L4 threshold={params.l4_threshold}, damping={params.damping}."
            ),
            artifacts={
                "run_id": params.run_id,
                "l4_threshold": params.l4_threshold,
                "damping": params.damping,
                "calibrate": params.calibrate,
            },
        )

    except Exception as exc:
        logger.error("Confidence layers scoring failed: %s", exc)
        return ToolResult(success=False, message=f"Failed: {exc}")


def export_bibtex_tool(
    params: ExportBibtexInput,
    ctx: Context | None = None,
) -> ToolResult:
    """Export papers from a pipeline stage as BibTeX."""
    try:
        from research_pipeline.models.screening import RelevanceDecision
        from research_pipeline.storage.workspace import get_stage_dir
        from research_pipeline.summarization.bibtex_export import (
            export_candidates_bibtex,
            load_candidates_from_jsonl,
        )

        _report_progress(ctx, 0, 3, "Loading candidates")
        workspace = _resolve_workspace(params.workspace)
        run_root = _get_run_root(workspace, params.run_id)
        stage_dir = get_stage_dir(run_root, params.stage)

        shortlist_path = stage_dir / "shortlist.json"
        if params.stage == "screen" and shortlist_path.exists():
            raw = json.loads(shortlist_path.read_text(encoding="utf-8"))
            decisions = [RelevanceDecision.model_validate(item) for item in raw]
            candidates = [decision.paper for decision in decisions]
        else:
            jsonl_candidates = [
                f for f in stage_dir.glob("*.jsonl") if f.stem.startswith("candidates")
            ]
            if not jsonl_candidates:
                jsonl_candidates = list(stage_dir.glob("*.jsonl"))
            if not jsonl_candidates:
                return ToolResult(
                    success=False,
                    message=f"No candidate JSONL files in {stage_dir}.",
                )

            jsonl_path = sorted(jsonl_candidates)[-1]
            _report_progress(ctx, 1, 3, "Loading candidates")
            candidates = load_candidates_from_jsonl(jsonl_path)
        if not candidates:
            return ToolResult(
                success=False,
                message=f"No candidates found in {stage_dir}.",
            )

        out_path = (
            Path(params.output) if params.output else stage_dir / "references.bib"
        )

        _report_progress(ctx, 2, 3, "Exporting BibTeX")
        count = export_candidates_bibtex(candidates, out_path)

        _report_progress(ctx, 3, 3, "Complete")
        return ToolResult(
            success=True,
            message=f"Exported {count} BibTeX entries to {out_path}.",
            artifacts={
                "run_id": params.run_id,
                "stage": params.stage,
                "path": str(out_path),
                "count": count,
            },
        )
    except Exception as exc:
        logger.error("export_bibtex failed: %s", exc)
        return ToolResult(success=False, message=f"Failed: {exc}")


def report_tool(
    params: ReportInput,
    ctx: Context | None = None,
) -> ToolResult:
    """Render a synthesis report using a configurable template."""
    try:
        from research_pipeline.models.summary import (
            CrossPaperSynthesisRecord,
            SynthesisReport,
        )
        from research_pipeline.storage.workspace import get_stage_dir
        from research_pipeline.summarization.report_templates import (
            list_templates,
            render_report_to_file,
        )

        _report_progress(ctx, 0, 4, "Validating template")
        available = list_templates()
        template_name = params.template
        if (
            template_name not in available
            and template_name != "structured_synthesis"
            and not params.custom_template
        ):
            return ToolResult(
                success=False,
                message=(
                    f"Unknown template {template_name!r}. "
                    f"Available: {', '.join(available)}"
                ),
            )

        workspace = _resolve_workspace(params.workspace)
        run_root = _get_run_root(workspace, params.run_id)
        stage_dir = get_stage_dir(run_root, "summarize")

        structured_json = stage_dir / "synthesis_report.json"
        legacy_json = stage_dir / "synthesis.json"
        candidates = (
            [structured_json, legacy_json]
            if template_name == "structured_synthesis"
            else [legacy_json, structured_json]
        )
        synthesis_json = next((path for path in candidates if path.exists()), None)
        if synthesis_json is None:
            return ToolResult(
                success=False,
                message=f"No synthesis_report.json or synthesis.json in {stage_dir}.",
            )

        _report_progress(ctx, 1, 4, "Loading synthesis")
        data = json.loads(synthesis_json.read_text(encoding="utf-8"))
        if "report" in data and "topic" in data["report"]:
            data = data["report"]
        if "corpus" in data and "taxonomy" in data:
            report: SynthesisReport | CrossPaperSynthesisRecord = (
                CrossPaperSynthesisRecord.model_validate(data)
            )
            if template_name != "structured_synthesis" and not params.custom_template:
                template_name = "structured_synthesis"
        else:
            report = SynthesisReport.model_validate(data)

        custom_tmpl: str | None = None
        if params.custom_template:
            tmpl_path = Path(params.custom_template)
            if not tmpl_path.exists():
                return ToolResult(
                    success=False,
                    message=f"Custom template not found: {tmpl_path}",
                )
            custom_tmpl = tmpl_path.read_text(encoding="utf-8")

        out_path = (
            Path(params.output)
            if params.output
            else stage_dir / f"report_{template_name}.md"
        )

        _report_progress(ctx, 2, 4, "Rendering report")
        render_report_to_file(
            report,
            out_path,
            template_name=template_name,
            custom_template=custom_tmpl,
        )

        _report_progress(ctx, 4, 4, "Complete")
        return ToolResult(
            success=True,
            message=f"Report ({template_name}) written to {out_path}.",
            artifacts={
                "run_id": params.run_id,
                "template": template_name,
                "path": str(out_path),
            },
        )
    except Exception as exc:
        logger.error("report failed: %s", exc)
        return ToolResult(success=False, message=f"Failed: {exc}")


def cluster_tool(
    params: ClusterInput,
    ctx: Context | None = None,
) -> ToolResult:
    """Cluster papers by topic similarity using TF-IDF."""
    try:
        from research_pipeline.screening.clustering import cluster_candidates
        from research_pipeline.storage.workspace import get_stage_dir
        from research_pipeline.summarization.bibtex_export import (
            load_candidates_from_jsonl,
        )

        _report_progress(ctx, 0, 3, "Loading candidates")
        workspace = _resolve_workspace(params.workspace)
        run_root = _get_run_root(workspace, params.run_id)
        stage_dir = get_stage_dir(run_root, params.stage)

        jsonl_candidates = sorted(stage_dir.glob("*.jsonl"))
        if not jsonl_candidates:
            return ToolResult(
                success=False,
                message=f"No candidate JSONL files in {stage_dir}.",
            )

        jsonl_path = jsonl_candidates[-1]
        candidates = load_candidates_from_jsonl(jsonl_path)
        if not candidates:
            return ToolResult(
                success=False,
                message=f"No candidates found in {jsonl_path}.",
            )

        _report_progress(ctx, 1, 3, "Clustering")
        clusters = cluster_candidates(candidates, threshold=params.threshold)

        result_data = {
            "run_id": params.run_id,
            "threshold": params.threshold,
            "num_papers": len(candidates),
            "num_clusters": len(clusters),
            "clusters": [
                {
                    "cluster_id": c.cluster_id,
                    "label": c.label,
                    "paper_count": len(c.paper_ids),
                    "paper_ids": c.paper_ids,
                    "top_terms": c.top_terms,
                }
                for c in clusters
            ],
        }

        out_path = Path(params.output) if params.output else stage_dir / "clusters.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(result_data, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        _report_progress(ctx, 3, 3, "Complete")
        return ToolResult(
            success=True,
            message=(
                f"Clustered {len(candidates)} papers into "
                f"{len(clusters)} groups → {out_path}."
            ),
            artifacts={
                "run_id": params.run_id,
                "path": str(out_path),
                "num_papers": len(candidates),
                "num_clusters": len(clusters),
            },
        )
    except Exception as exc:
        logger.error("cluster failed: %s", exc)
        return ToolResult(success=False, message=f"Failed: {exc}")


def enrich_tool(
    params: EnrichInput,
    ctx: Context | None = None,
) -> ToolResult:
    """Enrich candidates with missing abstracts/metadata from Semantic Scholar."""
    try:
        from research_pipeline.config.loader import load_config
        from research_pipeline.models.candidate import CandidateRecord
        from research_pipeline.sources.enrichment import enrich_candidates
        from research_pipeline.storage.manifests import read_jsonl, write_jsonl
        from research_pipeline.storage.workspace import get_stage_dir

        _report_progress(ctx, 0, 3, "Loading candidates")
        config = load_config(
            Path(params.config_path) if params.config_path else None,
        )
        workspace = _resolve_workspace(params.workspace)
        run_dir = _get_run_root(workspace, params.run_id)

        if params.stage == "screened":
            stage_dir = get_stage_dir(run_dir, "screen")
            jsonl_file = stage_dir / "screened.jsonl"
        else:
            stage_dir = get_stage_dir(run_dir, "search")
            jsonl_file = stage_dir / "candidates.jsonl"

        if not jsonl_file.exists():
            return ToolResult(
                success=False,
                message=f"Candidates file not found: {jsonl_file}",
            )

        records = read_jsonl(jsonl_file, CandidateRecord)
        missing_before = sum(1 for r in records if not r.abstract)

        _report_progress(ctx, 1, 3, "Enriching via Semantic Scholar")
        s2_api_key = getattr(config, "semantic_scholar_api_key", "") or ""
        enriched_count = enrich_candidates(records, s2_api_key=s2_api_key)

        _report_progress(ctx, 2, 3, "Writing results")
        output_file = stage_dir / f"{jsonl_file.stem}_enriched.jsonl"
        write_jsonl(output_file, records)

        summary = {
            "total_candidates": len(records),
            "enriched_count": enriched_count,
            "missing_abstracts_before": missing_before,
            "missing_abstracts_after": sum(1 for r in records if not r.abstract),
        }
        summary_file = stage_dir / "enrichment_summary.json"
        summary_file.write_text(json.dumps(summary, indent=2))

        _report_progress(ctx, 3, 3, "Complete")
        return ToolResult(
            success=True,
            message=(
                f"Enriched {enriched_count}/{len(records)} candidates. "
                f"Output: {output_file}"
            ),
            artifacts={
                "run_id": params.run_id,
                "enriched_count": enriched_count,
                "total": len(records),
                "output_path": str(output_file),
                "summary_path": str(summary_file),
            },
        )
    except Exception as exc:
        logger.error("enrich failed: %s", exc)
        return ToolResult(success=False, message=f"Failed: {exc}")


def cite_context_tool(
    params: CiteContextInput,
    ctx: Context | None = None,
) -> ToolResult:
    """Extract citation contexts from converted Markdown papers."""
    try:
        from research_pipeline.extraction.citation_context import (
            contexts_to_dicts,
            extract_citation_contexts,
        )
        from research_pipeline.storage.workspace import get_stage_dir

        _report_progress(ctx, 0, 3, "Finding Markdown files")
        workspace = _resolve_workspace(params.workspace)
        run_dir = _get_run_root(workspace, params.run_id)

        convert_dir = get_stage_dir(run_dir, "convert")
        md_files = sorted(convert_dir.glob("**/*.md"))
        if not md_files:
            return ToolResult(
                success=False,
                message=f"No Markdown files in {convert_dir}.",
            )

        _report_progress(ctx, 1, 3, "Extracting citation contexts")
        all_contexts: dict[str, list[dict[str, object]]] = {}
        total_count = 0
        for md_file in md_files:
            text = md_file.read_text(encoding="utf-8")
            contexts = extract_citation_contexts(
                text,
                context_window=params.window,
            )
            if contexts:
                paper_key = md_file.stem
                all_contexts[paper_key] = contexts_to_dicts(contexts)
                total_count += len(contexts)

        output_path = (
            Path(params.output)
            if params.output
            else convert_dir / "citation_contexts.json"
        )
        output_path.write_text(
            json.dumps(all_contexts, indent=2, ensure_ascii=False),
        )

        _report_progress(ctx, 3, 3, "Complete")
        return ToolResult(
            success=True,
            message=(
                f"Extracted {total_count} citation contexts from "
                f"{len(all_contexts)}/{len(md_files)} papers → {output_path}."
            ),
            artifacts={
                "run_id": params.run_id,
                "total_contexts": total_count,
                "papers_with_contexts": len(all_contexts),
                "total_papers": len(md_files),
                "path": str(output_path),
            },
        )
    except Exception as exc:
        logger.error("cite_context failed: %s", exc)
        return ToolResult(success=False, message=f"Failed: {exc}")


def watch_tool(
    params: WatchInput,
    ctx: Context | None = None,
) -> ToolResult:
    """Check for new papers matching saved watch queries on arXiv."""
    try:
        from research_pipeline.arxiv.client import ArxivClient
        from research_pipeline.arxiv.rate_limit import ArxivRateLimiter
        from research_pipeline.cli.cmd_watch import (
            DEFAULT_QUERIES_FILE,
            _load_queries,
            _load_watch_state,
            _save_watch_state,
        )
        from research_pipeline.infra.http import create_session

        queries_path = Path(params.queries) if params.queries else DEFAULT_QUERIES_FILE
        _report_progress(ctx, 0, 3, "Loading watch queries")
        queries = _load_queries(queries_path)
        if not queries:
            return ToolResult(
                success=False,
                message=(
                    f"No queries found. Create {queries_path} with watch queries."
                ),
            )

        state_path = queries_path.parent / "watch_state.json"
        state = _load_watch_state(state_path)

        _report_progress(ctx, 1, 3, "Checking arXiv for new papers")
        session = create_session()
        rate_limiter = ArxivRateLimiter()
        client = ArxivClient(session=session, rate_limiter=rate_limiter)

        from datetime import UTC, datetime, timedelta

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
                last_checked = now - timedelta(days=params.lookback)

            try:
                results = client.search(
                    query=query_text,
                    max_results=params.max_results,
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

            state[name] = now.isoformat()

        _save_watch_state(state_path, state)

        if params.output and all_new_papers:
            out = Path(params.output)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(
                json.dumps(all_new_papers, indent=2, ensure_ascii=False),
            )

        _report_progress(ctx, 3, 3, "Complete")
        return ToolResult(
            success=True,
            message=(
                f"Watch complete: {total_new} new papers across {len(queries)} queries."
            ),
            artifacts={
                "total_new": total_new,
                "queries_checked": len(queries),
                "papers": all_new_papers,
            },
        )
    except Exception as exc:
        logger.error("watch failed: %s", exc)
        return ToolResult(success=False, message=f"Failed: {exc}")


def analyze_claims_tool(
    params: AnalyzeClaimsInput, ctx: Context | None = None
) -> ToolResult:
    """Decompose paper summaries into atomic claims with evidence classification."""
    try:
        from research_pipeline.analysis.decomposer import decompose_paper
        from research_pipeline.config.loader import load_config
        from research_pipeline.models.summary import PaperSummary
        from research_pipeline.storage.manifests import read_jsonl, write_jsonl
        from research_pipeline.storage.workspace import get_stage_dir, init_run

        config = load_config()
        ws = Path(params.workspace) if params.workspace else Path(config.workspace)
        run_id_str, run_root = init_run(ws, params.run_id or None)

        _report_progress(ctx, 0, 3, "Loading summaries")

        summary_dir = get_stage_dir(run_root, "summarize")
        summary_path = summary_dir / "paper_summaries.jsonl"
        if summary_path.exists():
            raw = read_jsonl(summary_path)
        else:
            summary_files = list(summary_dir.glob("*.summary.json"))
            if not summary_files:
                return ToolResult(
                    success=False,
                    message="No paper summaries found. Run 'summarize' first.",
                )
            raw = []
            for sf in summary_files:
                raw.append(json.loads(sf.read_text(encoding="utf-8")))

        summaries = [PaperSummary.model_validate(d) for d in raw]
        md_dir = get_stage_dir(run_root, "convert")

        _report_progress(ctx, 1, 3, "Decomposing claims")

        results = []
        for summary in summaries:
            md_path = md_dir / f"{summary.arxiv_id}.md"
            if not md_path.exists():
                md_path = md_dir / f"{summary.arxiv_id}{summary.version}.md"

            markdown_path_str = str(md_path) if md_path.exists() else None
            decomp = decompose_paper(
                summary=summary,
                markdown_path=markdown_path_str,
            )
            results.append(decomp)

        _report_progress(ctx, 2, 3, "Writing results")

        claims_dir = summary_dir / "claims"
        claims_dir.mkdir(parents=True, exist_ok=True)
        output_path = claims_dir / "claim_decomposition.jsonl"
        write_jsonl(output_path, [r.model_dump(mode="json") for r in results])

        total_claims = sum(r.total_claims for r in results)
        total_supported = sum(r.evidence_summary.get("supported", 0) for r in results)

        _report_progress(ctx, 3, 3, "Done")
        return ToolResult(
            success=True,
            message=(
                f"Decomposed {len(results)} papers into {total_claims} claims "
                f"({total_supported} supported)."
            ),
            artifacts={
                "output": str(output_path),
                "papers": len(results),
                "total_claims": total_claims,
                "supported": total_supported,
            },
        )
    except Exception as exc:
        logger.error("analyze_claims failed: %s", exc)
        return ToolResult(success=False, message=f"Failed: {exc}")


def score_claims_tool(
    params: ScoreClaimsInput, ctx: Context | None = None
) -> ToolResult:
    """Score confidence for decomposed claims."""
    try:
        from research_pipeline.confidence.scorer import score_decomposition
        from research_pipeline.config.loader import load_config
        from research_pipeline.llm.providers import create_llm_provider
        from research_pipeline.models.claim import ClaimDecomposition
        from research_pipeline.storage.manifests import read_jsonl, write_jsonl
        from research_pipeline.storage.workspace import get_stage_dir, init_run

        config = load_config()
        ws = Path(params.workspace) if params.workspace else Path(config.workspace)
        run_id_str, run_root = init_run(ws, params.run_id or None)

        claims_dir = get_stage_dir(run_root, "summarize") / "claims"
        claims_path = claims_dir / "claim_decomposition.jsonl"
        if not claims_path.exists():
            return ToolResult(
                success=False,
                message="No claim decompositions found. Run 'analyze-claims' first.",
            )

        _report_progress(ctx, 0, 3, "Loading decompositions")
        raw = read_jsonl(claims_path)
        decompositions = [ClaimDecomposition.model_validate(d) for d in raw]

        llm_provider = create_llm_provider(config.llm)

        _report_progress(ctx, 1, 3, "Scoring claims")
        results = []
        for decomp in decompositions:
            scored = score_decomposition(decomp, llm_provider)
            results.append(scored)

        _report_progress(ctx, 2, 3, "Writing results")
        output_path = claims_dir / "scored_claims.jsonl"
        write_jsonl(output_path, [r.model_dump(mode="json") for r in results])

        total_claims = sum(len(r.claims) for r in results)
        avg_confidence = sum(
            c.confidence_score for r in results for c in r.claims
        ) / max(total_claims, 1)

        _report_progress(ctx, 3, 3, "Done")
        return ToolResult(
            success=True,
            message=(
                f"Scored {total_claims} claims across {len(results)} papers. "
                f"Average confidence: {avg_confidence:.3f}."
            ),
            artifacts={
                "output": str(output_path),
                "papers": len(results),
                "total_claims": total_claims,
                "avg_confidence": round(avg_confidence, 3),
                "llm_available": llm_provider is not None,
            },
        )
    except Exception as exc:
        logger.error("score_claims failed: %s", exc)
        return ToolResult(success=False, message=f"Failed: {exc}")


def kg_stats_tool(params: KGStatsInput, ctx: Context | None = None) -> ToolResult:
    """Show knowledge graph statistics."""
    try:
        from research_pipeline.storage.knowledge_graph import KnowledgeGraph

        db_path = Path(params.db_path) if params.db_path else None
        kg = KnowledgeGraph(db_path=db_path)
        try:
            stats = kg.stats()
        finally:
            kg.close()

        return ToolResult(
            success=True,
            message=(
                f"KG has {stats['total_entities']} entities, "
                f"{stats['total_triples']} triples."
            ),
            artifacts=stats,
        )
    except Exception as exc:
        logger.error("kg_stats failed: %s", exc)
        return ToolResult(success=False, message=f"Failed: {exc}")


def kg_query_tool(params: KGQueryInput, ctx: Context | None = None) -> ToolResult:
    """Query an entity and its relations in the knowledge graph."""
    try:
        from research_pipeline.storage.knowledge_graph import KnowledgeGraph

        db_path = Path(params.db_path) if params.db_path else None
        kg = KnowledgeGraph(db_path=db_path)
        try:
            entity = kg.get_entity(params.entity_id)
            if entity is None:
                return ToolResult(
                    success=False,
                    message=f"Entity not found: {params.entity_id}",
                )

            neighbors = kg.get_neighbors(params.entity_id)

            entity_data = {
                "entity_id": entity.entity_id,
                "name": entity.name,
                "type": entity.entity_type.value,
                "properties": entity.properties,
            }

            relations = []
            for t in neighbors:
                direction = (
                    "outgoing" if t.subject_id == params.entity_id else "incoming"
                )
                other = (
                    t.object_id if t.subject_id == params.entity_id else t.subject_id
                )
                relations.append(
                    {
                        "direction": direction,
                        "relation": t.relation.value,
                        "other_entity": other,
                        "confidence": t.confidence,
                    }
                )
        finally:
            kg.close()

        return ToolResult(
            success=True,
            message=(
                f"Entity '{entity.name}' ({entity.entity_type.value}) "
                f"with {len(relations)} relations."
            ),
            artifacts={
                "entity": entity_data,
                "relations": relations,
            },
        )
    except Exception as exc:
        logger.error("kg_query failed: %s", exc)
        return ToolResult(success=False, message=f"Failed: {exc}")


def kg_ingest_tool(params: KGIngestInput, ctx: Context | None = None) -> ToolResult:
    """Ingest pipeline results into the knowledge graph."""
    try:
        from research_pipeline.config.loader import load_config
        from research_pipeline.models.candidate import CandidateRecord
        from research_pipeline.models.claim import ClaimDecomposition
        from research_pipeline.storage.knowledge_graph import KnowledgeGraph
        from research_pipeline.storage.manifests import read_jsonl
        from research_pipeline.storage.workspace import get_stage_dir, init_run

        config = load_config()
        ws = Path(params.workspace) if params.workspace else Path(config.workspace)
        run_id_str, run_root = init_run(ws, params.run_id or None)

        db_path = Path(params.db_path) if params.db_path else None
        kg = KnowledgeGraph(db_path=db_path)

        try:
            total = 0
            claim_papers = 0

            _report_progress(ctx, 0, 2, "Ingesting candidates")

            screen_dir = get_stage_dir(run_root, "screen")
            shortlist_path = screen_dir / "shortlist.jsonl"
            if shortlist_path.exists():
                raw = read_jsonl(shortlist_path)
                candidates = [CandidateRecord.model_validate(d) for d in raw]
                added = kg.ingest_from_candidates(candidates, run_id=run_id_str)
                total += added

            _report_progress(ctx, 1, 2, "Ingesting claims")

            claims_dir = get_stage_dir(run_root, "summarize") / "claims"
            claims_path = claims_dir / "claim_decomposition.jsonl"
            if claims_path.exists():
                raw = read_jsonl(claims_path)
                for d in raw:
                    decomp = ClaimDecomposition.model_validate(d)
                    added = kg.ingest_from_claims(decomp, run_id=run_id_str)
                    total += added
                    claim_papers += 1

            stats = kg.stats()
        finally:
            kg.close()

        _report_progress(ctx, 2, 2, "Done")
        return ToolResult(
            success=True,
            message=(
                f"Ingested {total} items. KG now has "
                f"{stats['total_entities']} entities, "
                f"{stats['total_triples']} triples."
            ),
            artifacts={
                "total_ingested": total,
                "claim_papers": claim_papers,
                "kg_entities": stats["total_entities"],
                "kg_triples": stats["total_triples"],
            },
        )
    except Exception as exc:
        logger.error("kg_ingest failed: %s", exc)
        return ToolResult(success=False, message=f"Failed: {exc}")


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
        logger.error("memory_stats failed: %s", exc)
        return ToolResult(success=False, message=f"Failed: {exc}")


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
        logger.error("memory_episodes failed: %s", exc)
        return ToolResult(success=False, message=f"Failed: {exc}")


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
        logger.error("memory_search failed: %s", exc)
        return ToolResult(success=False, message=f"Failed: {exc}")


def evaluate_tool(params: EvaluateInput, ctx: Context | None = None) -> ToolResult:
    """Evaluate pipeline outputs against their schemas."""
    try:
        from research_pipeline.evaluation.schema_eval import (
            evaluate_run,
            evaluate_stage,
        )

        ws = Path(params.workspace)
        run_root = ws / params.run_id

        if not run_root.exists():
            return ToolResult(
                success=False,
                message=f"Run not found: {run_root}",
            )

        _report_progress(ctx, 0, 2, "Evaluating")

        if params.stage:
            report = evaluate_stage(run_root, params.stage)
            reports = [report]
        else:
            reports = evaluate_run(run_root)

        _report_progress(ctx, 1, 2, "Building results")

        all_passed = all(r.passed for r in reports)
        results = []
        for r in reports:
            checks = []
            for c in r.checks:
                checks.append(
                    {
                        "name": c.name,
                        "passed": c.passed,
                        "description": c.description,
                        "details": c.details,
                        "severity": c.severity,
                    }
                )
            results.append(
                {
                    "stage": r.stage,
                    "passed": r.passed,
                    "error_count": r.error_count,
                    "warning_count": r.warning_count,
                    "checks": checks,
                }
            )

        _report_progress(ctx, 2, 2, "Done")
        verdict = "PASS" if all_passed else "FAIL"
        return ToolResult(
            success=True,
            message=f"Evaluation: {verdict} ({len(reports)} stage(s) checked).",
            artifacts={
                "verdict": verdict,
                "all_passed": all_passed,
                "stages": results,
            },
        )
    except Exception as exc:
        logger.error("evaluate failed: %s", exc)
        return ToolResult(success=False, message=f"Failed: {exc}")
