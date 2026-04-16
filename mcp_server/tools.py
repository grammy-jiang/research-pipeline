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
    AnalyzePapersInput,
    BlindingAuditInput,
    CoherenceInput,
    CompareRunsInput,
    ConsolidationInput,
    ConvertFileInput,
    ConvertFineInput,
    ConvertPdfsInput,
    ConvertRoughInput,
    DownloadPdfsInput,
    EvalLogInput,
    EvaluateQualityInput,
    EvidenceAggregateInput,
    ExpandCitationsInput,
    ExportHtmlInput,
    ExtractContentInput,
    FeedbackInput,
    GateInfoInput,
    GetRunManifestInput,
    ListBackendsInput,
    ManageIndexInput,
    ModelRoutingInfoInput,
    PlanTopicInput,
    RunPipelineInput,
    ScreenCandidatesInput,
    SearchInput,
    SummarizePapersInput,
    ToolResult,
    ValidateReportInput,
    VerifyStageInput,
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
    """Search arXiv and/or Google Scholar using the query plan."""
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
            message=(f"Found {len(deduped)} unique candidates " f"({source_summary})."),
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
        from research_pipeline.summarization.per_paper import summarize_paper
        from research_pipeline.summarization.synthesis import synthesize

        ws = _resolve_workspace(params.workspace)
        rid = _resolve_run_id(params.run_id)
        run_root = _get_run_root(ws, rid)

        convert_dir = get_stage_dir(run_root, "convert")
        summarize_dir = get_stage_dir(run_root, "summarize")
        summarize_dir.mkdir(parents=True, exist_ok=True)

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
        for md_path in md_files:
            entry = id_map.get(md_path.stem)
            arxiv_id = entry.arxiv_id if entry else md_path.stem
            version = entry.version if entry else "v1"
            title = md_path.stem  # best-effort title from filename
            summary = summarize_paper(md_path, arxiv_id, version, title, topic_terms)
            summaries.append(summary)

        synthesis = synthesize(summaries, topic)

        logger.info("Summarized %d papers, synthesis written", len(summaries))
        return ToolResult(
            success=True,
            message=f"Summarized {len(summaries)} papers with cross-paper synthesis.",
            artifacts={
                "summarize_dir": str(summarize_dir),
                "summary_count": len(summaries),
                "synthesis": str(synthesis) if synthesis else "",
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
                    "Use list_papers=true to browse or gc=true to "
                    "clean stale entries."
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
        from research_pipeline.models.summary import SynthesisReport
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
        report_path = sum_dir / "synthesis_report.json"
        if not report_path.exists():
            # Fall back to synthesis.json
            report_path = sum_dir / "synthesis.json"
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

        from research_pipeline.models.summary import SynthesisReport
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
