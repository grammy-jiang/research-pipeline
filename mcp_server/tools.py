"""MCP tool implementations that wrap core pipeline services.

Each function accepts a typed input schema and returns a ToolResult.
These are pure adapter functions — no business logic.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from mcp_server.schemas import (
    ConvertFileInput,
    ConvertPdfsInput,
    DownloadPdfsInput,
    ExtractContentInput,
    GetRunManifestInput,
    PlanTopicInput,
    RunPipelineInput,
    ScreenCandidatesInput,
    SearchArxivInput,
    SummarizePapersInput,
    ToolResult,
)

logger = logging.getLogger(__name__)


def _resolve_workspace(workspace: str) -> Path:
    """Resolve workspace path."""
    return Path(workspace).expanduser().resolve()


def _resolve_run_id(run_id: str) -> str:
    """Generate a run ID if not provided."""
    if run_id:
        return run_id
    from arxiv_paper_pipeline.infra.clock import utc_now

    return utc_now().strftime("%Y%m%dT%H%M%SZ")


def _get_run_root(ws: Path, rid: str) -> Path:
    """Compute the run root directory from workspace and run ID."""
    from arxiv_paper_pipeline.infra.paths import run_dir

    return run_dir(ws, rid)


def plan_topic(params: PlanTopicInput) -> ToolResult:
    """Create a query plan from a natural language topic."""
    try:
        from arxiv_paper_pipeline.config.loader import load_config
        from arxiv_paper_pipeline.models.query_plan import QueryPlan
        from arxiv_paper_pipeline.storage.workspace import init_run

        ws = _resolve_workspace(params.workspace)
        rid = _resolve_run_id(params.run_id)
        config = load_config()

        rid, run_root = init_run(ws, rid)
        plan = QueryPlan(
            topic_raw=params.topic,
            topic_normalized=params.topic.lower().strip(),
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


def search_arxiv(params: SearchArxivInput) -> ToolResult:
    """Search arXiv using the query plan."""
    try:
        from arxiv_paper_pipeline.arxiv.client import ArxivClient
        from arxiv_paper_pipeline.arxiv.dedup import dedup_across_queries
        from arxiv_paper_pipeline.arxiv.query_builder import build_query_from_plan
        from arxiv_paper_pipeline.arxiv.rate_limit import ArxivRateLimiter
        from arxiv_paper_pipeline.infra.cache import FileCache
        from arxiv_paper_pipeline.infra.http import create_session
        from arxiv_paper_pipeline.models.query_plan import QueryPlan
        from arxiv_paper_pipeline.storage.workspace import get_stage_dir

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

        # Load query plan
        plan_path = get_stage_dir(run_root, "plan") / "query_plan.json"
        if not plan_path.exists():
            return ToolResult(
                success=False,
                message="No query plan found. Run plan_topic first.",
            )

        plan_data = json.loads(plan_path.read_text())
        plan_obj = QueryPlan.model_validate(plan_data)
        queries = build_query_from_plan(plan_obj)
        rate_limiter = ArxivRateLimiter()
        session = create_session()
        cache = FileCache(ws / ".cache" / "http")
        client = ArxivClient(
            session=session,
            rate_limiter=rate_limiter,
            cache=cache,
        )

        all_candidates = []
        for query in queries:
            candidates, _raw_paths = client.search(query)
            all_candidates.extend(candidates)

        deduped = dedup_across_queries(all_candidates)

        stage_dir.mkdir(parents=True, exist_ok=True)
        with candidates_path.open("w") as fh:
            for c in deduped:
                fh.write(c.model_dump_json() + "\n")

        logger.info("Search returned %d unique candidates", len(deduped))
        return ToolResult(
            success=True,
            message=f"Found {len(deduped)} unique candidates.",
            artifacts={
                "candidates": str(candidates_path),
                "count": len(deduped),
            },
        )
    except Exception as exc:
        logger.error("search_arxiv failed: %s", exc)
        return ToolResult(success=False, message=f"Failed: {exc}")


def screen_candidates(params: ScreenCandidatesInput) -> ToolResult:
    """Two-stage relevance screening of candidates."""
    try:
        from arxiv_paper_pipeline.config.loader import load_config
        from arxiv_paper_pipeline.models.candidate import CandidateRecord
        from arxiv_paper_pipeline.screening.heuristic import (
            score_candidates,
            select_topk,
        )
        from arxiv_paper_pipeline.storage.workspace import get_stage_dir

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


def download_pdfs(params: DownloadPdfsInput) -> ToolResult:
    """Download shortlisted PDFs from arXiv."""
    try:
        from arxiv_paper_pipeline.arxiv.rate_limit import ArxivRateLimiter
        from arxiv_paper_pipeline.config.loader import load_config
        from arxiv_paper_pipeline.download.pdf import download_batch
        from arxiv_paper_pipeline.infra.http import create_session
        from arxiv_paper_pipeline.storage.workspace import get_stage_dir

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


def convert_pdfs(params: ConvertPdfsInput) -> ToolResult:
    """Convert downloaded PDFs to Markdown."""
    try:
        from arxiv_paper_pipeline.conversion.docling_backend import DoclingBackend
        from arxiv_paper_pipeline.models.download import DownloadManifestEntry
        from arxiv_paper_pipeline.storage.workspace import get_stage_dir

        ws = _resolve_workspace(params.workspace)
        rid = _resolve_run_id(params.run_id)
        run_root = _get_run_root(ws, rid)

        download_dir = get_stage_dir(run_root, "download")
        manifest_path = download_dir / "download_manifest.jsonl"
        if not manifest_path.exists():
            return ToolResult(
                success=False,
                message="No download manifest found. Run download_pdfs first.",
            )

        convert_dir = get_stage_dir(run_root, "convert")
        convert_dir.mkdir(parents=True, exist_ok=True)

        backend = DoclingBackend()
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
                result = backend.convert(pdf_path, convert_dir)
                results.append(result)

        conv_manifest = convert_dir / "convert_manifest.jsonl"
        with conv_manifest.open("w") as fh:
            for r in results:
                fh.write(r.model_dump_json() + "\n")

        success_count = sum(1 for r in results if r.status == "converted")
        logger.info("Converted %d/%d PDFs", success_count, len(results))
        return ToolResult(
            success=True,
            message=f"Converted {success_count}/{len(results)} PDFs to Markdown.",
            artifacts={
                "manifest": str(conv_manifest),
                "convert_dir": str(convert_dir),
                "converted": success_count,
            },
        )
    except ImportError:
        return ToolResult(
            success=False,
            message=(
                "Docling is not installed. "
                "Install with: pip install 'arxiv-paper-pipeline[docling]'"
            ),
        )
    except Exception as exc:
        logger.error("convert_pdfs failed: %s", exc)
        return ToolResult(success=False, message=f"Failed: {exc}")


def extract_content(params: ExtractContentInput) -> ToolResult:
    """Extract structured content from converted Markdown."""
    try:
        from arxiv_paper_pipeline.extraction.extractor import extract_from_markdown
        from arxiv_paper_pipeline.models.download import DownloadManifestEntry
        from arxiv_paper_pipeline.storage.workspace import get_stage_dir

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


def summarize_papers(params: SummarizePapersInput) -> ToolResult:
    """Generate per-paper summaries and cross-paper synthesis."""
    try:
        from arxiv_paper_pipeline.models.download import DownloadManifestEntry
        from arxiv_paper_pipeline.storage.workspace import get_stage_dir
        from arxiv_paper_pipeline.summarization.per_paper import summarize_paper
        from arxiv_paper_pipeline.summarization.synthesis import synthesize

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


def run_pipeline(params: RunPipelineInput) -> ToolResult:
    """Run the full pipeline end-to-end."""
    try:
        from arxiv_paper_pipeline.pipeline.orchestrator import run_pipeline as _run

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


def get_run_manifest(params: GetRunManifestInput) -> ToolResult:
    """Inspect a run's manifest and artifacts."""
    try:
        from arxiv_paper_pipeline.storage.manifests import load_manifest

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


def convert_file(params: ConvertFileInput) -> ToolResult:
    """Convert a single PDF file to Markdown (standalone, no workspace needed)."""
    try:
        from arxiv_paper_pipeline.conversion.docling_backend import DoclingBackend

        pdf_path = Path(params.pdf_path).expanduser().resolve()
        if not pdf_path.exists():
            return ToolResult(
                success=False,
                message=f"PDF file not found: {pdf_path}",
            )

        output_dir = Path(params.output_dir) if params.output_dir else pdf_path.parent
        output_dir = output_dir.expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        backend = DoclingBackend()
        result = backend.convert(pdf_path, output_dir)

        md_path = output_dir / f"{pdf_path.stem}.md"
        logger.info("Converted %s → %s (status=%s)", pdf_path, md_path, result.status)
        return ToolResult(
            success=result.status in ("converted", "skipped_exists"),
            message=f"Conversion {result.status}: {pdf_path.name} → {md_path.name}",
            artifacts={
                "markdown_path": str(md_path) if md_path.exists() else "",
                "status": result.status,
            },
        )
    except ImportError:
        return ToolResult(
            success=False,
            message=(
                "Docling is not installed. "
                "Install with: pip install 'arxiv-paper-pipeline[docling]'"
            ),
        )
    except Exception as exc:
        logger.error("convert_file failed: %s", exc)
        return ToolResult(success=False, message=f"Failed: {exc}")
