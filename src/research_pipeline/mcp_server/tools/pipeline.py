from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from research_pipeline.mcp_server.schemas import (
    CompareRunsInput,
    ComputeSemanticScoresInput,
    DownloadPdfsInput,
    EvalLogInput,
    ExpandCitationsInput,
    ExtractContentInput,
    FeedbackInput,
    GetRunManifestInput,
    ManageIndexInput,
    PlanTopicInput,
    RunPipelineInput,
    ScreenCandidatesInput,
    SearchInput,
    SummarizePapersInput,
    ToolResult,
    VerifyStageInput,
)
from research_pipeline.mcp_server.tools._common import (
    _get_run_root,
    _load_id_map,
    _log_info,
    _raise_tool_error,
    _report_progress,
    _resolve_latest_run_id,
    _resolve_run_id,
    _resolve_workspace,
    _sanitize_candidates,
    logger,
)

if TYPE_CHECKING:
    from mcp.server.fastmcp import Context


def plan_topic(params: PlanTopicInput, ctx: Context | None = None) -> ToolResult:
    """Create a query plan from a natural language topic."""
    try:
        from research_pipeline.config.loader import load_config
        from research_pipeline.storage.workspace import init_run

        ws = _resolve_workspace(params.workspace)
        if params.config_path and params.workspace == "./workspace":
            ws = Path(load_config(Path(params.config_path)).workspace)
        rid = _resolve_run_id(params.run_id)
        config = load_config(Path(params.config_path) if params.config_path else None)

        rid, run_root = init_run(ws, rid)
        from research_pipeline.pipeline.query_planning import build_query_plan

        plan = build_query_plan(params.topic, config)
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
        _raise_tool_error("plan_topic", exc)


def search(params: SearchInput, ctx: Context | None = None) -> ToolResult:
    """Search sources with the same coverage and failure semantics as the CLI."""
    try:
        from research_pipeline.config.loader import load_config
        from research_pipeline.models.query_plan import QueryPlan
        from research_pipeline.pipeline.source_search import (
            _resolve_sources,
            execute_search,
            load_search_report,
        )
        from research_pipeline.storage.workspace import get_stage_dir

        config = load_config(Path(params.config_path) if params.config_path else None)
        ws = _resolve_workspace(params.workspace)
        if params.config_path and params.workspace == "./workspace":
            ws = Path(load_config(Path(params.config_path)).workspace)
        rid = _resolve_run_id(params.run_id)
        run_root = _get_run_root(ws, rid)
        stage_dir = get_stage_dir(run_root, "search")
        candidates_path = stage_dir / "candidates.jsonl"
        coverage_path = stage_dir / "source_coverage.json"
        if params.resume:
            saved = load_search_report(stage_dir)
            if saved is not None:
                return ToolResult(
                    success=saved.status != "failed",
                    message=(
                        f"Resumed: {len(saved.candidates)} candidates; "
                        f"status={saved.status}."
                    ),
                    artifacts={
                        "candidates": str(candidates_path),
                        "count": len(saved.candidates),
                        "status": saved.status,
                        "source_coverage": str(coverage_path),
                        "sources": saved.sources,
                    },
                )
        plan_path = get_stage_dir(run_root, "plan") / "query_plan.json"
        if plan_path.exists():
            plan = QueryPlan.model_validate_json(plan_path.read_text())
        elif params.topic:
            from research_pipeline.pipeline.query_planning import build_query_plan

            plan = build_query_plan(params.topic, config)
            plan_path.write_text(plan.model_dump_json(indent=2))
        else:
            return ToolResult(
                success=False, message="No query plan found and no topic provided."
            )

        sources = _resolve_sources(params.source, config.sources.enabled)
        report = execute_search(plan, config, sources, stage_dir)
        _sanitize_candidates(report.candidates)
        stage_dir.mkdir(parents=True, exist_ok=True)
        candidates_path.write_text(
            "".join(
                candidate.model_dump_json() + "\n" for candidate in report.candidates
            )
        )
        coverage = report.model_dump(mode="json", exclude={"candidates"})
        coverage["candidate_count"] = len(report.candidates)
        coverage_path.write_text(json.dumps(coverage, indent=2))
        _report_progress(ctx, len(sources), len(sources), "Search coverage recorded")
        return ToolResult(
            success=report.status != "failed",
            message=(
                f"Found {len(report.candidates)} candidates; status={report.status}."
            ),
            artifacts={
                "candidates": str(candidates_path),
                "count": len(report.candidates),
                "source_coverage": str(coverage_path),
                "status": report.status,
                "sources": {
                    name: sum(
                        a.candidate_count for a in report.attempts if a.source == name
                    )
                    for name in sources
                },
            },
        )
    except Exception as exc:
        _raise_tool_error("search", exc)


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
                message="No candidates found. Run tool_search first.",
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
        from research_pipeline.models.screening import RelevanceDecision

        shortlist_data = [
            RelevanceDecision(
                paper=c,
                cheap=score,
                final_score=score.cheap_score,
                download=True,
                download_reason="score_threshold",
            ).model_dump(mode="json")
            for c, score in shortlist[: config.screen.download_top_n]
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
        _raise_tool_error("screen_candidates", exc)


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
                message="No shortlist found. Run tool_screen_candidates first.",
            )

        download_dir = get_stage_dir(run_root, "download")
        download_dir.mkdir(parents=True, exist_ok=True)

        shortlist = json.loads(shortlist_path.read_text())
        papers = [
            entry.get("paper", entry.get("candidate", entry))
            for entry in shortlist
            if entry.get("download", True)
        ]
        session = create_session(
            config.contact_email, config.arxiv.min_interval_seconds
        )
        rate_limiter = ArxivRateLimiter(config.arxiv.min_interval_seconds)

        results = download_batch(
            papers,
            download_dir,
            session,
            rate_limiter,
            max_downloads=config.download.max_per_run,
        )

        manifest_path = (
            get_stage_dir(run_root, "download_root") / "download_manifest.jsonl"
        )
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
        _raise_tool_error("download_pdfs", exc)


def extract_content(
    params: ExtractContentInput, ctx: Context | None = None
) -> ToolResult:
    """Extract structured content from converted Markdown."""
    try:
        from research_pipeline.extraction.extractor import extract_from_markdown
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
                message="No Markdown files found. Run tool_convert_pdfs first.",
            )

        # Load download manifest to get arxiv_id/version per file
        manifest_path = (
            get_stage_dir(run_root, "download_root") / "download_manifest.jsonl"
        )
        id_map = _load_id_map(manifest_path)

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
        _raise_tool_error("extract_content", exc)


def summarize_papers(
    params: SummarizePapersInput, ctx: Context | None = None
) -> ToolResult:
    """Generate per-paper summaries and cross-paper synthesis."""
    try:
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
                message="No Markdown files found. Run tool_convert_pdfs first.",
            )

        # Load plan for topic terms
        plan_path = get_stage_dir(run_root, "plan") / "query_plan.json"
        plan_data = json.loads(plan_path.read_text())
        topic_terms = plan_data.get("must_terms", []) + plan_data.get("nice_terms", [])
        topic = plan_data.get("topic_raw", "")

        # Load download manifest for arxiv_id/version/title
        manifest_path = (
            get_stage_dir(run_root, "download_root") / "download_manifest.jsonl"
        )
        id_map = _load_id_map(manifest_path)

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
        _raise_tool_error("summarize_papers", exc)


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
        _raise_tool_error("run_pipeline", exc)


def get_run_manifest(
    params: GetRunManifestInput, ctx: Context | None = None
) -> ToolResult:
    """Inspect a run's manifest and artifacts."""
    try:
        from research_pipeline.storage.manifests import load_manifest

        ws = _resolve_workspace(params.workspace)
        rid = _resolve_latest_run_id(ws, params.run_id)
        if not rid:
            return ToolResult(
                success=False,
                message="No runs found in workspace; specify a run_id.",
            )
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
        _raise_tool_error("get_run_manifest", exc)


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
        _raise_tool_error("expand_citations", exc)


def compute_semantic_scores(
    params: ComputeSemanticScoresInput, ctx: Context | None = None
) -> ToolResult:
    """Compute SPECTER2 semantic similarity scores for all screened candidates.

    Embeds the topic query and each candidate's title+abstract using
    SPECTER2, then returns per-candidate cosine similarity scores.
    The agent uses these scores to decide which papers to prioritise.
    Scores are in [0, 1] (min-max normalised across the batch).
    """
    try:
        from research_pipeline.models.candidate import CandidateRecord
        from research_pipeline.screening.embedding import score_semantic
        from research_pipeline.storage.manifests import read_jsonl
        from research_pipeline.storage.workspace import get_stage_dir, init_run

        ws = _resolve_workspace(params.workspace)
        rid = _resolve_run_id(params.run_id)
        _rid, run_root = init_run(ws, rid)

        screen_dir = get_stage_dir(run_root, "screen")
        search_dir = get_stage_dir(run_root, "search")

        candidates_path = screen_dir / "shortlist.jsonl"
        if not candidates_path.exists():
            candidates_path = search_dir / "candidates.jsonl"

        if not candidates_path.exists():
            return ToolResult(
                success=False,
                message=(
                    "No candidates found. "
                    "Run tool_search or tool_screen_candidates first."
                ),
            )

        raw_records = read_jsonl(candidates_path)
        candidates = [CandidateRecord(**r) for r in raw_records]

        _log_info(ctx, f"Computing SPECTER2 scores for {len(candidates)} candidates")

        scores = score_semantic(
            params.topic,
            candidates,
            model_name=params.model_name,
            batch_size=params.batch_size,
        )

        results = [
            {"arxiv_id": c.arxiv_id, "semantic_score": s}
            for c, s in zip(candidates, scores, strict=True)
        ]

        logger.info("Semantic scoring complete: %d scores", len(results))
        return ToolResult(
            success=True,
            message=f"Semantic scores computed for {len(results)} candidates.",
            artifacts={
                "scores": results,
                "run_id": _rid,
                "count": len(results),
                "model": params.model_name,
            },
        )
    except Exception as exc:
        _raise_tool_error("compute_semantic_scores", exc)


def manage_index(params: ManageIndexInput, ctx: Context | None = None) -> ToolResult:
    """Manage the global paper index for incremental runs."""
    try:
        from research_pipeline.storage.global_index import GlobalPaperIndex

        action = params.action or (
            "gc" if params.gc else "list" if params.list_papers else ""
        )

        db_path_val = Path(params.db_path) if params.db_path else None
        index = GlobalPaperIndex(db_path=db_path_val)

        try:
            if action == "gc":
                removed = index.garbage_collect()
                return ToolResult(
                    success=True,
                    message=f"Garbage collected {removed} stale entries.",
                    artifacts={"removed": removed},
                )

            if action == "list":
                papers = index.list_papers(limit=100)
                return ToolResult(
                    success=True,
                    message=f"Found {len(papers)} indexed papers.",
                    artifacts={"papers": papers, "count": len(papers)},
                )

            return ToolResult(
                success=True,
                message=(
                    "Use action='list' to browse or action='gc' to clean stale entries."
                ),
            )
        finally:
            index.close()
    except Exception as exc:
        _raise_tool_error("manage_index", exc)


def compare_runs(params: CompareRunsInput, ctx: Context | None = None) -> ToolResult:
    """Compare two pipeline runs and produce a structured diff."""
    try:
        from research_pipeline.pipeline.compare import compare_runs as _compare
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
        _raise_tool_error("compare_runs", exc)


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
        _raise_tool_error("verify_stage", exc)


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

        result: dict[str, Any] = {"recorded": recorded, "run_id": run_id}

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
        _raise_tool_error("record_feedback", exc)


def query_eval_log(params: EvalLogInput, ctx: Context | None = None) -> ToolResult:
    """Query three-channel evaluation logs for a run.

    Channels: traces (JSONL), audit (SQLite), snapshots (filesystem).
    """
    try:
        from research_pipeline.infra.eval_logging import EvalLogger

        # Use the shared workspace resolver like every other tool — the previous
        # `storage.workspace.resolve_workspace` import did not exist, so this tool
        # raised on every call (surfaced while fixing run_id resolution, #110).
        ws = _resolve_workspace(params.workspace)
        rid = _resolve_latest_run_id(ws, params.run_id)
        if not rid:
            return ToolResult(
                success=False,
                message="No runs found in workspace; specify a run_id.",
            )
        run_root = ws / rid
        if not run_root.exists():
            return ToolResult(
                success=False,
                message=f"Run not found: {rid}",
            )

        eval_log = EvalLogger(run_root)
        result: dict[str, Any] = {"run_id": rid}

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
        _raise_tool_error("query_eval_log", exc)
