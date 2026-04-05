"""Pipeline orchestrator: end-to-end stage sequencing with resume support."""

import json
import logging
from pathlib import Path

from research_pipeline import __version__
from research_pipeline.arxiv.client import ArxivClient
from research_pipeline.arxiv.dedup import dedup_across_queries
from research_pipeline.arxiv.query_builder import build_query_from_plan
from research_pipeline.arxiv.rate_limit import ArxivRateLimiter
from research_pipeline.config.loader import load_config
from research_pipeline.config.models import PipelineConfig
from research_pipeline.conversion.registry import (
    _ensure_builtins_registered,
    get_backend,
)
from research_pipeline.download.pdf import download_batch
from research_pipeline.extraction.extractor import extract_from_markdown
from research_pipeline.infra.cache import FileCache
from research_pipeline.infra.clock import date_window, utc_now
from research_pipeline.infra.http import create_session
from research_pipeline.models.candidate import CandidateRecord
from research_pipeline.models.manifest import RunManifest, StageRecord
from research_pipeline.models.query_plan import QueryPlan
from research_pipeline.models.screening import RelevanceDecision
from research_pipeline.screening.heuristic import score_candidates, select_topk
from research_pipeline.storage.manifests import (
    load_manifest,
    save_manifest,
    update_stage,
    write_jsonl,
)
from research_pipeline.storage.workspace import get_stage_dir, init_run
from research_pipeline.summarization.per_paper import summarize_paper
from research_pipeline.summarization.synthesis import synthesize

logger = logging.getLogger(__name__)


def _create_converter(config: PipelineConfig) -> "ConverterBackend":  # noqa: F821
    """Create a converter backend from pipeline config."""
    _ensure_builtins_registered()
    backend_name = config.conversion.backend
    kwargs: dict[str, object] = {}
    if backend_name == "docling":
        kwargs["timeout_seconds"] = config.conversion.timeout_seconds
    elif backend_name == "marker":
        mc = config.conversion.marker
        kwargs["force_ocr"] = mc.force_ocr
        if mc.use_llm:
            kwargs["use_llm"] = True
            if mc.llm_service:
                kwargs["llm_service"] = mc.llm_service
            if mc.llm_api_key:
                kwargs["llm_api_key"] = mc.llm_api_key
    return get_backend(backend_name, **kwargs)


def _is_stage_complete(manifest: RunManifest, stage: str) -> bool:
    """Check if a stage is already completed in the manifest."""
    record = manifest.stages.get(stage)
    return record is not None and record.status == "completed"


def _record_stage(
    manifest: RunManifest,
    stage: str,
    status: str,
    started_at: object,
    output_paths: list[str] | None = None,
    errors: list[str] | None = None,
) -> RunManifest:
    """Create a stage record and update the manifest."""
    ended = utc_now()
    started = started_at if started_at else ended
    duration = int((ended - started).total_seconds() * 1000)  # type: ignore[union-attr]

    record = StageRecord(
        stage_name=stage,
        status=status,
        started_at=started,  # type: ignore[arg-type]
        ended_at=ended,
        duration_ms=duration,
        output_paths=output_paths or [],
        errors=errors or [],
    )
    return update_stage(manifest, record)


def run_pipeline(
    topic: str,
    config: PipelineConfig | None = None,
    run_id: str | None = None,
    resume: bool = False,
    workspace: Path | None = None,
) -> RunManifest:
    """Execute the full pipeline from topic to synthesis.

    Args:
        topic: Research topic (natural language).
        config: Pipeline configuration. Loaded from file if not provided.
        run_id: Optional run ID for resume.
        resume: If True, skip completed stages.
        workspace: Workspace directory.

    Returns:
        Final RunManifest with all stage records.
    """
    if config is None:
        config = load_config()

    ws = workspace or Path(config.workspace)
    run_id, run_root = init_run(ws, run_id)

    # Load or create manifest
    manifest: RunManifest | None = None
    if resume:
        manifest = load_manifest(run_root)

    if manifest is None:
        manifest = RunManifest(
            run_id=run_id,
            created_at=utc_now(),
            package_version=__version__,
            config_snapshot=config.model_dump(),
            topic_input=topic,
        )

    # Save initial config
    config_path = run_root / "run_config.json"
    config_path.write_text(
        json.dumps(config.model_dump(), indent=2, default=str),
        encoding="utf-8",
    )

    # Setup shared resources
    rate_limiter = ArxivRateLimiter(min_interval=config.arxiv.min_interval_seconds)
    session = create_session(config.contact_email)
    cache: FileCache | None = None
    if config.cache.enabled:
        cache_dir = Path(config.cache.cache_dir).expanduser()
        cache = FileCache(cache_dir, ttl_hours=config.cache.search_snapshot_ttl_hours)

    arxiv_client = ArxivClient(
        rate_limiter=rate_limiter,
        cache=cache,
        session=session,
        base_url=config.arxiv.base_url,
        contact_email=config.contact_email,
        request_timeout=config.arxiv.request_timeout_seconds,
    )

    # --- Stage: plan ---
    if not (resume and _is_stage_complete(manifest, "plan")):
        started = utc_now()
        logger.info("Stage: plan")
        plan_dir = get_stage_dir(run_root, "plan")

        plan = QueryPlan(
            topic_raw=topic,
            topic_normalized=topic.lower().strip(),
            must_terms=topic.lower().split()[:3],
            nice_terms=topic.lower().split()[3:6],
            negative_terms=[],
            candidate_categories=[],
            query_variants=[],
        )

        plan_path = plan_dir / "query_plan.json"
        plan_path.write_text(plan.model_dump_json(indent=2), encoding="utf-8")
        manifest = _record_stage(
            manifest,
            "plan",
            "completed",
            started,
            output_paths=[str(plan_path)],
        )
        save_manifest(run_root, manifest)
    else:
        plan_dir = get_stage_dir(run_root, "plan")
        plan_data = json.loads(
            (plan_dir / "query_plan.json").read_text(encoding="utf-8")
        )
        plan = QueryPlan.model_validate(plan_data)

    # --- Stage: search ---
    if not (resume and _is_stage_complete(manifest, "search")):
        started = utc_now()
        logger.info("Stage: search")
        search_dir = get_stage_dir(run_root, "search")
        raw_dir = search_dir / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)

        enabled_sources = config.sources.enabled
        all_candidates: list[CandidateRecord] = []

        def _do_arxiv() -> list[CandidateRecord]:
            queries = build_query_from_plan(plan)
            date_from, date_to = date_window(plan.primary_months)
            arxiv_lists = []
            for q in queries:
                batch, _ = arxiv_client.search(
                    query=q,
                    max_results=config.arxiv.default_page_size,
                    date_from=date_from,
                    date_to=date_to,
                    save_raw_dir=raw_dir,
                )
                arxiv_lists.append(batch)
            result = dedup_across_queries(arxiv_lists)
            logger.info("arXiv: %d candidates", len(result))
            return result

        def _do_scholar() -> list[CandidateRecord]:
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

        # Run enabled sources in parallel
        from concurrent.futures import ThreadPoolExecutor, as_completed

        futures = {}
        with ThreadPoolExecutor(max_workers=len(enabled_sources)) as executor:
            if "arxiv" in enabled_sources:
                futures[executor.submit(_do_arxiv)] = "arxiv"
            if "scholar" in enabled_sources:
                futures[executor.submit(_do_scholar)] = "scholar"

            for future in as_completed(futures):
                source_name = futures[future]
                try:
                    all_candidates.extend(future.result())
                except Exception as exc:
                    logger.error("%s search failed: %s", source_name, exc)

        # Cross-source dedup
        from research_pipeline.sources.base import dedup_cross_source

        candidates = dedup_cross_source(all_candidates)

        # Save candidates as JSONL
        candidates_path = search_dir / "candidates.jsonl"
        write_jsonl(
            candidates_path,
            [c.model_dump(mode="json") for c in candidates],
        )

        manifest = _record_stage(
            manifest,
            "search",
            "completed",
            started,
            output_paths=[str(candidates_path)],
        )
        save_manifest(run_root, manifest)
    else:
        from research_pipeline.storage.manifests import read_jsonl

        search_dir = get_stage_dir(run_root, "search")
        raw_data = read_jsonl(search_dir / "candidates.jsonl")
        candidates = [CandidateRecord.model_validate(d) for d in raw_data]

    # --- Stage: screen ---
    if not (resume and _is_stage_complete(manifest, "screen")):
        started = utc_now()
        logger.info("Stage: screen")
        screen_dir = get_stage_dir(run_root, "screen")

        scores = score_candidates(
            candidates,
            must_terms=plan.must_terms,
            nice_terms=plan.nice_terms,
            negative_terms=plan.negative_terms,
            target_categories=plan.candidate_categories,
        )

        write_jsonl(
            screen_dir / "cheap_scores.jsonl",
            [s.model_dump(mode="json") for s in scores],
        )

        top_candidates = select_topk(
            candidates,
            scores,
            top_k=config.screen.cheap_top_k,
        )

        # Build shortlist
        shortlist = []
        for candidate, score in top_candidates[: config.screen.download_top_n]:
            decision = RelevanceDecision(
                paper=candidate,
                cheap=score,
                llm=None,
                final_score=score.cheap_score,
                download=True,
                download_reason="score_threshold",
            )
            shortlist.append(decision)

        shortlist_path = screen_dir / "shortlist.json"
        shortlist_path.write_text(
            json.dumps(
                [d.model_dump(mode="json") for d in shortlist],
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )

        manifest = _record_stage(
            manifest,
            "screen",
            "completed",
            started,
            output_paths=[str(shortlist_path)],
        )
        save_manifest(run_root, manifest)
    else:
        screen_dir = get_stage_dir(run_root, "screen")
        raw_shortlist = json.loads(
            (screen_dir / "shortlist.json").read_text(encoding="utf-8")
        )
        shortlist = [RelevanceDecision.model_validate(d) for d in raw_shortlist]

    # --- Stage: download ---
    if not (resume and _is_stage_complete(manifest, "download")):
        started = utc_now()
        logger.info("Stage: download")
        pdf_dir = get_stage_dir(run_root, "download") / "pdf"
        pdf_dir.mkdir(parents=True, exist_ok=True)

        papers_to_download = [
            {
                "arxiv_id": d.paper.arxiv_id,
                "version": d.paper.version,
                "pdf_url": d.paper.pdf_url,
            }
            for d in shortlist
            if d.download
        ]

        entries = download_batch(
            papers_to_download,
            output_dir=pdf_dir,
            session=session,
            rate_limiter=rate_limiter,
            max_downloads=config.download.max_per_run,
        )

        manifest_path = (
            get_stage_dir(run_root, "download_root") / "download_manifest.jsonl"
        )
        write_jsonl(
            manifest_path,
            [e.model_dump(mode="json") for e in entries],
        )

        manifest = _record_stage(
            manifest,
            "download",
            "completed",
            started,
            output_paths=[str(manifest_path)],
        )
        save_manifest(run_root, manifest)
    else:
        from research_pipeline.models.download import DownloadManifestEntry
        from research_pipeline.storage.manifests import read_jsonl

        dl_data = read_jsonl(
            get_stage_dir(run_root, "download_root") / "download_manifest.jsonl"
        )
        entries = [DownloadManifestEntry.model_validate(d) for d in dl_data]

    # --- Stage: convert ---
    if not (resume and _is_stage_complete(manifest, "convert")):
        started = utc_now()
        logger.info("Stage: convert")
        md_dir = get_stage_dir(run_root, "convert")
        converter = _create_converter(config)

        convert_entries = []
        for dl_entry in entries:
            if dl_entry.status not in ("downloaded", "skipped_exists"):
                continue
            pdf_path = Path(dl_entry.local_path)
            if not pdf_path.exists():
                logger.warning("PDF not found: %s", pdf_path)
                continue
            result = converter.convert(pdf_path, md_dir)
            convert_entries.append(result)

        conv_manifest_path = (
            get_stage_dir(run_root, "convert_root") / "convert_manifest.jsonl"
        )
        write_jsonl(
            conv_manifest_path,
            [e.model_dump(mode="json") for e in convert_entries],
        )

        manifest = _record_stage(
            manifest,
            "convert",
            "completed",
            started,
            output_paths=[str(conv_manifest_path)],
        )
        save_manifest(run_root, manifest)
    else:
        from research_pipeline.models.conversion import ConvertManifestEntry
        from research_pipeline.storage.manifests import read_jsonl

        conv_data = read_jsonl(
            get_stage_dir(run_root, "convert_root") / "convert_manifest.jsonl"
        )
        convert_entries = [ConvertManifestEntry.model_validate(d) for d in conv_data]

    # --- Stage: extract ---
    if not (resume and _is_stage_complete(manifest, "extract")):
        started = utc_now()
        logger.info("Stage: extract")
        extract_dir = get_stage_dir(run_root, "extract")

        extractions = []
        for conv_entry in convert_entries:
            if conv_entry.status not in ("converted", "skipped_exists"):
                continue
            md_path = Path(conv_entry.markdown_path)
            if not md_path.exists():
                logger.warning("Markdown not found: %s", md_path)
                continue
            extraction = extract_from_markdown(
                md_path, conv_entry.arxiv_id, conv_entry.version
            )
            extract_path = (
                extract_dir / f"{conv_entry.arxiv_id}{conv_entry.version}.extract.json"
            )
            extract_path.write_text(
                extraction.model_dump_json(indent=2), encoding="utf-8"
            )
            extractions.append(extraction)

        manifest = _record_stage(
            manifest,
            "extract",
            "completed",
            started,
            output_paths=[str(extract_dir)],
        )
        save_manifest(run_root, manifest)

    # --- Stage: summarize ---
    if not (resume and _is_stage_complete(manifest, "summarize")):
        started = utc_now()
        logger.info("Stage: summarize")
        sum_dir = get_stage_dir(run_root, "summarize")

        summaries = []
        for conv_entry in convert_entries:
            if conv_entry.status not in ("converted", "skipped_exists"):
                continue
            md_path = Path(conv_entry.markdown_path)
            if not md_path.exists():
                continue

            # Find title from shortlist
            paper_title = conv_entry.arxiv_id
            for d in shortlist:
                if d.paper.arxiv_id == conv_entry.arxiv_id:
                    paper_title = d.paper.title
                    break

            summary = summarize_paper(
                markdown_path=md_path,
                arxiv_id=conv_entry.arxiv_id,
                version=conv_entry.version,
                title=paper_title,
                topic_terms=plan.must_terms + plan.nice_terms,
            )
            summaries.append(summary)

            # Save per-paper summary
            sum_path = (
                sum_dir / f"{conv_entry.arxiv_id}{conv_entry.version}.summary.json"
            )
            sum_path.write_text(summary.model_dump_json(indent=2), encoding="utf-8")

        # Cross-paper synthesis
        report = synthesize(summaries, plan.topic_raw)
        synthesis_path = sum_dir / "synthesis.md"
        synthesis_json = sum_dir / "synthesis.json"
        synthesis_json.write_text(report.model_dump_json(indent=2), encoding="utf-8")

        # Generate readable Markdown synthesis
        md_lines = [
            f"# Synthesis: {report.topic}",
            "",
            f"Papers analyzed: {report.paper_count}",
            "",
        ]
        if report.open_questions:
            md_lines.append("## Open Questions")
            md_lines.append("")
            for q in report.open_questions:
                md_lines.append(f"- {q}")
            md_lines.append("")

        md_lines.append("## Paper Summaries")
        md_lines.append("")
        for ps in report.paper_summaries:
            md_lines.append(f"### {ps.title}")
            md_lines.append(f"- **ID**: {ps.arxiv_id}{ps.version}")
            md_lines.append(f"- **Objective**: {ps.objective}")
            md_lines.append(f"- **Methodology**: {ps.methodology}")
            if ps.findings:
                md_lines.append("- **Findings**:")
                for f in ps.findings:
                    md_lines.append(f"  - {f}")
            md_lines.append("")

        synthesis_path.write_text("\n".join(md_lines), encoding="utf-8")

        manifest = _record_stage(
            manifest,
            "summarize",
            "completed",
            started,
            output_paths=[str(synthesis_path), str(synthesis_json)],
        )
        save_manifest(run_root, manifest)

    logger.info("Pipeline complete: run_id=%s", run_id)
    return manifest
