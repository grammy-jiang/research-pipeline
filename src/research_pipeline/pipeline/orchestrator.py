"""Pipeline orchestrator: end-to-end stage sequencing with resume support."""

import json
import logging
from pathlib import Path

from research_pipeline import __version__
from research_pipeline.arxiv.client import ArxivClient
from research_pipeline.arxiv.dedup import dedup_across_queries
from research_pipeline.arxiv.query_builder import build_query_from_plan
from research_pipeline.arxiv.rate_limit import ArxivRateLimiter
from research_pipeline.cli.cmd_plan import (
    _generate_query_variants,
    _split_topic_terms,
)
from research_pipeline.config.loader import load_config
from research_pipeline.config.models import PipelineConfig
from research_pipeline.conversion.registry import (
    _ensure_builtins_registered,
    get_backend,
)
from research_pipeline.download.pdf import download_batch
from research_pipeline.extraction.extractor import extract_from_markdown
from research_pipeline.infra.audit import AuditLogger, EventType
from research_pipeline.infra.cache import FileCache
from research_pipeline.infra.clock import date_window, utc_now
from research_pipeline.infra.http import create_session
from research_pipeline.infra.sanitize import sanitize_text
from research_pipeline.llm.providers import create_llm_provider
from research_pipeline.models.candidate import CandidateRecord
from research_pipeline.models.manifest import RunManifest, StageRecord
from research_pipeline.models.query_plan import QueryPlan
from research_pipeline.models.screening import (
    RelevanceDecision,
    parse_shortlist_lenient,
)
from research_pipeline.models.summary import PaperSummary
from research_pipeline.pipeline.topology import (
    PipelineProfile,
    classify_query_complexity,
    get_stages,
    profile_summary,
    should_run_stage,
)
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


def _verify_plan(run_root: Path) -> list[str]:
    """Verify plan stage output is substantive.

    Returns:
        List of errors (empty if OK).
    """
    errors = []
    plan_path = get_stage_dir(run_root, "plan") / "query_plan.json"
    if not plan_path.exists():
        errors.append("query_plan.json not found")
        return errors

    try:
        data = json.loads(plan_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        errors.append(f"Cannot read query_plan.json: {exc}")
        return errors

    must = data.get("must_terms", [])
    if not must or len(must) < 1:
        errors.append("query_plan has no must_terms")

    variants = data.get("query_variants", [])
    if len(variants) < 2:
        errors.append(f"query_plan has only {len(variants)} variants (need ≥2)")

    return errors


def _verify_search(run_root: Path) -> list[str]:
    """Verify search stage output is substantive."""
    errors = []
    candidates_path = get_stage_dir(run_root, "search") / "candidates.jsonl"
    if not candidates_path.exists():
        errors.append("candidates.jsonl not found")
        return errors

    from research_pipeline.storage.manifests import read_jsonl

    records = read_jsonl(candidates_path)
    if not records:
        errors.append("candidates.jsonl is empty — no papers found")
        return errors

    # Check all records have essential fields
    missing_fields = 0
    for r in records:
        if not r.get("arxiv_id") and not r.get("title"):
            missing_fields += 1
    if missing_fields > 0:
        errors.append(f"{missing_fields} candidates missing arxiv_id or title")

    return errors


def _verify_screen(run_root: Path) -> list[str]:
    """Verify screen stage output is substantive."""
    errors = []
    screen_dir = get_stage_dir(run_root, "screen")
    shortlist_path = screen_dir / "shortlist.json"
    if not shortlist_path.exists():
        errors.append("shortlist.json not found")
        return errors

    try:
        data = json.loads(shortlist_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        errors.append(f"Cannot read shortlist.json: {exc}")
        return errors

    if not data:
        errors.append("shortlist is empty — no papers selected for download")

    return errors


def _verify_download(run_root: Path) -> list[str]:
    """Verify download stage output: PDF files exist and are non-trivial."""
    errors = []
    pdf_dir = get_stage_dir(run_root, "download")
    pdfs = list(pdf_dir.glob("*.pdf"))
    if not pdfs:
        errors.append("No PDF files found in download directory")
        return errors

    small_pdfs = [p for p in pdfs if p.stat().st_size < 10_000]
    if small_pdfs:
        errors.append(f"{len(small_pdfs)} PDF(s) smaller than 10KB (likely corrupt)")

    return errors


def _verify_convert(run_root: Path) -> list[str]:
    """Verify convert stage output: Markdown files exist and are non-empty."""
    errors = []
    md_dir = get_stage_dir(run_root, "convert")
    md_files = list(md_dir.glob("*.md"))
    if not md_files:
        errors.append("No Markdown files found in convert directory")
        return errors

    empty_mds = [m for m in md_files if m.stat().st_size < 500]
    if empty_mds:
        errors.append(
            f"{len(empty_mds)} Markdown file(s) under 500 bytes (likely empty)"
        )

    return errors


def _verify_extract(run_root: Path) -> list[str]:
    """Verify extract stage output: extraction files exist."""
    errors = []
    extract_dir = get_stage_dir(run_root, "extract")
    extract_files = list(extract_dir.glob("*.extract.json"))
    if not extract_files:
        errors.append("No extraction files found")

    return errors


def _verify_summarize(run_root: Path) -> list[str]:
    """Verify summarize stage output."""
    errors = []
    sum_dir = get_stage_dir(run_root, "summarize")
    synthesis_json = sum_dir / "synthesis.json"
    synthesis_md = sum_dir / "synthesis.md"

    if not synthesis_json.exists() and not synthesis_md.exists():
        errors.append("No synthesis output found (synthesis.json or synthesis.md)")

    return errors


STAGE_VERIFIERS: dict[str, object] = {
    "plan": _verify_plan,
    "search": _verify_search,
    "screen": _verify_screen,
    "download": _verify_download,
    "convert": _verify_convert,
    "extract": _verify_extract,
    "summarize": _verify_summarize,
}


def verify_stage(run_root: Path, stage: str) -> list[str]:
    """Run verification checks for a completed stage.

    Args:
        run_root: Root run directory.
        stage: Stage name to verify.

    Returns:
        List of verification errors (empty means OK).
    """
    verifier = STAGE_VERIFIERS.get(stage)
    if verifier is None:
        return []  # No verifier for this stage
    errors = verifier(run_root)  # type: ignore[operator]
    if errors:
        logger.warning("Verification errors for stage '%s': %s", stage, errors)
    else:
        logger.debug("Stage '%s' verified OK", stage)
    return errors


def _record_stage(
    manifest: RunManifest,
    stage: str,
    status: str,
    started_at: object,
    output_paths: list[str] | None = None,
    errors: list[str] | None = None,
    audit: AuditLogger | None = None,
    run_root: Path | None = None,
) -> RunManifest:
    """Create a stage record and update the manifest.

    When *run_root* is provided, also writes an enhanced checkpoint
    JSON with timing and artifact hashes for hash-based resume.
    """
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

    if audit is not None:
        event_type = (
            EventType.STAGE_COMPLETED
            if status == "completed"
            else EventType.STAGE_FAILED
        )
        audit.emit(
            event_type,
            stage=stage,
            run_id=manifest.run_id,
            details={
                "status": status,
                "duration_ms": duration,
                "output_paths": output_paths or [],
                "errors": errors or [],
            },
        )

    # Write enhanced checkpoint with artifact hashes
    if run_root is not None:
        from research_pipeline.pipeline.checkpoint import write_checkpoint

        artifact_paths = [Path(p) for p in (output_paths or [])]
        started_iso = (
            started.isoformat()  # type: ignore[union-attr]
            if hasattr(started, "isoformat")
            else str(started)
        )
        write_checkpoint(
            run_root=run_root,
            stage=stage,
            status=status,
            started_at=started_iso,
            output_paths=artifact_paths,
            errors=errors,
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

    # Audit trail for this run
    audit = AuditLogger(run_root)

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
    audit.emit(
        EventType.CONFIG_LOADED,
        run_id=run_id,
        details={
            "sources": config.sources.enabled,
            "backend": config.conversion.backend,
        },
    )

    # Determine pipeline profile
    profile_str = config.profile
    if profile_str == "auto":
        profile = classify_query_complexity(topic)
        logger.info("Auto-detected profile: %s", profile.value)
    else:
        try:
            profile = PipelineProfile(profile_str)
        except ValueError:
            logger.warning("Unknown profile '%s', using standard", profile_str)
            profile = PipelineProfile.STANDARD

    logger.info("Pipeline profile: %s — %s", profile.value, profile_summary(profile))
    audit.emit(
        EventType.DECISION,
        run_id=run_id,
        details={"profile": profile.value, "stages": get_stages(profile)},
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

    # LLM provider (None when disabled — all LLM features degrade gracefully)
    llm_provider = create_llm_provider(config.llm)

    # Three-tier memory (optional — failures log warning, never crash pipeline)
    memory = None
    try:
        from research_pipeline.memory.episodic import Episode
        from research_pipeline.memory.manager import MemoryManager

        memory = MemoryManager(
            working_capacity=config.memory_working_capacity,
        )
        prior = memory.prior_knowledge(topic)
        if prior["past_runs"] > 0:
            logger.info(
                "Prior knowledge: %d past runs on similar topics",
                prior["past_runs"],
            )
        audit.emit(
            EventType.DECISION,
            run_id=run_id,
            details={"prior_knowledge": prior},
        )
    except Exception as exc:
        logger.warning("Memory init failed (continuing without memory): %s", exc)
        memory = None

    # --- Stage: plan ---
    if memory:
        memory.transition_stage("plan")
    if not (resume and _is_stage_complete(manifest, "plan")):
        started = utc_now()
        logger.info("Stage: plan")
        audit.emit(EventType.STAGE_STARTED, stage="plan", run_id=run_id)
        plan_dir = get_stage_dir(run_root, "plan")

        must_terms, nice_terms = _split_topic_terms(topic)
        query_variants = _generate_query_variants(
            must_terms,
            nice_terms,
            max_variants=config.search.max_query_variants,
        )

        plan = QueryPlan(
            topic_raw=topic,
            topic_normalized=topic.lower().strip(),
            must_terms=must_terms,
            nice_terms=nice_terms,
            negative_terms=[],
            candidate_categories=[],
            query_variants=query_variants,
        )

        plan_path = plan_dir / "query_plan.json"
        plan_path.write_text(plan.model_dump_json(indent=2), encoding="utf-8")
        manifest = _record_stage(
            manifest,
            "plan",
            "completed",
            started,
            output_paths=[str(plan_path)],
            audit=audit,
            run_root=run_root,
        )
        save_manifest(run_root, manifest)
    else:
        plan_dir = get_stage_dir(run_root, "plan")
        plan_data = json.loads(
            (plan_dir / "query_plan.json").read_text(encoding="utf-8")
        )
        plan = QueryPlan.model_validate(plan_data)

    # --- Stage: search ---
    if memory:
        memory.transition_stage("search")
    if not (resume and _is_stage_complete(manifest, "search")):
        started = utc_now()
        logger.info("Stage: search")
        audit.emit(EventType.STAGE_STARTED, stage="search", run_id=run_id)
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
            audit=audit,
            run_root=run_root,
        )
        save_manifest(run_root, manifest)
    else:
        from research_pipeline.storage.manifests import read_jsonl

        search_dir = get_stage_dir(run_root, "search")
        raw_data = read_jsonl(search_dir / "candidates.jsonl")
        candidates = [CandidateRecord.model_validate(d) for d in raw_data]

    # --- Stage: screen ---
    if memory:
        memory.transition_stage("screen")
    if not (resume and _is_stage_complete(manifest, "screen")):
        started = utc_now()
        logger.info("Stage: screen")
        audit.emit(EventType.STAGE_STARTED, stage="screen", run_id=run_id)
        screen_dir = get_stage_dir(run_root, "screen")

        # Sanitize candidate text before scoring (content security)
        for c in candidates:
            c.title = sanitize_text(c.title, max_length=500)
            c.abstract = sanitize_text(c.abstract, max_length=10_000)

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
            diversity=config.screen.diversity,
            diversity_lambda=config.screen.diversity_lambda,
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

        # Optional LLM second-pass screening
        if llm_provider is not None:
            from research_pipeline.screening.llm_judge import judge_batch

            llm_candidates = [d.paper for d in shortlist]
            judgments = judge_batch(
                llm_candidates,
                topic=plan.topic_raw,
                must_terms=plan.must_terms,
                llm_provider=llm_provider,
            )
            for i, judgment in enumerate(judgments):
                if judgment is not None:
                    shortlist[i] = shortlist[i].model_copy(
                        update={
                            "llm": judgment,
                            "final_score": (
                                0.6 * shortlist[i].cheap.cheap_score
                                + 0.4 * judgment.llm_score
                            ),
                        }
                    )

        shortlist_path = screen_dir / "shortlist.json"
        shortlist_path.write_text(
            json.dumps(
                [d.model_dump(mode="json") for d in shortlist],
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )

        # Confidence-gated retrieval depth classification
        from research_pipeline.screening.depth_gate import classify_retrieval_depth

        depth_tiers = classify_retrieval_depth(shortlist)
        depth_path = screen_dir / "depth_tiers.json"
        depth_path.write_text(
            json.dumps(
                [t.model_dump(mode="json") for t in depth_tiers],
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
        logger.info(
            "Depth tiers: %s",
            (
                {
                    t.tier: sum(1 for x in depth_tiers if x.tier == t.tier)
                    for t in depth_tiers
                }
                if depth_tiers
                else "none"
            ),
        )

        manifest = _record_stage(
            manifest,
            "screen",
            "completed",
            started,
            output_paths=[str(shortlist_path), str(depth_path)],
            audit=audit,
            run_root=run_root,
        )
        save_manifest(run_root, manifest)
    else:
        screen_dir = get_stage_dir(run_root, "screen")
        raw_shortlist = json.loads(
            (screen_dir / "shortlist.json").read_text(encoding="utf-8")
        )
        shortlist = [parse_shortlist_lenient(d) for d in raw_shortlist]

    # --- Stage: download ---
    if memory:
        memory.transition_stage("download")
    if should_run_stage(profile, "download"):
        if not (resume and _is_stage_complete(manifest, "download")):
            started = utc_now()
            logger.info("Stage: download")
            audit.emit(EventType.STAGE_STARTED, stage="download", run_id=run_id)
            pdf_dir = get_stage_dir(run_root, "download")
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
                audit=audit,
                run_root=run_root,
            )
            save_manifest(run_root, manifest)
        else:
            from research_pipeline.models.download import DownloadManifestEntry
            from research_pipeline.storage.manifests import read_jsonl

            dl_data = read_jsonl(
                get_stage_dir(run_root, "download_root") / "download_manifest.jsonl"
            )
            entries = [DownloadManifestEntry.model_validate(d) for d in dl_data]
    else:
        logger.info("Skipping download stage (profile: %s)", profile.value)
        entries = []

    # --- Stage: convert ---
    if memory:
        memory.transition_stage("convert")
    if should_run_stage(profile, "convert"):
        if not (resume and _is_stage_complete(manifest, "convert")):
            started = utc_now()
            logger.info("Stage: convert")
            audit.emit(EventType.STAGE_STARTED, stage="convert", run_id=run_id)
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
                audit=audit,
                run_root=run_root,
            )
            save_manifest(run_root, manifest)
        else:
            from research_pipeline.models.conversion import ConvertManifestEntry
            from research_pipeline.storage.manifests import read_jsonl

            conv_data = read_jsonl(
                get_stage_dir(run_root, "convert_root") / "convert_manifest.jsonl"
            )
            convert_entries = [
                ConvertManifestEntry.model_validate(d) for d in conv_data
            ]
    else:
        logger.info("Skipping convert stage (profile: %s)", profile.value)
        convert_entries = []

    # --- Stage: extract ---
    if memory:
        memory.transition_stage("extract")
    if should_run_stage(profile, "extract"):
        if not (resume and _is_stage_complete(manifest, "extract")):
            started = utc_now()
            logger.info("Stage: extract")
            audit.emit(EventType.STAGE_STARTED, stage="extract", run_id=run_id)
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
                    extract_dir
                    / f"{conv_entry.arxiv_id}{conv_entry.version}.extract.json"
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
                audit=audit,
                run_root=run_root,
            )
            save_manifest(run_root, manifest)
    else:
        logger.info("Skipping extract stage (profile: %s)", profile.value)

    # --- Stage: summarize ---
    if memory:
        memory.transition_stage("summarize")
    if not (resume and _is_stage_complete(manifest, "summarize")):
        started = utc_now()
        logger.info("Stage: summarize")
        audit.emit(EventType.STAGE_STARTED, stage="summarize", run_id=run_id)
        sum_dir = get_stage_dir(run_root, "summarize")

        summaries = []

        if should_run_stage(profile, "download"):
            # Standard/deep path: summarize from converted markdown
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
                    llm_provider=llm_provider,
                )
                summaries.append(summary)

                # Save per-paper summary
                sum_path = (
                    sum_dir / f"{conv_entry.arxiv_id}{conv_entry.version}.summary.json"
                )
                sum_path.write_text(summary.model_dump_json(indent=2), encoding="utf-8")
        else:
            # Quick path: synthesize from abstracts (no PDFs available)
            logger.info(
                "Quick profile: building summaries from abstracts (%d papers)",
                len(shortlist),
            )
            for decision in shortlist:
                paper = decision.paper
                abstract = paper.abstract or ""
                summary = PaperSummary(
                    arxiv_id=paper.arxiv_id,
                    version=paper.version,
                    title=paper.title,
                    objective=(
                        abstract[:500] if abstract else f"See paper: {paper.title}"
                    ),
                    methodology="(Abstract-only — quick profile, no PDF conversion)",
                    findings=[],
                    limitations=[
                        "Abstract-only summary; use standard/deep "
                        "profile for full analysis."
                    ],
                    evidence=[],
                    uncertainties=[],
                )
                summaries.append(summary)

                sum_path = sum_dir / f"{paper.arxiv_id}{paper.version}.summary.json"
                sum_path.write_text(summary.model_dump_json(indent=2), encoding="utf-8")

        # Cross-paper synthesis
        report = synthesize(summaries, plan.topic_raw, llm_provider=llm_provider)
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
            audit=audit,
            run_root=run_root,
        )
        save_manifest(run_root, manifest)

    # --- TER loop (deep profile, or when ter_max_iterations > 0) ---
    ter_max = config.ter_max_iterations
    if should_run_stage(profile, "expand") and ter_max > 0:
        from research_pipeline.pipeline.ter_loop import (
            GapAnalysis,
            TERIteration,
            TERResult,
            check_convergence,
            identify_gaps,
            save_ter_state,
        )

        logger.info("Starting THINK→EXECUTE→REFLECT loop (max %d iterations)", ter_max)
        audit.emit(EventType.STAGE_STARTED, stage="ter_loop", run_id=run_id)

        # Read current synthesis
        sum_dir = get_stage_dir(run_root, "summarize")
        synthesis_path = sum_dir / "synthesis.md"
        synthesis_text = ""
        if synthesis_path.exists():
            synthesis_text = synthesis_path.read_text(encoding="utf-8")

        existing_titles = [s.title for s in summaries]
        ter_result = TERResult()
        previous_gap_analysis: GapAnalysis | None = None

        for iteration_idx in range(ter_max):
            logger.info(
                "TER iteration %d/%d — THINK phase",
                iteration_idx + 1,
                ter_max,
            )

            # THINK: identify gaps
            gap_analysis = identify_gaps(
                synthesis_text, topic, existing_titles, llm_provider
            )
            logger.info(
                "Found %d gaps, %d new queries",
                gap_analysis.gap_count,
                len(gap_analysis.suggested_queries),
            )

            # REFLECT: check convergence
            converged, reason = check_convergence(
                gap_analysis, previous_gap_analysis, iteration_idx, ter_max
            )

            ter_iteration = TERIteration(
                iteration=iteration_idx,
                gaps_found=gap_analysis.gaps,
                queries_generated=gap_analysis.suggested_queries,
                converged=converged,
            )

            if converged:
                logger.info("TER converged: %s", reason)
                ter_result.converged = True
                ter_result.convergence_reason = reason
                ter_result.iterations.append(ter_iteration)
                ter_result.total_iterations = iteration_idx + 1
                break

            # EXECUTE: log the queries that would be searched
            logger.info(
                "TER iteration %d — EXECUTE phase: %d queries to search",
                iteration_idx + 1,
                len(gap_analysis.suggested_queries),
            )
            for q in gap_analysis.suggested_queries:
                logger.info("  Gap-filling query: %s", q)

            ter_result.iterations.append(ter_iteration)
            ter_result.total_iterations = iteration_idx + 1
            previous_gap_analysis = gap_analysis

            # Save state after each iteration
            save_ter_state(
                run_root,
                ter_result,
                {
                    "gaps": gap_analysis.gaps,
                    "queries": gap_analysis.suggested_queries,
                },
            )

        # Save final TER state
        save_ter_state(run_root, ter_result)
        logger.info(
            "TER loop complete: %d iterations, converged=%s (%s)",
            ter_result.total_iterations,
            ter_result.converged,
            ter_result.convergence_reason,
        )
        audit.emit(
            EventType.STAGE_COMPLETED,
            stage="ter_loop",
            run_id=run_id,
            details={
                "iterations": ter_result.total_iterations,
                "converged": ter_result.converged,
                "reason": ter_result.convergence_reason,
            },
        )

    # --- Deep profile: extra stages ---
    if should_run_stage(profile, "expand"):
        logger.info("Deep profile: citation expansion stage (expand)")
        # TODO: Wire expand stage into orchestrator
        logger.info("Expand stage not yet wired into orchestrator — run manually")
    if should_run_stage(profile, "quality"):
        logger.info("Deep profile: quality scoring stage")
        # TODO: Wire quality stage into orchestrator
        logger.info("Quality stage not yet wired into orchestrator — run manually")
    if should_run_stage(profile, "analyze_claims"):
        logger.info("Deep profile: claim analysis stage")
        # TODO: Wire analyze_claims stage into orchestrator
        logger.info(
            "Analyze-claims stage not yet wired into orchestrator — run manually"
        )
    if should_run_stage(profile, "score_claims"):
        logger.info("Deep profile: confidence scoring stage")
        # TODO: Wire score_claims stage into orchestrator
        logger.info("Score-claims stage not yet wired into orchestrator — run manually")

    logger.info("Pipeline complete: run_id=%s", run_id)
    audit.emit(
        EventType.STAGE_COMPLETED,
        stage="pipeline",
        run_id=run_id,
        details={"stages_completed": list(manifest.stages.keys())},
    )

    # Record episode in episodic memory
    if memory:
        try:
            episode = Episode(
                run_id=run_id,
                topic=topic,
                profile=profile.value,
                started_at=manifest.created_at,
                completed_at=utc_now(),
                stages_completed=list(manifest.stages.keys()),
                paper_count=len(candidates),
                shortlist_count=len(shortlist),
            )
            memory.record_run(episode)
        except Exception as exc:
            logger.warning("Failed to record episode: %s", exc)
        finally:
            memory.close()

    return manifest
