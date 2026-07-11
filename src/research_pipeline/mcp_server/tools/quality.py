from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from research_pipeline.mcp_server.schemas import (
    AnalyzePapersInput,
    CiteContextInput,
    ClusterInput,
    EnrichInput,
    EvaluateQualityInput,
    EvidenceAggregateInput,
    ExportBibtexInput,
    ExportHtmlInput,
    GateInfoInput,
    GetVenueTierInput,
    HorizonMetricInput,
    ModelRoutingInfoInput,
    ReportInput,
    RRPDiagnosticInput,
    ToolResult,
    ValidateReportInput,
    WatchInput,
)
from research_pipeline.mcp_server.tools._common import (
    _get_run_root,
    _log_info,
    _raise_tool_error,
    _report_progress,
    _resolve_run_id,
    _resolve_workspace,
    _sanitize_candidates,
    logger,
)

if TYPE_CHECKING:
    from mcp.server.fastmcp import Context


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
            "reproducibility_weight": qc.reproducibility_weight,
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
                message=(
                    "No candidates found. "
                    "Run tool_search or tool_screen_candidates first."
                ),
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
        _raise_tool_error("evaluate_quality", exc)


def get_venue_tier(params: GetVenueTierInput, ctx: Context | None = None) -> ToolResult:
    """Look up CORE venue tier and quality score for a venue name.

    Returns the tier ("A*", "A", "B", "C") and numeric score from the
    bundled CORE rankings, or score=0.1 (unknown) if not found.
    """
    try:
        from research_pipeline.quality.venue_scoring import (
            get_venue_tier as _get_tier,
        )
        from research_pipeline.quality.venue_scoring import (
            venue_score as _venue_score,
        )

        tier = _get_tier(params.venue_name, params.data_path)
        score = _venue_score(params.venue_name, params.data_path)
        return ToolResult(
            success=True,
            message=f"Venue '{params.venue_name}': tier={tier}, score={score}",
            artifacts={"venue": params.venue_name, "tier": tier, "score": score},
        )
    except Exception as exc:
        _raise_tool_error("get_venue_tier", exc)


def analyze_papers(
    params: AnalyzePapersInput, ctx: Context | None = None
) -> ToolResult:
    """Prepare per-paper analysis tasks or validate collected analysis results."""
    try:
        from research_pipeline.analysis.tasks import (
            discover_papers,
            generate_prompt,
            load_research_topic,
        )
        from research_pipeline.storage.workspace import get_stage_dir, init_run

        ws = _resolve_workspace(params.workspace)
        rid = _resolve_run_id(params.run_id)
        _rid, run_root = init_run(ws, rid)

        analysis_dir = get_stage_dir(run_root, "analysis")
        analysis_dir.mkdir(parents=True, exist_ok=True)

        collect_mode = params.mode == "collect" or (not params.mode and params.collect)

        if collect_mode:
            json_files = sorted(analysis_dir.glob("*_analysis.json"))
            if not json_files:
                return ToolResult(
                    success=False,
                    message=(
                        "No analysis JSON files found. "
                        "Launch the paper-analyzer sub-agent first."
                    ),
                )
            from research_pipeline.analysis.tasks import validate_analysis_json

            valid = 0
            total_errs = 0
            results_list = []
            for jf in json_files:
                errs = validate_analysis_json(jf)
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

        papers = discover_papers(run_root)
        if params.paper_ids:
            papers = [p for p in papers if p["arxiv_id"] in params.paper_ids]
        if not papers:
            return ToolResult(
                success=False,
                message="No converted papers found. Run tool_convert_pdfs first.",
            )

        topic = load_research_topic(run_root)
        prompts = [generate_prompt(p, topic, run_root) for p in papers]

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
        _raise_tool_error("analyze_papers", exc)


def validate_report(
    params: ValidateReportInput, ctx: Context | None = None
) -> ToolResult:
    """Validate a research report for completeness and quality."""
    try:
        from research_pipeline.summarization.report_validation import (
            validate_report as _validate,
        )

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
        _raise_tool_error("validate_report", exc)


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
        _raise_tool_error("aggregate_evidence", exc)


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
        _raise_tool_error("export_html", exc)


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
        _raise_tool_error("model_routing_info", exc)


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
        _raise_tool_error("gate_info", exc)


def export_bibtex_tool(
    params: ExportBibtexInput,
    ctx: Context | None = None,
) -> ToolResult:
    """Export papers from a pipeline stage as BibTeX."""
    try:
        from research_pipeline.models.screening import parse_shortlist_lenient
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
            decisions = [parse_shortlist_lenient(item) for item in raw]
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
        _raise_tool_error("export_bibtex", exc)


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
        _raise_tool_error("report", exc)


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
        _raise_tool_error("cluster", exc)


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

        # Enrichment pulls abstracts from Semantic Scholar (untrusted external
        # content); sanitize at this stage boundary before persisting (#104).
        _sanitize_candidates(records)

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
        _raise_tool_error("enrich", exc)


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
        _raise_tool_error("cite_context", exc)


def watch_tool(
    params: WatchInput,
    ctx: Context | None = None,
) -> ToolResult:
    """Check for new papers matching saved watch queries on arXiv."""
    try:
        from research_pipeline.arxiv.client import ArxivClient
        from research_pipeline.arxiv.rate_limit import ArxivRateLimiter
        from research_pipeline.infra.http import create_session
        from research_pipeline.infra.watch_state import (
            DEFAULT_QUERIES_FILE,
            load_queries,
            load_watch_state,
            save_watch_state,
        )

        queries_path = Path(params.queries) if params.queries else DEFAULT_QUERIES_FILE
        _report_progress(ctx, 0, 3, "Loading watch queries")
        queries = load_queries(queries_path)
        if not queries:
            return ToolResult(
                success=False,
                message=(
                    f"No queries found. Create {queries_path} with watch queries."
                ),
            )

        state_path = queries_path.parent / "watch_state.json"
        state = load_watch_state(state_path)

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

        save_watch_state(state_path, state)

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
        _raise_tool_error("watch", exc)


def horizon_metric_tool(
    params: HorizonMetricInput, ctx: Context | None = None
) -> ToolResult:
    """Compute the Unified Horizon Metric (UHM) for a long-horizon run.

    Resolves A3-5 from the Deep Research Report. See
    ``research_pipeline.evaluation.horizon`` for the formula.
    """
    try:
        from research_pipeline.evaluation.horizon import (
            HorizonInputs,
            compute_unified_horizon_metric,
        )

        result = compute_unified_horizon_metric(
            HorizonInputs(
                normalized_score=params.normalized_score,
                difficulty=params.difficulty,
                achieved_steps=params.achieved_steps,
                target_steps=params.target_steps,
                entropy_trend=params.entropy_trend,
                reliability=params.reliability,
            )
        )
        _log_info(ctx, f"UHM = {result.uhm:.4f}")
        return ToolResult(
            success=True,
            message=f"UHM = {result.uhm:.4f}",
            artifacts=result.to_dict(),
        )
    except Exception as exc:
        _raise_tool_error("horizon_metric", exc)


def rrp_diagnostic_tool(
    params: RRPDiagnosticInput, ctx: Context | None = None
) -> ToolResult:
    """Recall / Reasoning / Presentation diagnostic (Theme 16).

    Operationalizes the DeepResearch Bench II finding: Information Recall is
    the primary bottleneck; Presentation is usually near-saturated.
    """
    try:
        from research_pipeline.evaluation.recall_diagnostic import (
            compute_rrp_diagnostic,
        )

        diagnostic = compute_rrp_diagnostic(
            params.report_text, list(params.shortlist_ids)
        )
        _log_info(
            ctx,
            f"RRP: R={diagnostic.recall:.3f} "
            f"Rs={diagnostic.reasoning:.3f} "
            f"P={diagnostic.presentation:.3f} "
            f"(bottleneck={diagnostic.bottleneck})",
        )
        return ToolResult(
            success=True,
            message=(
                f"RRP bottleneck: {diagnostic.bottleneck} "
                f"(overall={diagnostic.overall:.3f})"
            ),
            artifacts=diagnostic.to_dict(),
        )
    except Exception as exc:
        _raise_tool_error("rrp_diagnostic", exc)
