"""MCP server entry point for research-pipeline.

Exposes pipeline stages as MCP tools via stdio transport.
Run with: research-pipeline mcp serve
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from mcp.server.fastmcp import Context, FastMCP
from mcp.types import LoggingLevel, ToolAnnotations

from research_pipeline.mcp_server import (
    completions,
    guard_wiring,
    logging_state,
    prompts,
    resources,
    toolsets,
)
from research_pipeline.mcp_server.schemas import (
    AnalyzeClaimsInput,
    AnalyzePapersInput,
    BriefExportObsidianInput,
    BriefGenerateDailyInput,
    BriefGenerateDossierInput,
    BriefPollSourcesInput,
    BriefRankEventsInput,
    BriefRecordFeedbackInput,
    BriefRunInput,
    BriefValidateReportInput,
    BriefWeeklySynthesisInput,
    CiteContextInput,
    ClusterInput,
    CompareRunsInput,
    ComputeSemanticScoresInput,
    ConvertFileInput,
    ConvertFineInput,
    ConvertPdfsInput,
    ConvertRoughInput,
    DownloadPdfsInput,
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
    GetRunManifestInput,
    GetVenueTierInput,
    HorizonMetricInput,
    KGIngestInput,
    KGQueryInput,
    KGStatsInput,
    ListBackendsInput,
    ManageIndexInput,
    MemoryEpisodesInput,
    MemorySearchInput,
    MemoryStatsInput,
    PlanTopicInput,
    ReportInput,
    RRPDiagnosticInput,
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
from research_pipeline.mcp_server.tools import (
    adaptive_stopping_tool,
    aggregate_evidence_tool,
    analyze_claims_tool,
    analyze_papers,
    blinding_audit_tool,
    brief_export_obsidian_tool,
    brief_generate_daily_tool,
    brief_generate_dossier_tool,
    brief_poll_sources_tool,
    brief_rank_events_tool,
    brief_record_feedback_tool,
    brief_run_tool,
    brief_validate_report_tool,
    brief_weekly_synthesis_tool,
    cbr_lookup_tool,
    cbr_retain_tool,
    cite_context_tool,
    cluster_tool,
    coherence_tool,
    compare_runs,
    compute_semantic_scores,
    confidence_layers_tool,
    consolidation_tool,
    convert_file,
    convert_fine,
    convert_pdfs,
    convert_rough,
    download_pdfs,
    dual_metrics_tool,
    enrich_tool,
    evaluate_quality,
    evaluate_tool,
    expand_citations,
    export_bibtex_tool,
    export_html_tool,
    extract_content,
    gate_info_tool,
    get_run_manifest,
    get_venue_tier,
    horizon_metric_tool,
    kg_ingest_tool,
    kg_quality_tool,
    kg_query_tool,
    kg_stats_tool,
    list_backends,
    manage_index,
    memory_episodes_tool,
    memory_search_tool,
    memory_stats_tool,
    model_routing_info_tool,
    plan_topic,
    query_eval_log,
    record_feedback,
    report_tool,
    rrp_diagnostic_tool,
    run_pipeline,
    score_claims_tool,
    screen_candidates,
    search,
    summarize_papers,
    validate_report,
    verify_stage,
    watch_tool,
)

logger = logging.getLogger(__name__)

mcp = FastMCP(
    "research-pipeline",
    instructions=(
        "MCP server for academic paper research: search multiple sources, "
        "screen, download, convert, extract, summarize papers."
    ),
)


@mcp._mcp_server.set_logging_level()
async def _handle_set_logging_level(level: LoggingLevel) -> None:
    """Honour ``logging/setLevel`` from the client.

    The server emits ``notifications/message`` (via ``ctx.info/.warning/...``),
    so per the MCP spec it MUST declare the ``logging`` capability — registering
    this handler is exactly what advertises it in ``initialize`` — and honour
    the client-set minimum level. Emissions are gated on this level in
    ``tools.py::_log_info`` and ``workflow/telemetry.py``. See issue #41.
    """
    logging_state.set_min_level(level)


@mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    ),
)
def tool_plan_topic(
    topic: str,
    ctx: Context,
    workspace: str = "./workspace",
    run_id: str = "",
) -> ToolResult:
    """Create a structured query plan from a natural language research topic.

    Normalizes the topic, generates query variants and candidate arXiv
    categories. This is the first step in the pipeline.
    """
    result = plan_topic(
        PlanTopicInput(topic=topic, workspace=workspace, run_id=run_id), ctx=ctx
    )
    return result


@mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=False,
        openWorldHint=True,
    ),
)
def tool_search(
    ctx: Context,
    workspace: str = "./workspace",
    run_id: str = "",
    topic: str = "",
    resume: bool = False,
    source: str = "",
) -> ToolResult:
    """Search configured academic paper sources.

    Queries enabled sources with rate limiting, parses responses,
    and deduplicates across sources and query variants.
    Use source='arxiv', 'scholar', 'semantic_scholar', 'openalex',
    'dblp', 'huggingface', 'all', or '' (config default).
    """
    result = search(
        SearchInput(
            workspace=workspace,
            run_id=run_id,
            topic=topic,
            resume=resume,
            source=source,
        ),
        ctx=ctx,
    )
    return result


@mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    ),
)
def tool_screen_candidates(
    ctx: Context,
    workspace: str = "./workspace",
    run_id: str = "",
    resume: bool = False,
) -> ToolResult:
    """Two-stage relevance screening: cheap BM25 scoring then shortlist selection.

    Reads candidates from the search stage, scores them, and produces
    a shortlist of the most relevant papers.
    """
    result = screen_candidates(
        ScreenCandidatesInput(workspace=workspace, run_id=run_id, resume=resume),
        ctx=ctx,
    )
    return result


@mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=True,
    ),
)
def tool_download_pdfs(
    ctx: Context,
    workspace: str = "./workspace",
    run_id: str = "",
    force: bool = False,
) -> ToolResult:
    """Download shortlisted PDFs from arXiv with rate-limit compliance.

    Respects arXiv's 3-second rate limit. Downloads are idempotent
    (skips existing files unless force=True).
    """
    result = download_pdfs(
        DownloadPdfsInput(workspace=workspace, run_id=run_id, force=force), ctx=ctx
    )
    return result


@mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    ),
)
def tool_convert_pdfs(
    ctx: Context,
    workspace: str = "./workspace",
    run_id: str = "",
    force: bool = False,
    backend: str = "",
) -> ToolResult:
    """Convert downloaded PDFs to Markdown.

    Supports multiple backends: docling, marker, pymupdf4llm (local) and
    mathpix, datalab, llamaparse, mistral_ocr, openai_vision (cloud/online).
    Use backend='' to use the config default.
    Requires the corresponding extra to be installed.
    """
    result = convert_pdfs(
        ConvertPdfsInput(
            workspace=workspace, run_id=run_id, force=force, backend=backend
        ),
        ctx=ctx,
    )
    return result


@mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    ),
)
def tool_extract_content(
    ctx: Context,
    workspace: str = "./workspace",
    run_id: str = "",
) -> ToolResult:
    """Extract structured content (chunks, sections) from converted Markdown.

    Performs chunking and indexing for downstream summarization.
    """
    result = extract_content(
        ExtractContentInput(workspace=workspace, run_id=run_id), ctx=ctx
    )
    return result


@mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    ),
)
def tool_summarize_papers(
    ctx: Context,
    workspace: str = "./workspace",
    run_id: str = "",
) -> ToolResult:
    """Generate per-paper summaries and cross-paper synthesis.

    Produces evidence-backed summaries with chunk citations,
    plus a synthesis report comparing findings across papers.
    """
    result = summarize_papers(
        SummarizePapersInput(workspace=workspace, run_id=run_id), ctx=ctx
    )
    return result


@mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=False,
        openWorldHint=True,
    ),
)
def tool_run_pipeline(
    topic: str,
    ctx: Context,
    workspace: str = "./workspace",
    run_id: str = "",
    resume: bool = False,
) -> ToolResult:
    """Run the full pipeline end-to-end.

    Stages: plan → search → screen → download → convert →
    extract → summarize. All stages produce auditable artifacts
    in the workspace.
    """
    result = run_pipeline(
        RunPipelineInput(
            topic=topic, workspace=workspace, run_id=run_id, resume=resume
        ),
        ctx=ctx,
    )
    return result


@mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    ),
)
def tool_get_run_manifest(
    ctx: Context,
    workspace: str = "./workspace",
    run_id: str = "",
) -> ToolResult:
    """Inspect a run's manifest: stages completed, artifacts produced, timing.

    Use this to check the status of a pipeline run.
    """
    result = get_run_manifest(
        GetRunManifestInput(workspace=workspace, run_id=run_id), ctx=ctx
    )
    return result


@mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    ),
)
def tool_convert_file(
    pdf_path: str,
    ctx: Context,
    output_dir: str = "",
    backend: str = "",
) -> ToolResult:
    """Convert a single PDF file to Markdown (standalone, no pipeline workspace needed).

    Supports multiple backends: docling, marker, pymupdf4llm (local) and
    mathpix, datalab, llamaparse, mistral_ocr, openai_vision (cloud/online).
    Use backend='' to use the config default. Useful for ad-hoc
    document conversion without running the full pipeline.
    """
    result = convert_file(
        ConvertFileInput(pdf_path=pdf_path, output_dir=output_dir, backend=backend),
        ctx=ctx,
    )
    return result


@mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    ),
)
def tool_list_backends(ctx: Context) -> ToolResult:
    """List available PDF-to-Markdown converter backends.

    Returns the names of all registered backends: docling, marker,
    pymupdf4llm (local) and mathpix, datalab, llamaparse, mistral_ocr,
    openai_vision (cloud/online).
    Each backend requires its corresponding extra to be installed.
    """
    result = list_backends(ListBackendsInput(), ctx=ctx)
    return result


@mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=False,
        openWorldHint=True,
    ),
)
def tool_expand_citations(
    paper_ids: list[str],
    ctx: Context,
    workspace: str = "./workspace",
    run_id: str = "",
    direction: str = "both",
    limit: int = 50,
) -> ToolResult:
    """Expand citation graph for specified papers via Semantic Scholar.

    Fetches papers that cite or are referenced by the given seed papers.
    Use direction='citations', 'references', or 'both'.
    Requires explicit paper IDs (arXiv IDs or S2 paper IDs).
    """
    result = expand_citations(
        ExpandCitationsInput(
            paper_ids=paper_ids,
            workspace=workspace,
            run_id=run_id,
            direction=direction,
            limit=limit,
        ),
        ctx=ctx,
    )
    return result


@mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=True,
    ),
)
def tool_evaluate_quality(
    ctx: Context,
    workspace: str = "./workspace",
    run_id: str = "",
) -> ToolResult:
    """Compute composite quality scores for candidate papers.

    Evaluates papers on citation impact, venue reputation (CORE rankings),
    author h-index credibility, and recency. Requires a completed search
    or screen stage.
    """
    result = evaluate_quality(
        EvaluateQualityInput(workspace=workspace, run_id=run_id), ctx=ctx
    )
    return result


@mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    ),
)
def tool_get_venue_tier(
    venue_name: str,
    ctx: Context,
    data_path: str = "",
) -> ToolResult:
    """Look up CORE venue tier and quality score for a venue.

    Returns the tier label (A*, A, B, C) and numeric score for the given
    venue name. Uses the bundled CORE 2023 rankings data.
    """
    result = get_venue_tier(
        GetVenueTierInput(venue_name=venue_name, data_path=data_path), ctx=ctx
    )
    return result


@mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    ),
)
def tool_compute_semantic_scores(
    topic: str,
    ctx: Context,
    workspace: str = "./workspace",
    run_id: str = "",
    model_name: str = "allenai/specter2",
    batch_size: int = 32,
) -> ToolResult:
    """Compute SPECTER2 semantic similarity scores for all candidate papers.

    Embeds the topic query and each candidate paper using SPECTER2, then
    returns per-candidate cosine similarity scores in [0, 1] (min-max
    normalised). Requires a completed search or screen stage.
    """
    result = compute_semantic_scores(
        ComputeSemanticScoresInput(
            topic=topic,
            workspace=workspace,
            run_id=run_id,
            model_name=model_name,
            batch_size=batch_size,
        ),
        ctx=ctx,
    )
    return result


@mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    ),
)
def tool_convert_rough(
    ctx: Context,
    workspace: str = "./workspace",
    run_id: str = "",
    force: bool = False,
) -> ToolResult:
    """Fast Tier 2 conversion of all downloaded PDFs using pymupdf4llm.

    CPU-only, fast conversion for all papers. The agent reads rough
    markdown to decide which papers need fine conversion.
    Requires a completed download stage.
    """
    result = convert_rough(
        ConvertRoughInput(workspace=workspace, run_id=run_id, force=force), ctx=ctx
    )
    return result


@mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    ),
)
def tool_convert_fine(
    paper_ids: list[str],
    ctx: Context,
    workspace: str = "./workspace",
    run_id: str = "",
    force: bool = False,
    backend: str = "",
) -> ToolResult:
    """High-quality Tier 3 conversion of selected PDFs.

    Converts agent-selected papers using docling, marker, or cloud
    backend. Requires explicit paper IDs and a completed download stage.
    Use backend='' for config default.
    """
    result = convert_fine(
        ConvertFineInput(
            paper_ids=paper_ids,
            workspace=workspace,
            run_id=run_id,
            force=force,
            backend=backend,
        ),
        ctx=ctx,
    )
    return result


@mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    ),
)
def tool_manage_index(
    ctx: Context,
    list_papers: bool = False,
    gc: bool = False,
    db_path: str = "",
) -> ToolResult:
    """Manage the global paper index for incremental runs.

    Browse indexed papers (list_papers=true) or clean stale entries
    (gc=true). The global index deduplicates papers across runs.
    """
    result = manage_index(
        ManageIndexInput(list_papers=list_papers, gc=gc, db_path=db_path), ctx=ctx
    )
    return result


@mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    ),
)
def tool_analyze_papers(
    ctx: Context,
    workspace: str = "./workspace",
    run_id: str = "",
    collect: bool = False,
    paper_ids: list[str] | None = None,
) -> ToolResult:
    """Prepare per-paper analysis tasks or validate collected results.

    Without collect=True: discovers converted papers, generates analysis
    prompts/tasks for sub-agents. With collect=True: validates collected
    analysis JSON files against the required schema.
    """
    result = analyze_papers(
        AnalyzePapersInput(
            workspace=workspace,
            run_id=run_id,
            collect=collect,
            paper_ids=paper_ids or [],
        ),
        ctx=ctx,
    )
    return result


@mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    ),
)
def tool_validate_report(
    ctx: Context,
    report_path: str = "",
    workspace: str = "./workspace",
    run_id: str = "",
) -> ToolResult:
    """Validate a research report for completeness and quality.

    Checks for 14 required sections, confidence-level annotations,
    evidence citations, gap classifications, tables, Mermaid diagrams,
    and LaTeX formulas. Returns PASS/FAIL verdict with detailed scoring.
    """
    result = validate_report(
        ValidateReportInput(
            report_path=report_path, workspace=workspace, run_id=run_id
        ),
        ctx=ctx,
    )
    return result


@mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    ),
)
def tool_compare_runs(
    run_id_a: str,
    run_id_b: str,
    ctx: Context,
    workspace: str = "./workspace",
) -> ToolResult:
    """Compare two pipeline runs and produce a structured diff.

    Analyzes paper overlap, gap resolution, confidence-level changes,
    readiness assessment progression, and quality score differences
    between two runs. Useful for tracking research progress across
    iterations.
    """
    result = compare_runs(
        CompareRunsInput(workspace=workspace, run_id_a=run_id_a, run_id_b=run_id_b),
        ctx=ctx,
    )
    return result


@mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    ),
)
def tool_verify_stage(
    stage: str,
    ctx: Context,
    workspace: str = "./workspace",
    run_id: str = "",
) -> ToolResult:
    """Verify structural completeness of a pipeline stage output.

    Runs structural verification gates (not LLM-based) to confirm
    a stage produced valid output. Checks file existence, sizes,
    required fields, and format constraints. Stages: plan, search,
    screen, download, convert, extract, summarize.
    """
    result = verify_stage(
        VerifyStageInput(workspace=workspace, run_id=run_id, stage=stage),
        ctx=ctx,
    )
    return result


@mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    ),
)
def tool_record_feedback(
    ctx: Context,
    workspace: str = "./workspace",
    run_id: str = "",
    accept: list[str] | None = None,
    reject: list[str] | None = None,
    reason: str = "",
    show: bool = False,
    adjust: bool = False,
) -> ToolResult:
    """Record user accept/reject feedback on screened papers.

    Stores decisions in a persistent SQLite database. Accumulated
    feedback adjusts BM25 screening weights via ELO-style learning,
    improving future screening precision.

    Set adjust=True to recompute optimized weights after recording.
    Set show=True to display current feedback statistics.
    """
    result = record_feedback(
        FeedbackInput(
            workspace=workspace,
            run_id=run_id,
            accept=accept or [],
            reject=reject or [],
            reason=reason,
            show=show,
            adjust=adjust,
        ),
        ctx=ctx,
    )
    return result


@mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    ),
)
def tool_query_eval_log(
    ctx: Context,
    workspace: str = "./workspace",
    run_id: str = "",
    channel: str = "all",
    stage: str = "",
    limit: int = 50,
) -> ToolResult:
    """Query three-channel evaluation logs for a pipeline run.

    Three channels capture different aspects of execution:
    - traces: Execution flow (JSONL) with timing and causality
    - audit: Structured DB (SQLite) with who/what/when records
    - snapshots: Filesystem state captures at stage boundaries

    Use channel='summary' for an overview of all three channels.
    """
    result = query_eval_log(
        EvalLogInput(
            workspace=workspace,
            run_id=run_id,
            channel=channel,
            stage=stage,
            limit=limit,
        ),
        ctx=ctx,
    )
    return result


@mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    ),
)
def tool_aggregate_evidence(
    ctx: Context,
    workspace: str = "./workspace",
    run_id: str = "",
    min_pointers: int = 0,
    max_words: int = 50,
    similarity_threshold: float = 0.7,
    strip_rhetoric: bool = True,
    output_format: str = "text",
) -> ToolResult:
    """Aggregate evidence from synthesis, stripping rhetoric.

    Processes synthesis report through evidence-only aggregation:
    - Strips hedging, confidence claims, subjective opinions, filler
    - Normalizes statement length
    - Extracts and validates evidence pointers
    - Merges semantically similar statements
    - Filters by minimum evidence requirements
    """
    result = aggregate_evidence_tool(
        EvidenceAggregateInput(
            workspace=workspace,
            run_id=run_id,
            min_pointers=min_pointers,
            max_words=max_words,
            similarity_threshold=similarity_threshold,
            strip_rhetoric=strip_rhetoric,
            output_format=output_format,
        ),
        ctx=ctx,
    )
    return result


@mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    ),
)
def tool_export_html(
    ctx: Context,
    workspace: str = "./workspace",
    run_id: str = "",
    markdown_file: str = "",
    title: str = "Research Report",
    output: str = "",
) -> ToolResult:
    """Export synthesis report as self-contained HTML.

    Two modes:
    - run_id: Renders structured SynthesisReport as rich HTML with
      citation links, confidence badges, and collapsible sections.
    - markdown_file: Converts any Markdown file to styled HTML.
    """
    result = export_html_tool(
        ExportHtmlInput(
            workspace=workspace,
            run_id=run_id,
            markdown_file=markdown_file,
            title=title,
            output=output,
        ),
        ctx=ctx,
    )
    return result


@mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    ),
)
async def tool_model_routing_info(
    config_path: str = "",
) -> ToolResult:
    """Show current phase-aware model routing configuration.

    Returns which LLM provider is assigned to each phase tier
    (mechanical, intelligent, critical_safety) and the stage→tier mapping.
    """
    from research_pipeline.mcp_server.schemas import ModelRoutingInfoInput

    params = ModelRoutingInfoInput(config_path=config_path)
    result = model_routing_info_tool(params=params)
    return result


@mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    ),
)
async def tool_gate_info(
    config_path: str = "",
) -> ToolResult:
    """Show current HITL gate configuration.

    Returns which stages have approval gates and whether
    gates are in auto-approve or interactive mode.
    """
    from research_pipeline.mcp_server.schemas import GateInfoInput

    params = GateInfoInput(config_path=config_path)
    result = gate_info_tool(params=params)
    return result


@mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    ),
)
async def tool_coherence(
    run_ids: list[str],
    workspace: str = "./workspace",
) -> ToolResult:
    """Evaluate multi-session coherence across pipeline runs.

    Computes factual consistency, temporal ordering, knowledge update
    fidelity, and contradiction detection across 2+ runs.

    Args:
        run_ids: Two or more run IDs to evaluate.
        workspace: Workspace directory containing run outputs.
    """
    from research_pipeline.mcp_server.schemas import CoherenceInput

    params = CoherenceInput(run_ids=run_ids, workspace=workspace)
    result = coherence_tool(params=params)
    return result


@mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    ),
)
async def tool_consolidation(
    workspace: str = "./workspace",
    run_ids: list[str] | None = None,
    dry_run: bool = False,
    capacity: int = 100,
    threshold: float = 0.8,
    min_support: int = 2,
) -> ToolResult:
    """Consolidate cross-run memory: compress episodes, promote rules, prune stale.

    Implements episodic → semantic consolidation following the SEA/MLMF
    three-tier memory architecture. Automatically ingests synthesis
    results into the episode store and promotes recurring findings to rules.

    Args:
        workspace: Workspace directory containing run outputs.
        run_ids: Run IDs to ingest. If None, scans workspace.
        dry_run: Compute metrics without modifying store.
        capacity: Episode capacity before triggering consolidation.
        threshold: Fraction of capacity triggering consolidation.
        min_support: Min run appearances for rule promotion.
    """
    from research_pipeline.mcp_server.schemas import ConsolidationInput

    params = ConsolidationInput(
        workspace=workspace,
        run_ids=run_ids,
        dry_run=dry_run,
        capacity=capacity,
        threshold=threshold,
        min_support=min_support,
    )
    result = consolidation_tool(params=params)
    return result


@mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    ),
)
async def tool_blinding_audit(
    workspace: str = "./workspace",
    run_id: str = "",
    threshold: float = 0.4,
    store_results: bool = True,
) -> ToolResult:
    """Run epistemic blinding audit to detect LLM prior contamination.

    Implements A/B blinding protocol (arXiv 2604.06013): scans analysis
    outputs for identifying feature references and scores contamination
    level per paper and across the run.

    Args:
        workspace: Workspace directory containing run outputs.
        run_id: Specific run ID to audit (latest if empty).
        threshold: Contamination threshold for flagging papers.
        store_results: Whether to persist results to SQLite.
    """
    from research_pipeline.mcp_server.schemas import BlindingAuditInput

    params = BlindingAuditInput(
        workspace=workspace,
        run_id=run_id,
        threshold=threshold,
        store_results=store_results,
    )
    result = blinding_audit_tool(params=params)
    return result


@mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    ),
)
async def tool_dual_metrics(
    query: str,
    workspace: str = "./workspace",
    run_ids: list[str] | None = None,
    k: int = 5,
    store_results: bool = True,
) -> ToolResult:
    """Evaluate pipeline runs using Pass@k + Pass[k] dual metrics.

    Computes capability ceiling (Pass@k) and reliability floor (Pass[k])
    across multiple runs of the same research query. Applies a multiplicative
    safety gate that zeros scores when fabrication is detected.

    Based on the Claw-Eval framework (arXiv 2604.06132).

    Args:
        query: Research query these runs address.
        workspace: Workspace directory containing run outputs.
        run_ids: Run IDs to evaluate (auto-discover if empty).
        k: Number of samples for Pass@k / Pass[k] computation.
        store_results: Whether to persist results to SQLite.
    """
    from research_pipeline.mcp_server.schemas import DualMetricsInput

    params = DualMetricsInput(
        workspace=workspace,
        query=query,
        run_ids=run_ids or [],
        k=k,
        store_results=store_results,
    )
    result = dual_metrics_tool(params=params)
    return result


@mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    ),
)
async def tool_cbr_lookup(
    topic: str,
    workspace: str = "./workspace",
    max_results: int = 5,
    min_quality: float = 0.0,
) -> ToolResult:
    """Look up similar past cases and recommend a research strategy.

    Uses Case-Based Reasoning (arXiv 2506.18096) to retrieve successful
    past research strategies and adapt them for new topics.

    Args:
        topic: Research topic to look up similar past cases.
        workspace: Workspace directory.
        max_results: Maximum number of similar cases to retrieve.
        min_quality: Minimum synthesis quality to consider.
    """
    from research_pipeline.mcp_server.schemas import CbrLookupInput

    params = CbrLookupInput(
        workspace=workspace,
        topic=topic,
        max_results=max_results,
        min_quality=min_quality,
    )
    result = cbr_lookup_tool(params=params)
    return result


@mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    ),
)
async def tool_cbr_retain(
    run_id: str,
    topic: str,
    workspace: str = "./workspace",
    outcome: str = "unknown",
    strategy_notes: str = "",
) -> ToolResult:
    """Store a completed pipeline run as a CBR case.

    Extracts strategy information (queries, sources, quality) from run
    artifacts and stores for future retrieval by cbr-lookup.

    Args:
        run_id: Pipeline run ID to store as a case.
        topic: Research topic for this run.
        workspace: Workspace directory.
        outcome: Quality outcome: excellent, good, adequate, poor, failed.
        strategy_notes: Free-text notes about the strategy used.
    """
    from research_pipeline.mcp_server.schemas import CbrRetainInput

    params = CbrRetainInput(
        workspace=workspace,
        run_id=run_id,
        topic=topic,
        outcome=outcome,
        strategy_notes=strategy_notes,
    )
    result = cbr_retain_tool(params=params)
    return result


@mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    ),
)
async def tool_kg_quality(
    db_path: str = "",
    staleness_days: float = 365.0,
    sample_size: int = 0,
) -> ToolResult:
    """Evaluate knowledge graph quality across 5 dimensions.

    Three-layer composable architecture (TKDE 2022 + Text2KGBench):
    structural metrics → IC+EC consistency → TWCS sampling.

    Args:
        db_path: Path to KG SQLite database. Empty uses default.
        staleness_days: Threshold for timeliness staleness.
        sample_size: If > 0, also run TWCS sampling and return sample.
    """
    from research_pipeline.mcp_server.schemas import KGQualityInput

    params = KGQualityInput(
        db_path=db_path,
        staleness_days=staleness_days,
        sample_size=sample_size,
    )
    result = kg_quality_tool(params=params)
    return result


@mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    ),
)
async def tool_adaptive_stopping(
    batch_scores: list[list[float]],
    query: str = "",
    query_type: str = "auto",
    min_results: int = 5,
    max_budget: int = 500,
    relevance_threshold: float = 0.5,
) -> ToolResult:
    """Evaluate query-adaptive retrieval stopping criteria.

    Three strategies based on query type (HingeMem WWW '26):
    - recall: knee detection on cumulative relevance
    - precision: top-k saturation check
    - judgment: top-1 stability across batches

    Args:
        batch_scores: List of score lists, one per retrieval batch.
        query: Original query for auto-classifying stopping strategy.
        query_type: Query type: recall, precision, judgment, or auto.
        min_results: Minimum results before stopping considered.
        max_budget: Hard budget limit on total results.
        relevance_threshold: Score threshold for relevant results.
    """
    from research_pipeline.mcp_server.schemas import AdaptiveStoppingInput

    params = AdaptiveStoppingInput(
        batch_scores=batch_scores,
        query=query,
        query_type=query_type,
        min_results=min_results,
        max_budget=max_budget,
        relevance_threshold=relevance_threshold,
    )
    result = adaptive_stopping_tool(params=params)
    return result


@mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    ),
)
async def tool_confidence_layers(
    run_id: str,
    config_path: str = "",
    workspace: str = "./workspace",
    l4_threshold: float = 0.50,
    damping: float = 0.80,
    calibrate: bool = False,
) -> ToolResult:
    """Score claims through the 4-layer confidence architecture.

    L1 (fast signal) → L2 (adaptive granularity) → L3 (DINCO calibration)
    → L4 (selective verification for low-confidence claims only).

    Args:
        run_id: Run ID containing claim decompositions.
        config_path: Path to config.toml. Empty uses defaults.
        workspace: Workspace directory. Empty uses config default.
        l4_threshold: Confidence below which L4 verification triggers.
        damping: Fusion damping exponent (0-1).
        calibrate: Whether to fit Platt scaling from prior scored claims.
    """
    from research_pipeline.mcp_server.schemas import ConfidenceLayersInput

    params = ConfidenceLayersInput(
        run_id=run_id,
        config_path=config_path,
        workspace=workspace,
        l4_threshold=l4_threshold,
        damping=damping,
        calibrate=calibrate,
    )
    result = confidence_layers_tool(params=params)
    return result


@mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=False,
        openWorldHint=True,
    ),
)
async def tool_research_workflow(
    topic: str,
    ctx: Context,
    workspace: str = "./workspace",
    run_id: str = "",
    config_path: str = "",
    system_building: bool = False,
    source: str = "",
    max_iterations: int = 3,
    resume: bool = False,
) -> ToolResult:
    """Run a full harness-engineered research workflow.

    Orchestrates the entire pipeline: plan → search → screen → download →
    convert → extract → summarize, with optional sampling-based analysis,
    iterative synthesis, and user elicitation at decision gates.

    Features 6 harness engineering layers:
    - Telemetry: three-surface logging (cognitive/operational/contextual)
    - Context engineering: token budgets and paper compaction
    - Governance: schema-level state machine with verify-before-commit
    - Verification: structural output validation (not self-referential)
    - Monitoring: doom-loop detection and iteration drift tracking
    - Recovery: persistent state after every stage for crash-recovery

    Degrades gracefully:
    - Without sampling capability: pipeline-only mode (no LLM analysis)
    - Without elicitation capability: uses sensible defaults at gates
    """
    from research_pipeline.mcp_server.workflow.research import run_research_workflow

    result = await run_research_workflow(
        topic=topic,
        ctx=ctx,
        workspace=workspace,
        run_id=run_id,
        config_path=config_path,
        system_building=system_building,
        source=source,
        max_iterations=max_iterations,
        resume=resume,
    )
    # Wrap the workflow's dict in the uniform ToolResult envelope every other
    # tool returns, so structuredContent always carries success/message/artifacts
    # (#110). The full workflow payload is preserved under artifacts.
    success = bool(result.get("success", "error" not in result))
    message = str(
        result.get("message") or result.get("error") or "Research workflow complete."
    )
    return ToolResult(success=success, message=message, artifacts=result)


@mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    ),
)
async def tool_export_bibtex(
    run_id: str,
    stage: str = "screen",
    output: str = "",
    workspace: str = "./workspace",
) -> ToolResult:
    """Export papers from a pipeline stage as BibTeX.

    Reads candidate JSONL files from the specified stage and produces
    a .bib file suitable for LaTeX workflows.

    Args:
        run_id: Pipeline run ID.
        stage: Stage to export from (search, screen, download).
        output: Output .bib file path (default: auto in run dir).
        workspace: Workspace directory.
    """
    params = ExportBibtexInput(
        run_id=run_id,
        stage=stage,
        output=output,
        workspace=workspace,
    )
    result = export_bibtex_tool(params=params)
    return result


@mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    ),
)
async def tool_report(
    run_id: str,
    template: str = "survey",
    custom_template: str = "",
    output: str = "",
    workspace: str = "./workspace",
) -> ToolResult:
    """Render a synthesis report using a configurable template.

    Reads synthesis_report.json or synthesis.json from the summarize stage
    and renders it through a Jinja2 template to produce a formatted Markdown
    report.

    Args:
        run_id: Pipeline run ID.
        template: Report template (survey, gap_analysis, lit_review, executive).
        custom_template: Path to a custom Jinja2 template file.
        output: Output Markdown file path (default: auto in run dir).
        workspace: Workspace directory.
    """
    params = ReportInput(
        run_id=run_id,
        template=template,
        custom_template=custom_template,
        output=output,
        workspace=workspace,
    )
    result = report_tool(params=params)
    return result


@mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    ),
)
async def tool_cluster(
    run_id: str,
    stage: str = "screen",
    threshold: float = 0.15,
    output: str = "",
    workspace: str = "./workspace",
) -> ToolResult:
    """Cluster papers by topic similarity using TF-IDF.

    Groups screened candidates into topically coherent clusters for
    better organization before synthesis.

    Args:
        run_id: Pipeline run ID.
        stage: Stage to cluster from (search or screen).
        threshold: Cosine similarity threshold (0-1).
        output: Output JSON file path (default: auto in run dir).
        workspace: Workspace directory.
    """
    params = ClusterInput(
        run_id=run_id,
        stage=stage,
        threshold=threshold,
        output=output,
        workspace=workspace,
    )
    result = cluster_tool(params=params)
    return result


@mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=False,
        openWorldHint=True,
    ),
)
async def tool_enrich(
    run_id: str,
    stage: str = "candidates",
    workspace: str = "./workspace",
    config_path: str = "",
) -> ToolResult:
    """Enrich candidates with missing abstracts/metadata from Semantic Scholar.

    Queries Semantic Scholar by DOI or title to fill in missing abstracts
    and citation counts for candidate papers.

    Args:
        run_id: Pipeline run ID.
        stage: Stage to read candidates from (candidates or screened).
        workspace: Workspace directory.
        config_path: Path to config.toml. Empty uses defaults.
    """
    params = EnrichInput(
        run_id=run_id,
        stage=stage,
        workspace=workspace,
        config_path=config_path,
    )
    result = enrich_tool(params=params)
    return result


@mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    ),
)
async def tool_cite_context(
    run_id: str,
    window: int = 1,
    output: str = "",
    workspace: str = "./workspace",
    config_path: str = "",
) -> ToolResult:
    """Extract citation contexts from converted Markdown papers.

    Finds citation markers in converted papers and extracts the
    surrounding sentences for citation analysis.

    Args:
        run_id: Pipeline run ID.
        window: Extra sentences before/after citation (0 = citing only).
        output: Output JSON file path (default: auto in convert dir).
        workspace: Workspace directory.
        config_path: Path to config.toml. Empty uses defaults.
    """
    params = CiteContextInput(
        run_id=run_id,
        window=window,
        output=output,
        workspace=workspace,
        config_path=config_path,
    )
    result = cite_context_tool(params=params)
    return result


@mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=False,
        openWorldHint=True,
    ),
)
async def tool_watch(
    queries: str = "",
    lookback: int = 7,
    max_results: int = 20,
    output: str = "",
    config_path: str = "",
) -> ToolResult:
    """Check for new papers matching saved watch queries on arXiv.

    Monitors saved queries for recently published papers. Designed
    to be called periodically to track new developments.

    Args:
        queries: Path to watch queries JSON file (default: ~/.cache/...).
        lookback: Days to look back for new papers.
        max_results: Maximum results per query.
        output: Output JSON file path for new papers found.
        config_path: Path to config.toml. Empty uses defaults.
    """
    params = WatchInput(
        queries=queries,
        lookback=lookback,
        max_results=max_results,
        output=output,
        config_path=config_path,
    )
    result = watch_tool(params=params)
    return result


@mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    ),
)
def tool_analyze_claims(
    run_id: str,
    ctx: Context,
    workspace: str = "./workspace",
) -> ToolResult:
    """Decompose paper summaries into atomic claims with evidence classification.

    Breaks each paper's summary into individual factual claims and classifies
    their evidence type (supported, unsupported, contradicted).

    Args:
        run_id: Pipeline run ID.
        workspace: Workspace directory. Empty uses config default.
    """
    result = analyze_claims_tool(
        AnalyzeClaimsInput(run_id=run_id, workspace=workspace), ctx=ctx
    )
    return result


@mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    ),
)
def tool_score_claims(
    run_id: str,
    ctx: Context,
    workspace: str = "./workspace",
) -> ToolResult:
    """Score confidence for decomposed claims using LLM evaluation.

    Assigns confidence scores to each atomic claim produced by
    analyze-claims, using LLM-based assessment when available.

    Args:
        run_id: Pipeline run ID.
        workspace: Workspace directory. Empty uses config default.
    """
    result = score_claims_tool(
        ScoreClaimsInput(run_id=run_id, workspace=workspace), ctx=ctx
    )
    return result


@mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    ),
)
def tool_kg_stats(
    db_path: str = "",
) -> ToolResult:
    """Show knowledge graph statistics.

    Returns entity and triple counts, type distributions, and
    other statistics about the knowledge graph.

    Args:
        db_path: Path to KG database. Empty uses default.
    """
    result = kg_stats_tool(KGStatsInput(db_path=db_path))
    return result


@mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    ),
)
def tool_kg_query(
    entity_id: str,
    db_path: str = "",
) -> ToolResult:
    """Query an entity and its relations in the knowledge graph.

    Returns the entity details and all connected relations
    (both incoming and outgoing).

    Args:
        entity_id: Entity ID to query.
        db_path: Path to KG database. Empty uses default.
    """
    result = kg_query_tool(KGQueryInput(entity_id=entity_id, db_path=db_path))
    return result


@mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=False,
        openWorldHint=False,
    ),
)
def tool_kg_ingest(
    run_id: str,
    ctx: Context,
    workspace: str = "./workspace",
    db_path: str = "",
) -> ToolResult:
    """Ingest pipeline results into the knowledge graph.

    Loads candidates and claim decompositions from a pipeline run
    and creates entities and triples in the knowledge graph.

    Args:
        run_id: Pipeline run ID to ingest from.
        workspace: Workspace directory. Empty uses config default.
        db_path: Path to KG database. Empty uses default.
    """
    result = kg_ingest_tool(
        KGIngestInput(run_id=run_id, workspace=workspace, db_path=db_path), ctx=ctx
    )
    return result


@mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    ),
)
def tool_memory_stats(
    episodic_db: str = "",
    kg_db: str = "",
) -> ToolResult:
    """Show memory tier statistics.

    Returns statistics for episodic memory (past runs) and
    knowledge graph (entities, triples).

    Args:
        episodic_db: Path to episodic memory database. Empty uses default.
        kg_db: Path to KG database. Empty uses default.
    """
    result = memory_stats_tool(MemoryStatsInput(episodic_db=episodic_db, kg_db=kg_db))
    return result


@mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    ),
)
def tool_memory_episodes(
    limit: int = 20,
    episodic_db: str = "",
) -> ToolResult:
    """List recent episodic memories (past pipeline runs).

    Returns recent pipeline run episodes with topic, paper counts,
    and completion status.

    Args:
        limit: Maximum episodes to return.
        episodic_db: Path to episodic memory database. Empty uses default.
    """
    result = memory_episodes_tool(
        MemoryEpisodesInput(limit=limit, episodic_db=episodic_db)
    )
    return result


@mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    ),
)
def tool_memory_search(
    topic: str,
    limit: int = 10,
    episodic_db: str = "",
) -> ToolResult:
    """Search episodic memory for past runs on a topic.

    Finds previous pipeline runs that match the given topic,
    useful for finding related prior research.

    Args:
        topic: Topic to search for.
        limit: Maximum results to return.
        episodic_db: Path to episodic memory database. Empty uses default.
    """
    result = memory_search_tool(
        MemorySearchInput(topic=topic, limit=limit, episodic_db=episodic_db)
    )
    return result


@mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    ),
)
def tool_evaluate(
    run_id: str,
    workspace: str = "./workspace",
    stage: str = "",
) -> ToolResult:
    """Evaluate pipeline outputs against their schemas.

    Validates that pipeline stage outputs conform to expected
    formats and schemas. Can check a single stage or all stages.

    Args:
        run_id: Pipeline run ID to evaluate.
        workspace: Workspace directory.
        stage: Specific stage to evaluate. Empty checks all.
    """
    result = evaluate_tool(
        EvaluateInput(run_id=run_id, workspace=workspace, stage=stage)
    )
    return result


@mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    ),
)
def tool_horizon_metric(
    normalized_score: float,
    achieved_steps: int,
    target_steps: int,
    difficulty: float = 0.5,
    entropy_trend: float = 0.0,
    reliability: float = 1.0,
) -> ToolResult:
    """Compute the Unified Horizon Metric (UHM).

    Combines quality, difficulty, horizon-length, and stability into a single
    comparable number in [0, 1]. Closes gap A3-5 of the Deep Research Report.

    Args:
        normalized_score: Task quality in [0, 1].
        achieved_steps: Trajectory length actually completed.
        target_steps: Benchmark target horizon.
        difficulty: Task difficulty in [0, 1].
        entropy_trend: Token-entropy slope across trajectory (neg=locking).
        reliability: Optional Pass[k] reliability floor in [0, 1].
    """
    result = horizon_metric_tool(
        HorizonMetricInput(
            normalized_score=normalized_score,
            achieved_steps=achieved_steps,
            target_steps=target_steps,
            difficulty=difficulty,
            entropy_trend=entropy_trend,
            reliability=reliability,
        )
    )
    return result


@mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    ),
)
def tool_rrp_diagnostic(
    report_text: str,
    shortlist_ids: list[str] | None = None,
) -> ToolResult:
    """Recall / Reasoning / Presentation diagnostic (Theme 16).

    Decomposes a synthesis report's quality along three axes to localize
    bottlenecks, following DeepResearch Bench II findings (Info Recall is
    typically <50% while Presentation is near-saturated).

    Args:
        report_text: Rendered synthesis report text.
        shortlist_ids: Paper IDs that were supposed to inform the synthesis.
    """
    result = rrp_diagnostic_tool(
        RRPDiagnosticInput(
            report_text=report_text,
            shortlist_ids=list(shortlist_ids or []),
        )
    )
    return result


@mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=False,
        openWorldHint=True,
    ),
)
def brief_poll_sources(
    registry_path: str = "",
    workspace: str = "./workspace",
    date: str = "",
    fixture_base_dir: str = "",
) -> ToolResult:
    """Poll configured daily AI intelligence sources.

    This technical-intelligence tool is networked only to registry-allowed
    sources and writes raw plus normalized briefing artifacts locally.
    """
    return brief_poll_sources_tool(
        BriefPollSourcesInput(
            registry_path=registry_path,
            workspace=workspace,
            date=date,
            fixture_base_dir=fixture_base_dir,
        )
    )


@mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    ),
)
def brief_rank_events(
    workspace: str = "./workspace",
    date: str = "",
    registry_path: str = "",
    use_memory: bool = True,
    use_feedback: bool = True,
) -> ToolResult:
    """Deduplicate and rank normalized daily intelligence events locally."""
    return brief_rank_events_tool(
        BriefRankEventsInput(
            workspace=workspace,
            date=date,
            registry_path=registry_path,
            use_memory=use_memory,
            use_feedback=use_feedback,
        )
    )


@mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    ),
)
def brief_generate_daily(workspace: str = "./workspace", date: str = "") -> ToolResult:
    """Generate a template-based daily AI intelligence Markdown brief."""
    return brief_generate_daily_tool(
        BriefGenerateDailyInput(workspace=workspace, date=date)
    )


@mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    ),
)
def brief_validate_report(workspace: str = "./workspace", date: str = "") -> ToolResult:
    """Validate daily brief sections, budgets, duplicate titles, and evidence links."""
    return brief_validate_report_tool(
        BriefValidateReportInput(workspace=workspace, date=date)
    )


@mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=False,
        openWorldHint=True,
    ),
)
def brief_run(
    registry_path: str = "",
    workspace: str = "./workspace",
    date: str = "",
    fixture_base_dir: str = "",
) -> ToolResult:
    """Run poll, rank, generate, and validate for the daily intelligence brief."""
    return brief_run_tool(
        BriefRunInput(
            registry_path=registry_path,
            workspace=workspace,
            date=date,
            fixture_base_dir=fixture_base_dir,
        )
    )


@mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    ),
)
def brief_export_obsidian(
    vault_path: str,
    workspace: str = "./workspace",
    date: str = "",
    registry_path: str = "",
) -> ToolResult:
    """Export daily, topic, and source briefing notes under a configured vault."""
    return brief_export_obsidian_tool(
        BriefExportObsidianInput(
            vault_path=vault_path,
            workspace=workspace,
            date=date,
            registry_path=registry_path,
        )
    )


@mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=False,
        openWorldHint=False,
    ),
)
def brief_record_feedback(
    target_type: str,
    target_id: str,
    signal: str,
    workspace: str = "./workspace",
    date: str = "",
    reason: str = "",
    strength: float = 1.0,
) -> ToolResult:
    """Record explicit local feedback for briefing ranking."""
    return brief_record_feedback_tool(
        BriefRecordFeedbackInput(
            workspace=workspace,
            date=date,
            target_type=target_type,
            target_id=target_id,
            signal=signal,
            reason=reason,
            strength=strength,
        )
    )


@mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    ),
)
def brief_generate_dossier(
    cluster_id: str,
    workspace: str = "./workspace",
    date: str = "",
) -> ToolResult:
    """Generate one manual hot-topic dossier from a ranked briefing cluster."""
    return brief_generate_dossier_tool(
        BriefGenerateDossierInput(
            cluster_id=cluster_id,
            workspace=workspace,
            date=date,
        )
    )


@mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    ),
)
def brief_weekly_synthesis(
    week: str,
    workspace: str = "./workspace",
    output_path: str = "",
) -> ToolResult:
    """Generate a weekly daily-intelligence trend memo from local daily briefs."""
    return brief_weekly_synthesis_tool(
        BriefWeeklySynthesisInput(
            week=week,
            workspace=workspace,
            output_path=output_path,
        )
    )


# ---------------------------------------------------------------------------
# Resources
# ---------------------------------------------------------------------------


@mcp.resource(
    "runs://list",
    name="run_list",
    description="List all pipeline runs with metadata.",
    mime_type="application/json",
)
def resource_run_list() -> str:
    """List all pipeline runs."""
    return resources.list_runs()


@mcp.resource(
    "runs://{run_id}/manifest",
    name="run_manifest",
    description="Run metadata: stages completed, artifacts produced, timing.",
    mime_type="application/json",
)
def resource_run_manifest(run_id: str) -> str:
    """Read a run's manifest."""
    return resources.get_run_manifest(run_id)


@mcp.resource(
    "runs://{run_id}/plan",
    name="query_plan",
    description="Structured query plan with search terms and arXiv categories.",
    mime_type="application/json",
)
def resource_run_plan(run_id: str) -> str:
    """Read a run's query plan."""
    return resources.get_run_plan(run_id)


@mcp.resource(
    "runs://{run_id}/candidates",
    name="candidates",
    description="Search candidates (multi-source paper metadata).",
    mime_type="application/jsonl",
)
def resource_run_candidates(run_id: str) -> str:
    """Read a run's search candidates."""
    return resources.get_run_candidates(run_id)


@mcp.resource(
    "runs://{run_id}/shortlist",
    name="shortlist",
    description="Screened shortlist of relevant papers.",
    mime_type="application/json",
)
def resource_run_shortlist(run_id: str) -> str:
    """Read a run's screened shortlist."""
    return resources.get_run_shortlist(run_id)


@mcp.resource(
    "runs://{run_id}/papers/{paper_id}",
    name="paper_pdf",
    description="Downloaded paper PDF.",
    mime_type="application/pdf",
)
def resource_paper_pdf(run_id: str, paper_id: str) -> bytes:
    """Read a paper's downloaded PDF."""
    return resources.get_paper_pdf(run_id, paper_id)


@mcp.resource(
    "runs://{run_id}/markdown/{paper_id}",
    name="paper_markdown",
    description="Converted paper Markdown (best available tier).",
    mime_type="text/markdown",
)
def resource_paper_markdown(run_id: str, paper_id: str) -> str:
    """Read a paper's converted Markdown."""
    return resources.get_paper_markdown(run_id, paper_id)


@mcp.resource(
    "runs://{run_id}/summary/{paper_id}",
    name="paper_summary",
    description="Per-paper structured summary.",
    mime_type="application/json",
)
def resource_paper_summary(run_id: str, paper_id: str) -> str:
    """Read a paper's summary."""
    return resources.get_paper_summary(run_id, paper_id)


@mcp.resource(
    "runs://{run_id}/synthesis",
    name="synthesis_report",
    description="Cross-paper synthesis report.",
    mime_type="text/markdown",
)
def resource_synthesis_report(run_id: str) -> str:
    """Read a run's synthesis report."""
    return resources.get_synthesis_report(run_id)


@mcp.resource(
    "runs://{run_id}/quality",
    name="quality_scores",
    description="Composite quality evaluation scores.",
    mime_type="application/json",
)
def resource_quality_scores(run_id: str) -> str:
    """Read a run's quality scores."""
    return resources.get_quality_scores(run_id)


@mcp.resource(
    "config://current",
    name="current_config",
    description="Active pipeline configuration (TOML).",
    mime_type="application/toml",
)
def resource_current_config() -> str:
    """Read the current pipeline config."""
    return resources.get_current_config()


@mcp.resource(
    "index://papers",
    name="global_index",
    description="Global paper index for cross-run deduplication.",
    mime_type="application/json",
)
def resource_global_index() -> str:
    """Read the global paper index."""
    return resources.get_global_index()


@mcp.resource(
    "briefings://list",
    name="briefing_list",
    description="List daily AI intelligence briefing runs.",
    mime_type="application/json",
)
def resource_briefing_list() -> str:
    """List daily intelligence briefing runs."""
    return resources.list_briefings()


@mcp.resource(
    "briefings://{date}/daily",
    name="briefing_daily",
    description="Daily AI intelligence Markdown brief.",
    mime_type="text/markdown",
)
def resource_briefing_daily(date: str) -> str:
    """Read a daily intelligence brief."""
    return resources.get_briefing_daily(date)


@mcp.resource(
    "briefings://{date}/ranked",
    name="briefing_ranked_clusters",
    description="Ranked daily intelligence clusters.",
    mime_type="application/jsonl",
)
def resource_briefing_ranked(date: str) -> str:
    """Read ranked briefing clusters."""
    return resources.get_briefing_ranked(date)


@mcp.resource(
    "briefings://{date}/telemetry",
    name="briefing_telemetry",
    description="Daily intelligence telemetry JSONL.",
    mime_type="application/jsonl",
)
def resource_briefing_telemetry(date: str) -> str:
    """Read briefing telemetry."""
    return resources.get_briefing_telemetry(date)


@mcp.resource(
    "briefings://{date}/validation",
    name="briefing_validation",
    description="Daily intelligence validation result.",
    mime_type="application/json",
)
def resource_briefing_validation(date: str) -> str:
    """Read briefing validation."""
    return resources.get_briefing_validation(date)


@mcp.resource(
    "briefings://{date}/state",
    name="briefing_workflow_state",
    description="Replayable daily intelligence workflow state.",
    mime_type="application/json",
)
def resource_briefing_state(date: str) -> str:
    """Read briefing workflow state."""
    return resources.get_briefing_workflow_state(date)


@mcp.resource(
    "workflow://{run_id}/state",
    name="workflow_state",
    description=(
        "Current workflow state including stage statuses, execution log, "
        "iteration count, and content fingerprints."
    ),
    mime_type="application/json",
)
def resource_workflow_state(run_id: str) -> str:
    """Read the workflow state for a run."""
    from research_pipeline.mcp_server.workflow.state import load_state

    state = load_state("./workspace", run_id)
    if state is None:
        return json.dumps({"error": f"No workflow state found for run {run_id}"})
    return state.model_dump_json(indent=2)


@mcp.resource(
    "workflow://{run_id}/telemetry",
    name="workflow_telemetry",
    description=(
        "Workflow telemetry log: three-surface events "
        "(cognitive, operational, contextual) as JSONL."
    ),
    mime_type="application/jsonl",
)
def resource_workflow_telemetry(run_id: str) -> str:
    """Read workflow telemetry for a run."""
    tel_path = Path("./workspace") / run_id / "workflow" / "telemetry.jsonl"
    if not tel_path.exists():
        return json.dumps({"error": f"No telemetry found for run {run_id}"})
    return tel_path.read_text()


@mcp.resource(
    "workflow://{run_id}/budget",
    name="workflow_budget",
    description=(
        "Context budget usage for a workflow run: tokens consumed vs limits "
        "across system, paper, analysis, conversation, and output categories."
    ),
    mime_type="application/json",
)
def resource_workflow_budget(run_id: str) -> str:
    """Read context budget for a run."""
    from research_pipeline.mcp_server.workflow.state import load_state

    state = load_state("./workspace", run_id)
    if state is None:
        return json.dumps({"error": f"No workflow state found for run {run_id}"})
    return state.context_budget.model_dump_json(indent=2)


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------


@mcp.prompt(
    name="research_topic",
    description=(
        "Full research workflow guidance: generates system + user messages "
        "to drive plan → search → screen → download → convert → extract → summarize."
    ),
)
def prompt_research_topic(topic: str) -> list[dict[str, str]]:
    """Start a full research workflow for a topic."""
    return prompts.research_topic_prompt(topic)


@mcp.prompt(
    name="research_workflow",
    description=(
        "Harness-engineered workflow guidance: explains the 6-layer workflow "
        "architecture, sampling-based analysis, elicitation gates, and "
        "iterative synthesis with doom-loop detection."
    ),
)
def prompt_research_workflow(topic: str) -> list[dict[str, str]]:
    """Describe the harness-engineered research workflow."""
    return [
        {
            "role": "user",
            "content": (
                f"Use the research_workflow tool to research: {topic}\n\n"
                "This tool drives a full harness-engineered pipeline:\n"
                "1. Plan: generate query variants from the topic\n"
                "2. Search: query arXiv, Scholar, Semantic Scholar, etc.\n"
                "3. Screen: BM25 + optional SPECTER2 scoring\n"
                "4. Download: rate-limited PDF retrieval\n"
                "5. Convert: PDF→Markdown (multi-backend)\n"
                "6. Extract: chunk and index content\n"
                "7. Summarize: per-paper + cross-paper synthesis\n\n"
                "With sampling support, the server also:\n"
                "- Analyzes papers via LLM (bounded: 1 round per paper)\n"
                "- Synthesizes findings with gap classification\n"
                "- Iterates if system_building=true (max 3 rounds)\n\n"
                "Harness layers: telemetry, context budget, governance, "
                "structural verification, doom-loop monitoring, recovery.\n\n"
                "Set system_building=true for iterative synthesis with "
                "gap analysis and convergence detection."
            ),
        }
    ]


@mcp.prompt(
    name="analyze_paper",
    description=(
        "Analyze a specific converted paper: loads the markdown and creates "
        "a prompt for methodology, findings, and limitations analysis."
    ),
)
def prompt_analyze_paper(run_id: str, paper_id: str) -> list[dict[str, str]]:
    """Analyze a specific paper in a run."""
    return prompts.analyze_paper_prompt(run_id, paper_id)


@mcp.prompt(
    name="compare_papers",
    description=(
        "Compare all papers in a run: creates a comparative analysis prompt "
        "covering themes, contradictions, gaps, and rankings."
    ),
)
def prompt_compare_papers(run_id: str) -> list[dict[str, str]]:
    """Compare all papers in a run."""
    return prompts.compare_papers_prompt(run_id)


@mcp.prompt(
    name="refine_search",
    description=(
        "Refine search based on current results: analyzes candidates/shortlist "
        "and suggests improved query terms."
    ),
)
def prompt_refine_search(run_id: str) -> list[dict[str, str]]:
    """Refine search for a run."""
    return prompts.refine_search_prompt(run_id)


@mcp.prompt(
    name="quality_assessment",
    description=(
        "Assess paper quality: interprets quality evaluation scores "
        "and recommends papers to prioritize."
    ),
)
def prompt_quality_assessment(run_id: str) -> list[dict[str, str]]:
    """Assess quality scores for a run."""
    return prompts.quality_assessment_prompt(run_id)


# ---------------------------------------------------------------------------
# Completions
# ---------------------------------------------------------------------------


@mcp.completion()
async def handle_completion(ref, argument, context=None):  # type: ignore[no-untyped-def]
    """Auto-complete arguments for resource templates and prompts."""
    return await completions.handle_completion(ref, argument, context)


# Prune tools to the operator-selected capability domains (#46). Runs after
# every @mcp.tool() above is registered and before the guard, so the guard
# only registers the tools that remain active.
_active_toolsets = toolsets.apply_toolsets(mcp)

# Wire the zero-trust guard into tool dispatch (#45). Must run after every
# @mcp.tool() above has been registered so the registry is complete.
_guard = guard_wiring.install_guard(mcp)


if __name__ == "__main__":
    mcp.run()
