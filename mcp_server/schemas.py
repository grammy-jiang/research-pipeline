"""Pydantic schemas for MCP tool inputs and outputs."""

from __future__ import annotations

from pydantic import BaseModel, Field


class CommonParams(BaseModel):
    """Parameters shared by all MCP tools."""

    workspace: str = Field(
        default="./workspace",
        description="Path to the workspace directory for artifacts.",
    )
    run_id: str = Field(
        default="",
        description="Run identifier. Empty string means auto-generate.",
    )


class PlanTopicInput(CommonParams):
    """Input for the plan_topic tool."""

    topic: str = Field(description="Natural language research topic.")


class SearchInput(CommonParams):
    """Input for the search tool."""

    topic: str = Field(
        default="",
        description="Topic to search. If empty, reads from existing query plan.",
    )
    resume: bool = Field(
        default=False,
        description="Resume from existing search results if available.",
    )
    source: str = Field(
        default="",
        description=(
            "Search source(s): 'arxiv', 'scholar', 'semantic_scholar', "
            "'openalex', 'dblp', 'huggingface', 'all', or '' "
            "(empty = use config default)."
        ),
    )


class ScreenCandidatesInput(CommonParams):
    """Input for the screen_candidates tool."""

    resume: bool = Field(
        default=False,
        description="Resume from existing screening results.",
    )


class DownloadPdfsInput(CommonParams):
    """Input for the download_pdfs tool."""

    force: bool = Field(
        default=False,
        description="Re-download even if PDFs already exist.",
    )


class ConvertPdfsInput(CommonParams):
    """Input for the convert_pdfs tool."""

    force: bool = Field(
        default=False,
        description="Re-convert even if Markdown files already exist.",
    )
    backend: str = Field(
        default="",
        description=(
            "Converter backend: 'docling', 'marker', 'pymupdf4llm', "
            "or '' (use config default)."
        ),
    )


class ExtractContentInput(CommonParams):
    """Input for the extract_content tool."""


class SummarizePapersInput(CommonParams):
    """Input for the summarize_papers tool."""


class RunPipelineInput(CommonParams):
    """Input for the run_pipeline tool."""

    topic: str = Field(description="Natural language research topic.")
    resume: bool = Field(
        default=False,
        description="Resume from existing results where possible.",
    )


class GetRunManifestInput(BaseModel):
    """Input for the get_run_manifest tool."""

    workspace: str = Field(
        default="./workspace",
        description="Path to the workspace directory.",
    )
    run_id: str = Field(
        default="",
        description="Run identifier to inspect. Empty means latest.",
    )


class ConvertFileInput(BaseModel):
    """Input for the convert_file tool (standalone single-file conversion)."""

    pdf_path: str = Field(description="Path to a PDF file to convert.")
    output_dir: str = Field(
        default="",
        description="Output directory for Markdown. Default: same dir as PDF.",
    )
    backend: str = Field(
        default="",
        description=(
            "Converter backend: 'docling', 'marker', 'pymupdf4llm', "
            "or '' (use config default)."
        ),
    )


class ListBackendsInput(BaseModel):
    """Input for the list_backends tool."""

    # No parameters needed; included for consistency.


class ExpandCitationsInput(CommonParams):
    """Input for the expand_citations tool."""

    paper_ids: list[str] = Field(
        description="arXiv IDs or Semantic Scholar paper IDs to expand.",
    )
    direction: str = Field(
        default="both",
        description="Expansion direction: 'citations', 'references', or 'both'.",
    )
    limit: int = Field(
        default=50,
        description="Max related papers per seed paper per direction.",
    )
    snowball: bool = Field(
        default=False,
        description="Enable bidirectional snowball expansion with "
        "budget-aware stopping.",
    )
    snowball_max_rounds: int = Field(
        default=5,
        description="Max snowball iteration rounds.",
    )
    snowball_max_papers: int = Field(
        default=200,
        description="Hard cap on total discovered papers.",
    )
    query_terms: list[str] = Field(
        default_factory=list,
        description="Query terms for snowball relevance scoring.",
    )


class EvaluateQualityInput(CommonParams):
    """Input for the evaluate_quality tool."""


class ConvertRoughInput(CommonParams):
    """Input for the convert_rough tool."""

    force: bool = Field(
        default=False,
        description="Re-convert even if Markdown files already exist.",
    )


class ConvertFineInput(CommonParams):
    """Input for the convert_fine tool."""

    paper_ids: list[str] = Field(
        description="arXiv IDs of papers to fine-convert.",
    )
    force: bool = Field(
        default=False,
        description="Re-convert even if Markdown files already exist.",
    )
    backend: str = Field(
        default="",
        description=(
            "Converter backend override: 'docling', 'marker', 'pymupdf4llm', "
            "or '' (use config default)."
        ),
    )


class ManageIndexInput(BaseModel):
    """Input for the manage_index tool."""

    list_papers: bool = Field(
        default=False,
        description="List indexed papers (up to 100).",
    )
    gc: bool = Field(
        default=False,
        description="Garbage collect stale entries.",
    )
    db_path: str = Field(
        default="",
        description="Path to index database. Empty uses default.",
    )


class AnalyzePapersInput(CommonParams):
    """Input for the analyze_papers tool."""

    collect: bool = Field(
        default=False,
        description=(
            "If True, validate collected analysis JSON files "
            "instead of generating prompts."
        ),
    )
    paper_ids: list[str] = Field(
        default_factory=list,
        description="Optional list of specific paper arXiv IDs to analyze.",
    )


class ValidateReportInput(BaseModel):
    """Input for the validate_report tool."""

    report_path: str = Field(
        default="",
        description=(
            "Path to the report markdown file. If empty, uses run_id to find synthesis."
        ),
    )
    workspace: str = Field(
        default="./workspace",
        description="Path to the workspace directory.",
    )
    run_id: str = Field(
        default="",
        description=(
            "Run ID to find synthesis_report.md (used if report_path is empty)."
        ),
    )


class CompareRunsInput(BaseModel):
    """Input for the compare_runs tool."""

    workspace: str = Field(
        default="./workspace",
        description="Path to the workspace directory.",
    )
    run_id_a: str = Field(
        description="First run ID (baseline).",
    )
    run_id_b: str = Field(
        description="Second run ID (latest).",
    )


class VerifyStageInput(CommonParams):
    """Input for the verify_stage tool."""

    stage: str = Field(
        description=(
            "Stage to verify: 'plan', 'search', 'screen', 'download', "
            "'convert', 'extract', 'summarize'."
        ),
    )


class FeedbackInput(CommonParams):
    """Input for the record_feedback tool."""

    accept: list[str] = Field(
        default_factory=list,
        description="Paper IDs to mark as accepted.",
    )
    reject: list[str] = Field(
        default_factory=list,
        description="Paper IDs to mark as rejected.",
    )
    reason: str = Field(
        default="",
        description="Optional reason for the decisions.",
    )
    show: bool = Field(
        default=False,
        description="Show current feedback stats.",
    )
    adjust: bool = Field(
        default=False,
        description="Recompute adjusted BM25 weights from feedback.",
    )


class EvalLogInput(CommonParams):
    """Input for the query_eval_log tool."""

    channel: str = Field(
        default="all",
        description=("Channel to query: traces, audit, snapshots, summary, or all."),
    )
    stage: str = Field(
        default="",
        description="Filter by pipeline stage.",
    )
    limit: int = Field(
        default=50,
        description="Maximum records to return.",
    )


class EvidenceAggregateInput(CommonParams):
    """Input for the aggregate_evidence tool."""

    min_pointers: int = Field(
        default=0,
        description="Minimum evidence pointers per statement.",
    )
    max_words: int = Field(
        default=50,
        description="Maximum words per statement.",
    )
    similarity_threshold: float = Field(
        default=0.7,
        description="Threshold for merging similar statements (0-1).",
    )
    strip_rhetoric: bool = Field(
        default=True,
        description="Whether to strip rhetoric from statements.",
    )
    output_format: str = Field(
        default="text",
        description="Output format: text or json.",
    )


class ExportHtmlInput(CommonParams):
    """Input for the export_html tool."""

    markdown_file: str = Field(
        default="",
        description="Path to Markdown report (alternative to run_id).",
    )
    title: str = Field(
        default="Research Report",
        description="Report title (used with markdown_file mode).",
    )
    output: str = Field(
        default="",
        description="Output HTML file path (default: auto).",
    )


class ModelRoutingInfoInput(BaseModel):
    """Input for the model_routing_info tool."""

    config_path: str = Field(
        default="",
        description="Path to config.toml. Empty uses defaults.",
    )


class GateInfoInput(BaseModel):
    """Input for the gate_info tool."""

    config_path: str = Field(
        default="",
        description="Path to config.toml. Empty uses defaults.",
    )


class CoherenceInput(BaseModel):
    """Input for the coherence evaluation tool."""

    run_ids: list[str] = Field(
        description="Two or more run IDs to evaluate coherence across.",
        min_length=2,
    )
    workspace: str = Field(
        default="runs",
        description="Workspace directory containing run outputs.",
    )


class ConsolidationInput(BaseModel):
    """Input for the memory consolidation tool."""

    run_ids: list[str] | None = Field(
        default=None,
        description="Run IDs to ingest. If None, scans workspace.",
    )
    workspace: str = Field(
        default="runs",
        description="Workspace directory containing run outputs.",
    )
    dry_run: bool = Field(
        default=False,
        description="Compute metrics without modifying store.",
    )
    capacity: int = Field(
        default=100,
        description="Episode capacity before triggering consolidation.",
    )
    threshold: float = Field(
        default=0.8,
        description="Fraction of capacity triggering consolidation.",
    )
    min_support: int = Field(
        default=2,
        description="Min run appearances for rule promotion.",
    )


class BlindingAuditInput(BaseModel):
    """Input for the epistemic blinding audit tool."""

    workspace: str = Field(
        default="workspace",
        description="Workspace directory containing run outputs.",
    )
    run_id: str = Field(
        default="",
        description="Specific run ID to audit (latest if empty).",
    )
    threshold: float = Field(
        default=0.4,
        description="Contamination threshold for flagging papers.",
    )
    store_results: bool = Field(
        default=True,
        description="Whether to persist results to SQLite.",
    )


class DualMetricsInput(BaseModel):
    """Input for the dual-metrics evaluation tool."""

    workspace: str = Field(
        default="workspace",
        description="Workspace directory containing run outputs.",
    )
    query: str = Field(
        description="Research query these runs address.",
    )
    run_ids: list[str] = Field(
        default_factory=list,
        description="Run IDs to evaluate (auto-discover if empty).",
    )
    k: int = Field(
        default=5,
        description="Number of samples for Pass@k / Pass[k] computation.",
    )
    store_results: bool = Field(
        default=True,
        description="Whether to persist results to SQLite.",
    )


class CbrLookupInput(BaseModel):
    """Input for the CBR lookup tool."""

    workspace: str = Field(
        default="workspace",
        description="Workspace directory.",
    )
    topic: str = Field(
        description="Research topic to look up similar past cases.",
    )
    max_results: int = Field(
        default=5,
        description="Maximum number of similar cases to retrieve.",
    )
    min_quality: float = Field(
        default=0.0,
        description="Minimum synthesis quality to consider.",
    )


class CbrRetainInput(BaseModel):
    """Input for the CBR retain tool."""

    workspace: str = Field(
        default="workspace",
        description="Workspace directory.",
    )
    run_id: str = Field(
        description="Pipeline run ID to store as a case.",
    )
    topic: str = Field(
        description="Research topic for this run.",
    )
    outcome: str = Field(
        default="unknown",
        description="Quality outcome: excellent, good, adequate, poor, failed.",
    )
    strategy_notes: str = Field(
        default="",
        description="Free-text notes about the strategy used.",
    )


class KGQualityInput(BaseModel):
    """Input for the kg_quality tool.

    Evaluates knowledge graph quality across 5 dimensions using the
    three-layer composable architecture (TKDE 2022 + Text2KGBench).
    """

    db_path: str = Field(
        default="",
        description="Path to KG SQLite database. Empty uses default.",
    )
    staleness_days: float = Field(
        default=365.0,
        description="Threshold in days for a triple to be considered stale.",
    )
    sample_size: int = Field(
        default=0,
        description="If > 0, also run TWCS sampling and return sample.",
    )


class AdaptiveStoppingInput(BaseModel):
    """Input for the adaptive_stopping tool.

    Evaluates query-adaptive retrieval stopping criteria based on
    HingeMem (WWW '26) three stopping strategies: knee detection
    (recall), top-k saturation (precision), top-1 stability (judgment).
    """

    batch_scores: list[list[float]] = Field(
        description="List of score lists, one per retrieval batch.",
    )
    query: str = Field(
        default="",
        description="Original query for auto-classifying stopping strategy.",
    )
    query_type: str = Field(
        default="auto",
        description="Query type: recall, precision, judgment, or auto.",
    )
    min_results: int = Field(
        default=5,
        description="Minimum results before stopping is considered.",
    )
    max_budget: int = Field(
        default=500,
        description="Hard budget limit on total results.",
    )
    relevance_threshold: float = Field(
        default=0.5,
        description="Score threshold for a result to count as relevant.",
    )


class ConfidenceLayersInput(BaseModel):
    """Input for the confidence_layers tool.

    Scores claims through the 4-layer confidence architecture:
    L1 (fast signal) → L2 (adaptive granularity) → L3 (DINCO calibration)
    → L4 (selective verification for low-confidence claims).

    Based on Atomic Calibration, AGSC, DINCO, and LoVeC research.
    """

    run_id: str = Field(description="Run ID containing claim decompositions.")
    config_path: str = Field(
        default="",
        description="Path to config.toml. Empty uses defaults.",
    )
    workspace: str = Field(
        default="",
        description="Workspace directory. Empty uses config default.",
    )
    l4_threshold: float = Field(
        default=0.50,
        description="Confidence below which L4 verification triggers.",
    )
    damping: float = Field(
        default=0.80,
        description="Fusion damping exponent (0-1). Lower = more conservative.",
    )
    calibrate: bool = Field(
        default=False,
        description="Whether to fit Platt scaling from prior scored claims.",
    )


class ExportBibtexInput(CommonParams):
    """Input for the export_bibtex tool."""

    stage: str = Field(
        default="screen",
        description="Stage to export from: search, screen, or download.",
    )
    output: str = Field(
        default="",
        description="Output .bib file path (default: auto in run dir).",
    )


class ReportInput(CommonParams):
    """Input for the report tool."""

    template: str = Field(
        default="survey",
        description=(
            "Report template: survey, gap_analysis, lit_review, or executive."
        ),
    )
    custom_template: str = Field(
        default="",
        description="Path to a custom Jinja2 template file.",
    )
    output: str = Field(
        default="",
        description="Output Markdown file path (default: auto in run dir).",
    )


class ClusterInput(CommonParams):
    """Input for the cluster tool."""

    stage: str = Field(
        default="screen",
        description="Stage to cluster from: search or screen.",
    )
    threshold: float = Field(
        default=0.15,
        description=(
            "Cosine similarity threshold (0-1). Lower = fewer, larger clusters."
        ),
    )
    output: str = Field(
        default="",
        description="Output JSON file path (default: auto in run dir).",
    )


class EnrichInput(CommonParams):
    """Input for the enrich tool."""

    stage: str = Field(
        default="candidates",
        description="Stage to read candidates from: 'candidates' or 'screened'.",
    )
    config_path: str = Field(
        default="",
        description="Path to config.toml. Empty uses defaults.",
    )


class CiteContextInput(CommonParams):
    """Input for the cite_context tool."""

    window: int = Field(
        default=1,
        description=(
            "Extra sentences before/after citation (0 = citing sentence only)."
        ),
    )
    output: str = Field(
        default="",
        description=(
            "Output JSON file path (default: <convert_dir>/citation_contexts.json)."
        ),
    )
    config_path: str = Field(
        default="",
        description="Path to config.toml. Empty uses defaults.",
    )


class WatchInput(BaseModel):
    """Input for the watch tool."""

    queries: str = Field(
        default="",
        description=(
            "Path to a JSON file with watch queries. "
            "Empty uses default (~/.cache/research-pipeline/watch/watch_queries.json)."
        ),
    )
    lookback: int = Field(
        default=7,
        description="Days to look back for new papers.",
    )
    max_results: int = Field(
        default=20,
        description="Maximum results per query.",
    )
    output: str = Field(
        default="",
        description="Output JSON file path for new papers found.",
    )
    config_path: str = Field(
        default="",
        description="Path to config.toml. Empty uses defaults.",
    )


class ResearchWorkflowInput(CommonParams):
    """Input for the research_workflow tool.

    Drives the full harness-engineered research workflow with
    sampling, elicitation, verification, and doom-loop detection.
    """

    topic: str = Field(description="Natural language research topic.")
    config_path: str = Field(
        default="",
        description="Path to config.toml. Empty uses defaults.",
    )
    system_building: bool = Field(
        default=False,
        description=(
            "Enable system-building mode with iterative synthesis, "
            "gap analysis, and convergence loops."
        ),
    )
    source: str = Field(
        default="",
        description=(
            "Search sources: 'arxiv', 'scholar', 'all', or '' for config default."
        ),
    )
    max_iterations: int = Field(
        default=3,
        description="Maximum synthesis iterations (system-building mode).",
    )
    resume: bool = Field(
        default=False,
        description="Resume from last saved state instead of starting fresh.",
    )


class AnalyzeClaimsInput(CommonParams):
    """Input for the analyze_claims tool.

    Decomposes paper summaries into atomic claims with evidence
    classification (supported, partial, conflicting, inconclusive,
    unsupported) using BM25 retrieval against source Markdown.
    """


class ScoreClaimsInput(CommonParams):
    """Input for the score_claims tool.

    Scores confidence for decomposed claims using evidence strength,
    hedging-language detection (LVU), citation density, and retrieval
    quality. Optionally uses LLM multi-sample consistency when available.
    """


class KGStatsInput(BaseModel):
    """Input for the kg_stats tool."""

    db_path: str = Field(
        default="",
        description="Path to KG SQLite database. Empty uses default.",
    )


class KGQueryInput(BaseModel):
    """Input for the kg_query tool."""

    entity_id: str = Field(
        description="Entity identifier to look up in the knowledge graph.",
    )
    db_path: str = Field(
        default="",
        description="Path to KG SQLite database. Empty uses default.",
    )


class KGIngestInput(CommonParams):
    """Input for the kg_ingest tool.

    Reads candidates and claim decompositions from a completed run
    and populates the knowledge graph with entities and relations.
    """

    db_path: str = Field(
        default="",
        description="Path to KG SQLite database. Empty uses default.",
    )


class MemoryStatsInput(BaseModel):
    """Input for the memory_stats tool."""

    episodic_db: str = Field(
        default="",
        description="Path to episodic memory database. Empty uses default.",
    )
    kg_db: str = Field(
        default="",
        description="Path to KG database. Empty uses default.",
    )


class MemoryEpisodesInput(BaseModel):
    """Input for the memory_episodes tool."""

    limit: int = Field(
        default=10,
        description="Maximum number of episodes to return.",
    )
    episodic_db: str = Field(
        default="",
        description="Path to episodic memory database. Empty uses default.",
    )


class MemorySearchInput(BaseModel):
    """Input for the memory_search tool."""

    topic: str = Field(
        description="Topic to search in episodic memory.",
    )
    limit: int = Field(
        default=10,
        description="Maximum number of results to return.",
    )
    episodic_db: str = Field(
        default="",
        description="Path to episodic memory database. Empty uses default.",
    )


class EvaluateInput(BaseModel):
    """Input for the evaluate tool.

    Validates pipeline outputs against their Pydantic schemas,
    checking completeness, score ranges, and cross-field consistency.
    """

    run_id: str = Field(
        description="Run ID to evaluate.",
    )
    stage: str = Field(
        default="",
        description="Specific stage to evaluate. Empty means all stages.",
    )
    workspace: str = Field(
        default="runs",
        description="Workspace directory containing run outputs.",
    )


class HorizonMetricInput(BaseModel):
    """Input for the ``horizon_metric`` tool (A3-5 Unified Horizon Metric)."""

    normalized_score: float = Field(description="Normalized task quality in [0, 1].")
    difficulty: float = Field(default=0.5, description="Task difficulty in [0, 1].")
    achieved_steps: int = Field(description="Trajectory length actually completed.")
    target_steps: int = Field(description="Benchmark target horizon.")
    entropy_trend: float = Field(
        default=0.0,
        description="Token-entropy slope across trajectory (neg=locking).",
    )
    reliability: float = Field(
        default=1.0,
        description="Optional Pass[k] reliability floor in [0, 1].",
    )


class RRPDiagnosticInput(BaseModel):
    """Input for the ``rrp_diagnostic`` tool (Theme 16 R/R/P)."""

    report_text: str = Field(description="Rendered synthesis report text.")
    shortlist_ids: list[str] = Field(
        default_factory=list,
        description="Paper IDs that were supposed to inform the synthesis.",
    )


class ToolResult(BaseModel):
    """Standard result envelope for MCP tool outputs."""

    success: bool = Field(description="Whether the operation succeeded.")
    message: str = Field(description="Human-readable summary of the result.")
    artifacts: dict = Field(
        default_factory=dict,
        description="Paths to generated artifacts or structured data.",
    )
