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
            "Search source(s): 'arxiv', 'scholar', 'all', or ''"
            " (empty = use config default)."
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
            "Path to the report markdown file. "
            "If empty, uses run_id to find synthesis."
        ),
    )
    workspace: str = Field(
        default="./workspace",
        description="Path to the workspace directory.",
    )
    run_id: str = Field(
        default="",
        description=(
            "Run ID to find synthesis_report.md " "(used if report_path is empty)."
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
        description=("Channel to query: traces, audit, snapshots, " "summary, or all."),
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
            "Search sources: 'arxiv', 'scholar', 'all', " "or '' for config default."
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


class ToolResult(BaseModel):
    """Standard result envelope for MCP tool outputs."""

    success: bool = Field(description="Whether the operation succeeded.")
    message: str = Field(description="Human-readable summary of the result.")
    artifacts: dict = Field(
        default_factory=dict,
        description="Paths to generated artifacts or structured data.",
    )
