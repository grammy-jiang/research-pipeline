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


class ToolResult(BaseModel):
    """Standard result envelope for MCP tool outputs."""

    success: bool = Field(description="Whether the operation succeeded.")
    message: str = Field(description="Human-readable summary of the result.")
    artifacts: dict = Field(
        default_factory=dict,
        description="Paths to generated artifacts or structured data.",
    )
