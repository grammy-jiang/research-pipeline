"""MCP server entry point for research-pipeline.

Exposes pipeline stages as MCP tools via stdio transport.
Run with: python -m mcp_server.server
"""

from __future__ import annotations

import logging

from mcp.server.fastmcp import Context, FastMCP
from mcp.types import ToolAnnotations

from mcp_server import completions, prompts, resources
from mcp_server.schemas import (
    ConvertFileInput,
    ConvertFineInput,
    ConvertPdfsInput,
    ConvertRoughInput,
    DownloadPdfsInput,
    EvaluateQualityInput,
    ExpandCitationsInput,
    ExtractContentInput,
    GetRunManifestInput,
    ListBackendsInput,
    ManageIndexInput,
    PlanTopicInput,
    RunPipelineInput,
    ScreenCandidatesInput,
    SearchInput,
    SummarizePapersInput,
)
from mcp_server.tools import (
    convert_file,
    convert_fine,
    convert_pdfs,
    convert_rough,
    download_pdfs,
    evaluate_quality,
    expand_citations,
    extract_content,
    get_run_manifest,
    list_backends,
    manage_index,
    plan_topic,
    run_pipeline,
    screen_candidates,
    search,
    summarize_papers,
)

logger = logging.getLogger(__name__)

mcp = FastMCP(
    "research-pipeline",
    instructions=(
        "MCP server for academic paper research: search multiple sources, "
        "screen, download, convert, extract, summarize papers."
    ),
)


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
) -> dict:
    """Create a structured query plan from a natural language research topic.

    Normalizes the topic, generates query variants and candidate arXiv
    categories. This is the first step in the pipeline.
    """
    result = plan_topic(
        PlanTopicInput(topic=topic, workspace=workspace, run_id=run_id), ctx=ctx
    )
    return result.model_dump()


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
) -> dict:
    """Search arXiv and/or Google Scholar for academic papers.

    Queries enabled sources with rate limiting, parses responses,
    and deduplicates across sources and query variants.
    Use source='arxiv', 'scholar', 'all', or '' (config default).
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
    return result.model_dump()


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
) -> dict:
    """Two-stage relevance screening: cheap BM25 scoring then shortlist selection.

    Reads candidates from the search stage, scores them, and produces
    a shortlist of the most relevant papers.
    """
    result = screen_candidates(
        ScreenCandidatesInput(workspace=workspace, run_id=run_id, resume=resume),
        ctx=ctx,
    )
    return result.model_dump()


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
) -> dict:
    """Download shortlisted PDFs from arXiv with rate-limit compliance.

    Respects arXiv's 3-second rate limit. Downloads are idempotent
    (skips existing files unless force=True).
    """
    result = download_pdfs(
        DownloadPdfsInput(workspace=workspace, run_id=run_id, force=force), ctx=ctx
    )
    return result.model_dump()


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
) -> dict:
    """Convert downloaded PDFs to Markdown.

    Supports multiple backends: docling, marker, pymupdf4llm.
    Use backend='' to use the config default.
    Requires the corresponding extra to be installed.
    """
    result = convert_pdfs(
        ConvertPdfsInput(
            workspace=workspace, run_id=run_id, force=force, backend=backend
        ),
        ctx=ctx,
    )
    return result.model_dump()


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
) -> dict:
    """Extract structured content (chunks, sections) from converted Markdown.

    Performs chunking and indexing for downstream summarization.
    """
    result = extract_content(
        ExtractContentInput(workspace=workspace, run_id=run_id), ctx=ctx
    )
    return result.model_dump()


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
) -> dict:
    """Generate per-paper summaries and cross-paper synthesis.

    Produces evidence-backed summaries with chunk citations,
    plus a synthesis report comparing findings across papers.
    """
    result = summarize_papers(
        SummarizePapersInput(workspace=workspace, run_id=run_id), ctx=ctx
    )
    return result.model_dump()


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
) -> dict:
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
    return result.model_dump()


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
) -> dict:
    """Inspect a run's manifest: stages completed, artifacts produced, timing.

    Use this to check the status of a pipeline run.
    """
    result = get_run_manifest(
        GetRunManifestInput(workspace=workspace, run_id=run_id), ctx=ctx
    )
    return result.model_dump()


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
) -> dict:
    """Convert a single PDF file to Markdown (standalone, no pipeline workspace needed).

    Supports multiple backends: docling, marker, pymupdf4llm.
    Use backend='' to use the config default. Useful for ad-hoc
    document conversion without running the full pipeline.
    """
    result = convert_file(
        ConvertFileInput(pdf_path=pdf_path, output_dir=output_dir, backend=backend),
        ctx=ctx,
    )
    return result.model_dump()


@mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    ),
)
def tool_list_backends(ctx: Context) -> dict:
    """List available PDF-to-Markdown converter backends.

    Returns the names of all registered backends (docling, marker, pymupdf4llm).
    Each backend requires its corresponding extra to be installed.
    """
    result = list_backends(ListBackendsInput(), ctx=ctx)
    return result.model_dump()


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
) -> dict:
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
    return result.model_dump()


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
) -> dict:
    """Compute composite quality scores for candidate papers.

    Evaluates papers on citation impact, venue reputation (CORE rankings),
    author h-index credibility, and recency. Requires a completed search
    or screen stage.
    """
    result = evaluate_quality(
        EvaluateQualityInput(workspace=workspace, run_id=run_id), ctx=ctx
    )
    return result.model_dump()


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
) -> dict:
    """Fast Tier 2 conversion of all downloaded PDFs using pymupdf4llm.

    CPU-only, fast conversion for all papers. The agent reads rough
    markdown to decide which papers need fine conversion.
    Requires a completed download stage.
    """
    result = convert_rough(
        ConvertRoughInput(workspace=workspace, run_id=run_id, force=force), ctx=ctx
    )
    return result.model_dump()


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
) -> dict:
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
    return result.model_dump()


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
) -> dict:
    """Manage the global paper index for incremental runs.

    Browse indexed papers (list_papers=true) or clean stale entries
    (gc=true). The global index deduplicates papers across runs.
    """
    result = manage_index(
        ManageIndexInput(list_papers=list_papers, gc=gc, db_path=db_path), ctx=ctx
    )
    return result.model_dump()


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


if __name__ == "__main__":
    mcp.run()
