"""MCP server entry point for arxiv-paper-pipeline.

Exposes pipeline stages as MCP tools via stdio transport.
Run with: python -m mcp_server.server
"""

from __future__ import annotations

import logging

from mcp.server.fastmcp import FastMCP

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
)
from mcp_server.tools import (
    convert_file,
    convert_pdfs,
    download_pdfs,
    extract_content,
    get_run_manifest,
    plan_topic,
    run_pipeline,
    screen_candidates,
    search_arxiv,
    summarize_papers,
)

logger = logging.getLogger(__name__)

mcp = FastMCP(
    "arxiv-paper-pipeline",
    instructions=(
        "MCP server for arXiv paper research: search, screen, download, "
        "convert, extract, summarize papers."
    ),
)


@mcp.tool()
def tool_plan_topic(
    topic: str,
    workspace: str = "./workspace",
    run_id: str = "",
) -> dict:
    """Create a structured query plan from a natural language research topic.

    Normalizes the topic, generates query variants and candidate arXiv
    categories. This is the first step in the pipeline.
    """
    result = plan_topic(PlanTopicInput(topic=topic, workspace=workspace, run_id=run_id))
    return result.model_dump()


@mcp.tool()
def tool_search_arxiv(
    workspace: str = "./workspace",
    run_id: str = "",
    topic: str = "",
    resume: bool = False,
) -> dict:
    """Search arXiv using the query plan and return deduplicated candidates.

    Executes API queries with rate limiting, parses Atom responses,
    and deduplicates across query variants.
    """
    result = search_arxiv(
        SearchArxivInput(workspace=workspace, run_id=run_id, topic=topic, resume=resume)
    )
    return result.model_dump()


@mcp.tool()
def tool_screen_candidates(
    workspace: str = "./workspace",
    run_id: str = "",
    resume: bool = False,
) -> dict:
    """Two-stage relevance screening: cheap BM25 scoring then shortlist selection.

    Reads candidates from the search stage, scores them, and produces
    a shortlist of the most relevant papers.
    """
    result = screen_candidates(
        ScreenCandidatesInput(workspace=workspace, run_id=run_id, resume=resume)
    )
    return result.model_dump()


@mcp.tool()
def tool_download_pdfs(
    workspace: str = "./workspace",
    run_id: str = "",
    force: bool = False,
) -> dict:
    """Download shortlisted PDFs from arXiv with rate-limit compliance.

    Respects arXiv's 3-second rate limit. Downloads are idempotent
    (skips existing files unless force=True).
    """
    result = download_pdfs(
        DownloadPdfsInput(workspace=workspace, run_id=run_id, force=force)
    )
    return result.model_dump()


@mcp.tool()
def tool_convert_pdfs(
    workspace: str = "./workspace",
    run_id: str = "",
    force: bool = False,
) -> dict:
    """Convert downloaded PDFs to Markdown using Docling.

    Requires the docling extra: pip install 'arxiv-paper-pipeline[docling]'.
    """
    result = convert_pdfs(
        ConvertPdfsInput(workspace=workspace, run_id=run_id, force=force)
    )
    return result.model_dump()


@mcp.tool()
def tool_extract_content(
    workspace: str = "./workspace",
    run_id: str = "",
) -> dict:
    """Extract structured content (chunks, sections) from converted Markdown.

    Performs chunking and indexing for downstream summarization.
    """
    result = extract_content(ExtractContentInput(workspace=workspace, run_id=run_id))
    return result.model_dump()


@mcp.tool()
def tool_summarize_papers(
    workspace: str = "./workspace",
    run_id: str = "",
) -> dict:
    """Generate per-paper summaries and cross-paper synthesis.

    Produces evidence-backed summaries with chunk citations,
    plus a synthesis report comparing findings across papers.
    """
    result = summarize_papers(SummarizePapersInput(workspace=workspace, run_id=run_id))
    return result.model_dump()


@mcp.tool()
def tool_run_pipeline(
    topic: str,
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
        RunPipelineInput(topic=topic, workspace=workspace, run_id=run_id, resume=resume)
    )
    return result.model_dump()


@mcp.tool()
def tool_get_run_manifest(
    workspace: str = "./workspace",
    run_id: str = "",
) -> dict:
    """Inspect a run's manifest: stages completed, artifacts produced, timing.

    Use this to check the status of a pipeline run.
    """
    result = get_run_manifest(GetRunManifestInput(workspace=workspace, run_id=run_id))
    return result.model_dump()


@mcp.tool()
def tool_convert_file(
    pdf_path: str,
    output_dir: str = "",
) -> dict:
    """Convert a single PDF file to Markdown (standalone, no pipeline workspace needed).

    Uses Docling to convert any PDF to Markdown. Useful for ad-hoc
    document conversion without running the full pipeline.
    """
    result = convert_file(ConvertFileInput(pdf_path=pdf_path, output_dir=output_dir))
    return result.model_dump()


if __name__ == "__main__":
    mcp.run()
