"""CLI root application and subcommand registration."""

import logging
from pathlib import Path

import typer

from research_pipeline import __version__
from research_pipeline.infra.logging import setup_logging

app = typer.Typer(
    name="research-pipeline",
    help=(
        "Multi-source academic paper research pipeline.\n\n"
        "Stages: plan → search → screen → download → convert → "
        "extract → summarize.\n\n"
        "Quick start:  research-pipeline run 'your research topic'\n\n"
        "Step-by-step:  research-pipeline plan 'topic'  then use "
        "--run-id for each subsequent stage."
    ),
    no_args_is_help=True,
)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"research-pipeline {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        "-V",
        help="Show version and exit.",
        callback=_version_callback,
        is_eager=True,
    ),
) -> None:
    """Multi-source academic paper research pipeline."""


def _common_options(
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable debug logging."
    ),
    config: Path | None = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to config TOML file (default: ~/.research-pipeline.toml).",
    ),
    workspace: Path | None = typer.Option(
        None,
        "--workspace",
        "-w",
        help="Workspace root directory for storing runs.",
    ),
    run_id: str | None = typer.Option(
        None,
        "--run-id",
        help="Run ID. Auto-generated if not provided.",
    ),
) -> dict:  # type: ignore[type-arg]
    """Parse common options shared by all commands."""
    level = logging.DEBUG if verbose else logging.INFO
    setup_logging(level=level)
    return {
        "config_path": config,
        "workspace": workspace,
        "run_id": run_id,
    }


@app.command()
def plan(
    topic: str = typer.Argument(..., help="Research topic (natural language)."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
    config: Path | None = typer.Option(None, "--config", "-c"),
    workspace: Path | None = typer.Option(None, "--workspace", "-w"),
    run_id: str | None = typer.Option(None, "--run-id"),
) -> None:
    """Normalize a topic into a structured query plan.

    Creates a query plan with must/nice/negative terms for arXiv and
    Google Scholar searches. Outputs query_plan.json in the run directory.

    Example: research-pipeline plan 'local memory system for AI agents'
    """
    from research_pipeline.cli.cmd_plan import run_plan

    opts = _common_options(verbose, config, workspace, run_id)
    run_plan(topic, **opts)


@app.command()
def search(
    topic: str = typer.Argument(
        None, help="Research topic (or use --run-id to resume)."
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
    config: Path | None = typer.Option(None, "--config", "-c"),
    workspace: Path | None = typer.Option(None, "--workspace", "-w"),
    run_id: str | None = typer.Option(None, "--run-id"),
    resume: bool = typer.Option(False, "--resume"),
    source: str = typer.Option(
        None,
        "--source",
        "-s",
        help="Search source(s): arxiv, scholar, all (default: from config).",
    ),
) -> None:
    """Search arXiv and Google Scholar in parallel.

    Both sources run concurrently. Results are deduplicated by
    arxiv_id and title. Use --source to select a single source.

    Example: research-pipeline search 'AI agents' --source all
    """
    from research_pipeline.cli.cmd_search import run_search

    opts = _common_options(verbose, config, workspace, run_id)
    run_search(topic, resume=resume, source=source, **opts)


@app.command()
def screen(
    verbose: bool = typer.Option(False, "--verbose", "-v"),
    config: Path | None = typer.Option(None, "--config", "-c"),
    workspace: Path | None = typer.Option(None, "--workspace", "-w"),
    run_id: str = typer.Option(..., "--run-id", help="Run ID with search results."),
    resume: bool = typer.Option(False, "--resume"),
) -> None:
    """Score and rank candidates by relevance.

    Uses heuristic term-matching to score each candidate, then
    selects the top-k for download. Requires a completed search stage.

    Example: research-pipeline screen --run-id <RUN_ID>
    """
    from research_pipeline.cli.cmd_screen import run_screen

    opts = _common_options(verbose, config, workspace, run_id)
    run_screen(resume=resume, **opts)


@app.command()
def download(
    verbose: bool = typer.Option(False, "--verbose", "-v"),
    config: Path | None = typer.Option(None, "--config", "-c"),
    workspace: Path | None = typer.Option(None, "--workspace", "-w"),
    run_id: str = typer.Option(..., "--run-id", help="Run ID with screened shortlist."),
    force: bool = typer.Option(False, "--force"),
) -> None:
    """Download PDFs for shortlisted candidates.

    Downloads from arXiv with rate limiting. Use --force to
    re-download existing files. Requires a completed screen stage.

    Example: research-pipeline download --run-id <RUN_ID>
    """
    from research_pipeline.cli.cmd_download import run_download

    opts = _common_options(verbose, config, workspace, run_id)
    run_download(force=force, **opts)


@app.command()
def convert(
    verbose: bool = typer.Option(False, "--verbose", "-v"),
    config: Path | None = typer.Option(None, "--config", "-c"),
    workspace: Path | None = typer.Option(None, "--workspace", "-w"),
    run_id: str = typer.Option(..., "--run-id", help="Run ID with downloaded PDFs."),
    force: bool = typer.Option(False, "--force"),
    backend: str | None = typer.Option(
        None,
        "--backend",
        "-b",
        help="Converter backend: docling, marker, pymupdf4llm (default: from config).",
    ),
) -> None:
    """Convert PDFs to Markdown.

    Supports multiple backends: docling, marker, pymupdf4llm.
    Use --backend to override the config default.
    Requires the corresponding extra to be installed.

    Example: research-pipeline convert --run-id <RUN_ID> --backend marker
    """
    from research_pipeline.cli.cmd_convert import run_convert

    opts = _common_options(verbose, config, workspace, run_id)
    run_convert(force=force, backend=backend, **opts)


@app.command()
def extract(
    verbose: bool = typer.Option(False, "--verbose", "-v"),
    config: Path | None = typer.Option(None, "--config", "-c"),
    workspace: Path | None = typer.Option(None, "--workspace", "-w"),
    run_id: str = typer.Option(..., "--run-id", help="Run ID with converted Markdown."),
) -> None:
    """Extract structured sections from Markdown.

    Extracts title, abstract, sections, references from converted
    Markdown. Requires a completed convert stage.

    Example: research-pipeline extract --run-id <RUN_ID>
    """
    from research_pipeline.cli.cmd_extract import run_extract

    opts = _common_options(verbose, config, workspace, run_id)
    run_extract(**opts)


@app.command()
def summarize(
    verbose: bool = typer.Option(False, "--verbose", "-v"),
    config: Path | None = typer.Option(None, "--config", "-c"),
    workspace: Path | None = typer.Option(None, "--workspace", "-w"),
    run_id: str = typer.Option(..., "--run-id", help="Run ID with extractions."),
) -> None:
    """Generate per-paper summaries and synthesis.

    Creates individual paper summaries and a cross-paper synthesis
    document. Requires a completed extract stage.

    Example: research-pipeline summarize --run-id <RUN_ID>
    """
    from research_pipeline.cli.cmd_summarize import run_summarize

    opts = _common_options(verbose, config, workspace, run_id)
    run_summarize(**opts)


@app.command()
def run(
    topic: str = typer.Argument(..., help="Research topic (natural language)."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
    config: Path | None = typer.Option(None, "--config", "-c"),
    workspace: Path | None = typer.Option(None, "--workspace", "-w"),
    run_id: str | None = typer.Option(None, "--run-id"),
    resume: bool = typer.Option(False, "--resume"),
    source: str = typer.Option(
        None,
        "--source",
        "-s",
        help="Search source(s): arxiv, scholar, all (default: from config).",
    ),
) -> None:
    """Run all 7 stages end-to-end.

    plan > search > screen > download > convert > extract > summarize.

    Searches arXiv and Google Scholar in parallel, screens for
    relevance, downloads PDFs, converts to Markdown, extracts
    content, and generates summaries. Use --resume to continue
    a partially completed run.

    Example: research-pipeline run 'local memory system for AI agents'
    """
    from research_pipeline.cli.cmd_run import run_full

    opts = _common_options(verbose, config, workspace, run_id)
    run_full(topic, resume=resume, source=source, **opts)


@app.command()
def inspect(
    verbose: bool = typer.Option(False, "--verbose", "-v"),
    workspace: Path | None = typer.Option(None, "--workspace", "-w"),
    run_id: str | None = typer.Option(
        None, "--run-id", help="Specific run to inspect."
    ),
) -> None:
    """Show run status, artifacts, and cached data.

    Without --run-id, lists all runs. With --run-id, shows
    detailed stage status and output paths for that run.

    Example: research-pipeline inspect --run-id <RUN_ID>
    """
    from research_pipeline.cli.cmd_inspect import run_inspect

    _common_options(verbose)
    run_inspect(workspace=workspace, run_id=run_id)


@app.command()
def quality(
    verbose: bool = typer.Option(False, "--verbose", "-v"),
    config: Path | None = typer.Option(None, "--config", "-c"),
    workspace: Path | None = typer.Option(None, "--workspace", "-w"),
    run_id: str = typer.Option(
        ..., "--run-id", help="Run ID with search/screen results."
    ),
) -> None:
    """Compute quality scores for candidate papers.

    Evaluates papers on citation impact, venue reputation, author
    credibility, and recency. Requires a completed search or screen stage.

    Example: research-pipeline quality --run-id <RUN_ID>
    """
    from research_pipeline.cli.cmd_quality import run_quality

    opts = _common_options(verbose, config, workspace, run_id)
    run_quality(**opts)


@app.command(name="convert-file")
def convert_file(
    pdf_path: Path = typer.Argument(..., help="Path to the PDF file to convert."),
    output_dir: Path | None = typer.Option(
        None, "--output", "-o", help="Output directory (default: same as PDF)."
    ),
    backend: str | None = typer.Option(
        None,
        "--backend",
        "-b",
        help="Converter backend: docling, marker, pymupdf4llm (default: from config).",
    ),
    config: Path | None = typer.Option(None, "--config", "-c"),
) -> None:
    """Convert a single PDF to Markdown (standalone, no pipeline).

    Supports multiple backends: docling, marker, pymupdf4llm.
    Output defaults to the same directory as the input PDF.

    Example: research-pipeline convert-file paper.pdf -o ./output/ --backend marker
    """
    from research_pipeline.cli.cmd_convert_file import run_convert_file

    run_convert_file(
        pdf_path, output_dir=output_dir, backend=backend, config_path=config
    )


@app.command()
def expand(
    verbose: bool = typer.Option(False, "--verbose", "-v"),
    config: Path | None = typer.Option(None, "--config", "-c"),
    workspace: Path | None = typer.Option(None, "--workspace", "-w"),
    run_id: str = typer.Option(
        ..., "--run-id", help="Run ID to store expanded candidates."
    ),
    paper_ids: str = typer.Option(
        ...,
        "--paper-ids",
        help="Comma-separated arXiv IDs or S2 paper IDs to expand.",
    ),
    direction: str = typer.Option(
        "both",
        "--direction",
        "-d",
        help="citations, references, or both.",
    ),
    limit: int = typer.Option(
        50,
        "--limit",
        help="Max related papers per seed paper per direction.",
    ),
) -> None:
    """Expand citation graph for specified papers.

    Fetches papers that cite or are referenced by the given seed
    papers using the Semantic Scholar API. Requires explicit
    paper IDs — no autonomous selection.

    Example: research-pipeline expand --run-id <ID> --paper-ids 2401.12345,2401.67890
    """
    from research_pipeline.cli.cmd_expand import run_expand

    opts = _common_options(verbose, config, workspace, run_id)
    ids = [p.strip() for p in paper_ids.split(",") if p.strip()]
    run_expand(
        paper_ids=ids,
        direction=direction,
        limit_per_paper=limit,
        **opts,
    )


@app.command(name="convert-rough")
def convert_rough(
    verbose: bool = typer.Option(False, "--verbose", "-v"),
    config: Path | None = typer.Option(None, "--config", "-c"),
    workspace: Path | None = typer.Option(None, "--workspace", "-w"),
    run_id: str = typer.Option(..., "--run-id", help="Run ID with downloaded PDFs."),
    force: bool = typer.Option(False, "--force"),
) -> None:
    """Rough-convert all downloaded PDFs using pymupdf4llm.

    Fast CPU-only conversion for all papers (Tier 2). The agent
    reads rough markdown to decide which papers need fine conversion.

    Example: research-pipeline convert-rough --run-id <RUN_ID>
    """
    from research_pipeline.cli.cmd_convert_rough import run_convert_rough

    opts = _common_options(verbose, config, workspace, run_id)
    run_convert_rough(force=force, **opts)


@app.command(name="convert-fine")
def convert_fine(
    verbose: bool = typer.Option(False, "--verbose", "-v"),
    config: Path | None = typer.Option(None, "--config", "-c"),
    workspace: Path | None = typer.Option(None, "--workspace", "-w"),
    run_id: str = typer.Option(..., "--run-id", help="Run ID with downloaded PDFs."),
    paper_ids: str = typer.Option(
        ...,
        "--paper-ids",
        help="Comma-separated arXiv IDs to fine-convert.",
    ),
    force: bool = typer.Option(False, "--force"),
    backend: str | None = typer.Option(
        None,
        "--backend",
        "-b",
        help="Converter backend override.",
    ),
) -> None:
    """Fine-convert selected PDFs using high-quality backend.

    Converts agent-selected papers with docling, marker, or cloud
    backend (Tier 3). Requires explicit --paper-ids.

    Example: research-pipeline convert-fine --run-id <ID> --paper-ids 2401.12345
    """
    from research_pipeline.cli.cmd_convert_fine import run_convert_fine

    opts = _common_options(verbose, config, workspace, run_id)
    ids = [p.strip() for p in paper_ids.split(",") if p.strip()]
    run_convert_fine(paper_ids=ids, force=force, backend=backend, **opts)


@app.command()
def index(
    verbose: bool = typer.Option(False, "--verbose", "-v"),
    list_papers: bool = typer.Option(False, "--list", help="List indexed papers."),
    gc: bool = typer.Option(False, "--gc", help="Garbage collect stale entries."),
    db_path: str | None = typer.Option(
        None, "--db-path", help="Path to index database."
    ),
) -> None:
    """Manage the global paper index for incremental runs.

    Browse indexed papers or clean stale entries.

    Example: research-pipeline index --list
    """
    from research_pipeline.cli.cmd_index import run_index
    from research_pipeline.infra.logging import setup_logging

    level = logging.DEBUG if verbose else logging.INFO
    setup_logging(level=level)
    run_index(list_papers=list_papers, gc=gc, db_path=db_path)


@app.command(name="install-skill")
def install_skill(
    target: str = typer.Option(
        "",
        "--target",
        "-t",
        help=(
            "Target directory for skill installation. "
            "Default: ~/.claude/skills/research-pipeline"
        ),
    ),
    symlink: bool = typer.Option(
        False,
        "--symlink",
        "-s",
        help="Create a symlink instead of copying files.",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Overwrite existing skill directory.",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Install the research-pipeline skill to ~/.claude/skills/.

    Copies (or symlinks) the bundled SKILL.md, config.toml, and reference
    docs so that Claude Code and GitHub Copilot can discover them.

    Example: research-pipeline install-skill
    Example: research-pipeline install-skill --symlink --force
    """
    from research_pipeline.cli.cmd_install_skill import (
        DEFAULT_SKILL_DIR,
        run_install_skill,
    )
    from research_pipeline.infra.logging import setup_logging

    level = logging.DEBUG if verbose else logging.INFO
    setup_logging(level=level)

    target_path = Path(target) if target else DEFAULT_SKILL_DIR
    run_install_skill(target=target_path, symlink=symlink, force=force)


if __name__ == "__main__":
    app()
