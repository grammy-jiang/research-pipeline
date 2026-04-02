"""CLI root application and subcommand registration."""

import logging
from pathlib import Path

import typer

from arxiv_paper_pipeline.infra.logging import setup_logging

app = typer.Typer(
    name="arxiv-paper-pipeline",
    help="Search, screen, download, convert, and summarize arXiv papers.",
    no_args_is_help=True,
)


def _common_options(
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable debug logging."
    ),
    config: Path | None = typer.Option(
        None, "--config", "-c", help="Path to config TOML."
    ),
    workspace: Path | None = typer.Option(
        None, "--workspace", "-w", help="Workspace directory."
    ),
    run_id: str | None = typer.Option(
        None, "--run-id", help="Run ID (generated if not set)."
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
    """Normalize a topic into a structured query plan."""
    from arxiv_paper_pipeline.cli.cmd_plan import run_plan

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
) -> None:
    """Execute arXiv API search from a query plan."""
    from arxiv_paper_pipeline.cli.cmd_search import run_search

    opts = _common_options(verbose, config, workspace, run_id)
    run_search(topic, resume=resume, **opts)


@app.command()
def screen(
    verbose: bool = typer.Option(False, "--verbose", "-v"),
    config: Path | None = typer.Option(None, "--config", "-c"),
    workspace: Path | None = typer.Option(None, "--workspace", "-w"),
    run_id: str = typer.Option(..., "--run-id", help="Run ID with search results."),
    resume: bool = typer.Option(False, "--resume"),
) -> None:
    """Two-stage relevance screening of search candidates."""
    from arxiv_paper_pipeline.cli.cmd_screen import run_screen

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
    """Download shortlisted PDFs from arXiv."""
    from arxiv_paper_pipeline.cli.cmd_download import run_download

    opts = _common_options(verbose, config, workspace, run_id)
    run_download(force=force, **opts)


@app.command()
def convert(
    verbose: bool = typer.Option(False, "--verbose", "-v"),
    config: Path | None = typer.Option(None, "--config", "-c"),
    workspace: Path | None = typer.Option(None, "--workspace", "-w"),
    run_id: str = typer.Option(..., "--run-id", help="Run ID with downloaded PDFs."),
    force: bool = typer.Option(False, "--force"),
) -> None:
    """Convert downloaded PDFs to Markdown."""
    from arxiv_paper_pipeline.cli.cmd_convert import run_convert

    opts = _common_options(verbose, config, workspace, run_id)
    run_convert(force=force, **opts)


@app.command()
def extract(
    verbose: bool = typer.Option(False, "--verbose", "-v"),
    config: Path | None = typer.Option(None, "--config", "-c"),
    workspace: Path | None = typer.Option(None, "--workspace", "-w"),
    run_id: str = typer.Option(..., "--run-id", help="Run ID with converted Markdown."),
) -> None:
    """Extract structured content from converted Markdown."""
    from arxiv_paper_pipeline.cli.cmd_extract import run_extract

    opts = _common_options(verbose, config, workspace, run_id)
    run_extract(**opts)


@app.command()
def summarize(
    verbose: bool = typer.Option(False, "--verbose", "-v"),
    config: Path | None = typer.Option(None, "--config", "-c"),
    workspace: Path | None = typer.Option(None, "--workspace", "-w"),
    run_id: str = typer.Option(..., "--run-id", help="Run ID with extractions."),
) -> None:
    """Generate per-paper summaries and cross-paper synthesis."""
    from arxiv_paper_pipeline.cli.cmd_summarize import run_summarize

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
) -> None:
    """Run the full pipeline end-to-end."""
    from arxiv_paper_pipeline.cli.cmd_run import run_full

    opts = _common_options(verbose, config, workspace, run_id)
    run_full(topic, resume=resume, **opts)


@app.command()
def inspect(
    verbose: bool = typer.Option(False, "--verbose", "-v"),
    workspace: Path | None = typer.Option(None, "--workspace", "-w"),
    run_id: str | None = typer.Option(
        None, "--run-id", help="Specific run to inspect."
    ),
) -> None:
    """Inspect manifests, artifacts, and cache status."""
    from arxiv_paper_pipeline.cli.cmd_inspect import run_inspect

    _common_options(verbose)
    run_inspect(workspace=workspace, run_id=run_id)


if __name__ == "__main__":
    app()
