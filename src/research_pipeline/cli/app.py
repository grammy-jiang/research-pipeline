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
    diversity: bool | None = typer.Option(
        None,
        "--diversity/--no-diversity",
        help="Enable diversity-aware MMR reranking (overrides config).",
    ),
    diversity_lambda: float | None = typer.Option(
        None,
        "--diversity-lambda",
        help="Balance between relevance (0.0) and diversity (1.0). Default 0.3.",
    ),
) -> None:
    """Score and rank candidates by relevance.

    Uses heuristic term-matching to score each candidate, then
    selects the top-k for download. Requires a completed search stage.

    Example: research-pipeline screen --run-id <RUN_ID>
    """
    from research_pipeline.cli.cmd_screen import run_screen

    opts = _common_options(verbose, config, workspace, run_id)
    run_screen(
        resume=resume,
        diversity=diversity,
        diversity_lambda=diversity_lambda,
        **opts,
    )


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
    cross_encoder: bool | None = typer.Option(
        None,
        "--cross-encoder/--no-cross-encoder",
        help=(
            "Enable/disable cross-encoder reranking for chunk retrieval. "
            "Default: auto-detect (use if sentence-transformers is installed)."
        ),
    ),
) -> None:
    """Extract structured sections from Markdown.

    Extracts title, abstract, sections, references from converted
    Markdown. Requires a completed convert stage.

    Example: research-pipeline extract --run-id <RUN_ID>
    """
    from research_pipeline.cli.cmd_extract import run_extract

    opts = _common_options(verbose, config, workspace, run_id)
    opts["cross_encoder"] = cross_encoder
    run_extract(**opts)


@app.command()
def summarize(
    verbose: bool = typer.Option(False, "--verbose", "-v"),
    config: Path | None = typer.Option(None, "--config", "-c"),
    workspace: Path | None = typer.Option(None, "--workspace", "-w"),
    run_id: str = typer.Option(..., "--run-id", help="Run ID with extractions."),
    output_format: str = typer.Option(
        "markdown",
        "--output-format",
        "-f",
        help="Output format: markdown (default), json, bibtex, structured-json.",
    ),
) -> None:
    """Generate per-paper summaries and synthesis.

    Creates individual paper summaries and a cross-paper synthesis
    document. Requires a completed extract stage.

    Use --output-format to export as JSON, BibTeX, or structured
    evidence JSON in addition to the default Markdown synthesis.

    Example: research-pipeline summarize --run-id <RUN_ID>
    Example: research-pipeline summarize --run-id <RUN_ID> -f json
    Example: research-pipeline summarize --run-id <RUN_ID> -f bibtex
    """
    from research_pipeline.cli.cmd_summarize import run_summarize

    opts = _common_options(verbose, config, workspace, run_id)
    run_summarize(output_format=output_format, **opts)


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
    profile: str = typer.Option(
        "standard",
        "--profile",
        "-p",
        help="Pipeline profile: quick, standard, deep, or auto.",
    ),
    ter_iterations: int = typer.Option(
        3,
        "--ter-iterations",
        help="Max THINK→EXECUTE→REFLECT iterations (0 to disable).",
    ),
    auto_approve: bool = typer.Option(
        True,
        "--auto-approve/--interactive",
        help="Auto-approve HITL gates (default) or pause for review.",
    ),
) -> None:
    """Run pipeline stages end-to-end.

    Profiles control which stages execute:

    \b
    quick:    plan → search → screen → summarize (abstract-only)
    standard: full 7-stage pipeline (default)
    deep:     standard + expand + quality + claim analysis + TER loop
    auto:     detect from query complexity

    Example: research-pipeline run --profile quick 'transformer attention'
    """
    from research_pipeline.cli.cmd_run import run_full

    opts = _common_options(verbose, config, workspace, run_id)
    run_full(
        topic,
        resume=resume,
        source=source,
        profile=profile,
        ter_iterations=ter_iterations,
        auto_approve=auto_approve,
        **opts,
    )


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
    reference_boost: float = typer.Option(
        1.0,
        "--reference-boost",
        help="Multiplier for backward (reference) limit when direction=both. "
        "E.g. 2.0 fetches 2x more references than citations.",
    ),
    bfs_depth: int = typer.Option(
        0,
        "--bfs-depth",
        help="BFS expansion depth (0 = disabled, 2 = recommended). "
        "Multi-hop expansion with BM25 pruning at each hop.",
    ),
    bfs_top_k: int = typer.Option(
        10,
        "--bfs-top-k",
        help="Max papers to keep per BFS hop after BM25 ranking.",
    ),
    bfs_query: str = typer.Option(
        "",
        "--bfs-query",
        help="Comma-separated query terms for BFS BM25 pruning.",
    ),
    snowball: bool = typer.Option(
        False,
        "--snowball",
        help="Enable bidirectional snowball expansion mode with "
        "budget-aware stopping (replaces BFS).",
    ),
    snowball_max_rounds: int = typer.Option(
        5,
        "--snowball-max-rounds",
        help="Max snowball iteration rounds (default 5).",
    ),
    snowball_max_papers: int = typer.Option(
        200,
        "--snowball-max-papers",
        help="Hard cap on total discovered papers (default 200).",
    ),
    snowball_decay_threshold: float = typer.Option(
        0.10,
        "--snowball-decay-threshold",
        help="Stop when fraction of relevant new papers drops below "
        "this (0-1, default 0.10).",
    ),
    snowball_decay_patience: int = typer.Option(
        2,
        "--snowball-decay-patience",
        help="Consecutive low-relevance rounds before stopping (default 2).",
    ),
) -> None:
    """Expand citation graph for specified papers.

    Fetches papers that cite or are referenced by the given seed
    papers using the Semantic Scholar API. Requires explicit
    paper IDs — no autonomous selection.

    Three expansion modes:
    1. Single-hop (default): direct citations/references
    2. BFS: multi-hop with BM25 pruning (--bfs-depth 2)
    3. Snowball: iterative bidirectional with budget-aware stopping (--snowball)

    Example: research-pipeline expand --run-id <ID> --paper-ids 2401.12345,2401.67890
    Example: research-pipeline expand --run-id <ID> \\
        --paper-ids 2401.12345 --bfs-depth 2 --bfs-query "transformer,attention"
    Example: research-pipeline expand --run-id <ID> \\
        --paper-ids 2401.12345 --snowball --bfs-query "harness,engineering"
    """
    from research_pipeline.cli.cmd_expand import run_expand

    opts = _common_options(verbose, config, workspace, run_id)
    ids = [p.strip() for p in paper_ids.split(",") if p.strip()]
    run_expand(
        paper_ids=ids,
        direction=direction,
        limit_per_paper=limit,
        reference_boost=reference_boost,
        bfs_depth=bfs_depth,
        bfs_top_k=bfs_top_k,
        query_terms=bfs_query.split(",") if bfs_query else [],
        snowball=snowball,
        snowball_max_rounds=snowball_max_rounds,
        snowball_max_papers=snowball_max_papers,
        snowball_decay_threshold=snowball_decay_threshold,
        snowball_decay_patience=snowball_decay_patience,
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
    search: str | None = typer.Option(
        None, "--search", help="Full-text search across paper titles and abstracts."
    ),
    search_limit: int = typer.Option(
        50, "--search-limit", help="Max results for --search."
    ),
    db_path: str | None = typer.Option(
        None, "--db-path", help="Path to index database."
    ),
) -> None:
    """Manage the global paper index for incremental runs.

    Browse indexed papers, full-text search, or clean stale entries.

    Example: research-pipeline index --list
    Example: research-pipeline index --search "transformer attention"
    """
    from research_pipeline.cli.cmd_index import run_index
    from research_pipeline.infra.logging import setup_logging

    level = logging.DEBUG if verbose else logging.INFO
    setup_logging(level=level)
    run_index(
        list_papers=list_papers,
        gc=gc,
        search=search,
        search_limit=search_limit,
        db_path=db_path,
    )


@app.command(name="setup")
def setup(
    skill_target: str = typer.Option(
        "",
        "--skill-target",
        help=(
            "Target directory for skill installation. "
            "Default: ~/.claude/skills/research-pipeline"
        ),
    ),
    agents_target: str = typer.Option(
        "",
        "--agents-target",
        help=("Target directory for agent files. " "Default: ~/.claude/agents"),
    ),
    symlink: bool = typer.Option(
        False,
        "--symlink",
        "-s",
        help="Create symlinks instead of copying files.",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Overwrite existing files/directories.",
    ),
    skip_skill: bool = typer.Option(
        False,
        "--skip-skill",
        help="Skip skill installation.",
    ),
    skip_agents: bool = typer.Option(
        False,
        "--skip-agents",
        help="Skip agent installation.",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Install skill and agents to ~/.claude/ for AI assistant discovery.

    Copies (or symlinks) the bundled SKILL.md, config.toml, reference docs,
    and agent definitions (paper-analyzer, paper-screener, paper-synthesizer)
    so that Claude Code and GitHub Copilot can discover them.

    Example: research-pipeline setup
    Example: research-pipeline setup --symlink --force
    Example: research-pipeline setup --skip-agents
    """
    from research_pipeline.cli.cmd_setup import (
        DEFAULT_AGENTS_DIR,
        DEFAULT_SKILL_DIR,
        run_setup,
    )
    from research_pipeline.infra.logging import setup_logging

    level = logging.DEBUG if verbose else logging.INFO
    setup_logging(level=level)

    skill_path = Path(skill_target) if skill_target else DEFAULT_SKILL_DIR
    agents_path = Path(agents_target) if agents_target else DEFAULT_AGENTS_DIR
    run_setup(
        skill_target=skill_path,
        agents_target=agents_path,
        symlink=symlink,
        force=force,
        skip_skill=skip_skill,
        skip_agents=skip_agents,
    )


@app.command(name="install-skill", hidden=True)
def install_skill(
    target: str = typer.Option(
        "",
        "--target",
        "-t",
        help="[Deprecated: use 'setup' instead] Target directory.",
    ),
    symlink: bool = typer.Option(False, "--symlink", "-s"),
    force: bool = typer.Option(False, "--force", "-f"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """[Deprecated] Use 'research-pipeline setup' instead."""
    import warnings

    from research_pipeline.cli.cmd_setup import (
        DEFAULT_SKILL_DIR,
        run_setup,
    )
    from research_pipeline.infra.logging import setup_logging

    warnings.warn(
        "'install-skill' is deprecated. Use 'research-pipeline setup' instead.",
        DeprecationWarning,
        stacklevel=1,
    )

    level = logging.DEBUG if verbose else logging.INFO
    setup_logging(level=level)

    target_path = Path(target) if target else DEFAULT_SKILL_DIR
    run_setup(
        skill_target=target_path,
        symlink=symlink,
        force=force,
        skip_agents=True,
    )


@app.command()
def analyze(
    verbose: bool = typer.Option(False, "--verbose", "-v"),
    config: Path | None = typer.Option(None, "--config", "-c"),
    workspace: Path | None = typer.Option(None, "--workspace", "-w"),
    run_id: str = typer.Option(..., "--run-id", help="Run ID with converted papers."),
    collect: bool = typer.Option(
        False, "--collect", help="Validate collected analysis JSON files."
    ),
    paper_ids: str | None = typer.Option(
        None,
        "--paper-ids",
        help="Comma-separated arXiv IDs to analyze (default: all).",
    ),
) -> None:
    """Prepare per-paper analysis tasks or validate collected results.

    Without --collect: discovers converted papers and generates analysis
    task prompts in the analysis/ directory.

    With --collect: validates analysis JSON files produced by the
    paper-analyzer sub-agent against the required schema.

    Example: research-pipeline analyze --run-id <RUN_ID>
    Example: research-pipeline analyze --run-id <RUN_ID> --collect
    """
    from research_pipeline.cli.cmd_analyze import run_analyze

    opts = _common_options(verbose, config, workspace, run_id)
    ids = [p.strip() for p in paper_ids.split(",") if p.strip()] if paper_ids else None
    run_analyze(collect=collect, paper_ids=ids, **opts)


@app.command()
def validate(
    report: Path | None = typer.Option(
        None, "--report", "-r", help="Path to report markdown file."
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
    config: Path | None = typer.Option(None, "--config", "-c"),
    workspace: Path | None = typer.Option(None, "--workspace", "-w"),
    run_id: str | None = typer.Option(
        None, "--run-id", help="Run ID to find synthesis report."
    ),
    output: Path | None = typer.Option(
        None, "--output", "-o", help="Output path for validation JSON."
    ),
) -> None:
    """Validate a research report for completeness and quality.

    Checks for required sections, confidence-level annotations,
    evidence citations, gap classifications, tables, Mermaid diagrams,
    and LaTeX formulas. Produces a PASS/FAIL verdict with a score.

    Example: research-pipeline validate --report report.md
    Example: research-pipeline validate --run-id <RUN_ID>
    """
    from research_pipeline.cli.cmd_validate import run_validate
    from research_pipeline.infra.logging import setup_logging

    level = logging.DEBUG if verbose else logging.INFO
    setup_logging(level=level)

    run_validate(
        report=report,
        workspace=workspace,
        run_id=run_id,
        output=output,
    )


@app.command()
def compare(
    run_a: str = typer.Option(..., "--run-a", help="First run ID (baseline)."),
    run_b: str = typer.Option(..., "--run-b", help="Second run ID (latest)."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
    config: Path | None = typer.Option(None, "--config", "-c"),
    workspace: Path | None = typer.Option(None, "--workspace", "-w"),
    output: Path | None = typer.Option(
        None, "--output", "-o", help="Output path for comparison JSON."
    ),
) -> None:
    """Compare two pipeline runs: papers, findings, gaps, confidence.

    Produces a structured diff showing which papers are new, which gaps
    were resolved, how confidence levels changed, and whether the
    readiness verdict improved.

    Example: research-pipeline compare --run-a <RUN_A> --run-b <RUN_B>
    """
    from research_pipeline.cli.cmd_compare import run_compare
    from research_pipeline.infra.logging import setup_logging

    level = logging.DEBUG if verbose else logging.INFO
    setup_logging(level=level)

    run_compare(
        run_id_a=run_a,
        run_id_b=run_b,
        config_path=config,
        workspace=workspace,
        output=output,
    )


@app.command()
def coherence(
    run_ids: list[str] = typer.Argument(
        ..., help="Two or more run IDs to evaluate coherence across."
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
    config: Path | None = typer.Option(None, "--config", "-c"),
    workspace: Path | None = typer.Option(None, "--workspace", "-w"),
    output: Path | None = typer.Option(
        None, "--output", "-o", help="Output path for coherence report JSON."
    ),
) -> None:
    """Evaluate knowledge coherence across multiple pipeline runs.

    Computes factual consistency, temporal ordering, knowledge update
    fidelity, and contradiction detection across 2+ runs.

    Example: research-pipeline coherence <RUN_A> <RUN_B> [<RUN_C> ...]
    """
    from research_pipeline.cli.cmd_coherence import run_coherence_cmd
    from research_pipeline.infra.logging import setup_logging

    level = logging.DEBUG if verbose else logging.INFO
    setup_logging(level=level)

    run_coherence_cmd(
        run_ids=run_ids,
        config_path=config,
        workspace=workspace,
        output=output,
    )


@app.command()
def consolidate(
    run_ids: list[str] | None = typer.Argument(
        None, help="Run IDs to ingest. If omitted, scans workspace."
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
    config: Path | None = typer.Option(None, "--config", "-c"),
    workspace: Path | None = typer.Option(None, "--workspace", "-w"),
    output: Path | None = typer.Option(
        None, "--output", "-o", help="Output path for consolidation report JSON."
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Compute metrics without modifying store."
    ),
    capacity: int = typer.Option(
        100, "--capacity", help="Episode capacity before triggering consolidation."
    ),
    threshold: float = typer.Option(
        0.8, "--threshold", help="Fraction of capacity triggering consolidation."
    ),
    min_support: int = typer.Option(
        2, "--min-support", help="Min run appearances for rule promotion."
    ),
    staleness_days: int = typer.Option(
        90, "--staleness-days", help="Age threshold (days) for pruning."
    ),
) -> None:
    """Consolidate cross-run memory: compress episodes, promote rules, prune stale.

    Implements episodic → semantic consolidation following the SEA/MLMF
    three-tier memory architecture. Automatically ingests synthesis
    results from pipeline runs into the episode store.

    Example: research-pipeline consolidate
    Example: research-pipeline consolidate run1 run2 run3 --dry-run
    """
    from research_pipeline.cli.cmd_consolidate import run_consolidate_cmd
    from research_pipeline.infra.logging import setup_logging

    level = logging.DEBUG if verbose else logging.INFO
    setup_logging(level=level)

    run_consolidate_cmd(
        run_ids=run_ids if run_ids else None,
        config_path=config,
        workspace=workspace,
        output=output,
        dry_run=dry_run,
        capacity=capacity,
        threshold=threshold,
        min_support=min_support,
        staleness_days=staleness_days,
    )


@app.command("analyze-claims")
def analyze_claims(
    verbose: bool = typer.Option(False, "--verbose", "-v"),
    config: Path | None = typer.Option(None, "--config", "-c"),
    workspace: Path | None = typer.Option(None, "--workspace", "-w"),
    run_id: str = typer.Option(..., "--run-id", help="Run ID with paper summaries."),
) -> None:
    """Decompose paper summaries into atomic claims with evidence classification.

    Breaks each finding, limitation, objective, and methodology statement
    into atomic claims, then classifies evidence support using BM25 retrieval:
    supported, partial, conflicting, inconclusive, or unsupported.

    Example: research-pipeline analyze-claims --run-id <RUN_ID>
    """
    from research_pipeline.cli.cmd_analyze_claims import run_analyze_claims

    opts = _common_options(verbose, config, workspace, run_id)
    run_analyze_claims(**opts)


@app.command("score-claims")
def score_claims(
    verbose: bool = typer.Option(False, "--verbose", "-v"),
    config: Path | None = typer.Option(None, "--config", "-c"),
    workspace: Path | None = typer.Option(None, "--workspace", "-w"),
    run_id: str | None = typer.Option(None, "--run-id", "-r", help="Run ID."),
) -> None:
    """Score confidence for decomposed claims using multi-signal aggregation.

    Computes per-claim confidence scores using evidence strength, hedging
    language detection (LVU), citation density, and retrieval quality.
    When an LLM is available, adds multi-sample consistency verification.

    Example: research-pipeline score-claims --run-id <RUN_ID>
    """
    from research_pipeline.cli.cmd_score_claims import run_score_claims

    opts = _common_options(verbose, config, workspace, run_id)
    run_score_claims(**opts)


@app.command("kg-stats")
def kg_stats(
    db_path: Path | None = typer.Option(None, "--db", help="KG database path."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Show knowledge graph statistics.

    Displays entity and triple counts by type.

    Example: research-pipeline kg-stats
    """
    from research_pipeline.cli.cmd_kg import run_kg_stats
    from research_pipeline.infra.logging import setup_logging

    level = logging.DEBUG if verbose else logging.INFO
    setup_logging(level=level)
    run_kg_stats(db_path=db_path)


@app.command("kg-query")
def kg_query(
    entity_id: str = typer.Argument(help="Entity ID to query."),
    db_path: Path | None = typer.Option(None, "--db", help="KG database path."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Query an entity and its relations in the knowledge graph.

    Shows entity details and all connected triples.

    Example: research-pipeline kg-query 2401.12345
    """
    from research_pipeline.cli.cmd_kg import run_kg_query
    from research_pipeline.infra.logging import setup_logging

    level = logging.DEBUG if verbose else logging.INFO
    setup_logging(level=level)
    run_kg_query(entity_id=entity_id, db_path=db_path)


@app.command("kg-ingest")
def kg_ingest(
    verbose: bool = typer.Option(False, "--verbose", "-v"),
    config: Path | None = typer.Option(None, "--config", "-c"),
    workspace: Path | None = typer.Option(None, "--workspace", "-w"),
    run_id: str | None = typer.Option(None, "--run-id", "-r", help="Run ID."),
    db_path: Path | None = typer.Option(None, "--db", help="KG database path."),
) -> None:
    """Ingest pipeline results into the knowledge graph.

    Reads candidates and claim decompositions from a completed run
    and populates the knowledge graph with entities and relations.

    Example: research-pipeline kg-ingest --run-id <RUN_ID>
    """
    from research_pipeline.cli.cmd_kg import run_kg_ingest
    from research_pipeline.infra.logging import setup_logging

    level = logging.DEBUG if verbose else logging.INFO
    setup_logging(level=level)
    run_kg_ingest(
        config_path=config,
        workspace=workspace,
        run_id=run_id,
        db_path=db_path,
    )


@app.command("memory-stats")
def memory_stats(
    verbose: bool = typer.Option(False, "--verbose", "-v"),
    episodic_db: Path | None = typer.Option(
        None, "--episodic-db", help="Episodic memory database path."
    ),
    kg_db: Path | None = typer.Option(None, "--kg-db", help="KG database path."),
) -> None:
    """Show memory tier statistics.

    Displays working, episodic, and semantic memory summary.

    Example: research-pipeline memory-stats
    """
    from research_pipeline.cli.cmd_memory import run_memory_stats
    from research_pipeline.infra.logging import setup_logging

    level = logging.DEBUG if verbose else logging.INFO
    setup_logging(level=level)
    run_memory_stats(episodic_path=episodic_db, kg_path=kg_db)


@app.command("memory-episodes")
def memory_episodes(
    limit: int = typer.Option(10, "--limit", "-n", help="Max episodes to show."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
    episodic_db: Path | None = typer.Option(
        None, "--episodic-db", help="Episodic memory database path."
    ),
) -> None:
    """List recent episodic memories (past runs).

    Example: research-pipeline memory-episodes --limit 5
    """
    from research_pipeline.cli.cmd_memory import run_memory_episodes
    from research_pipeline.infra.logging import setup_logging

    level = logging.DEBUG if verbose else logging.INFO
    setup_logging(level=level)
    run_memory_episodes(limit=limit, episodic_path=episodic_db)


@app.command("memory-search")
def memory_search(
    topic: str = typer.Argument(..., help="Topic to search in episodic memory."),
    limit: int = typer.Option(10, "--limit", "-n", help="Max results."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
    episodic_db: Path | None = typer.Option(
        None, "--episodic-db", help="Episodic memory database path."
    ),
) -> None:
    """Search episodic memory for past runs on a topic.

    Example: research-pipeline memory-search "transformer"
    """
    from research_pipeline.cli.cmd_memory import run_memory_search
    from research_pipeline.infra.logging import setup_logging

    level = logging.DEBUG if verbose else logging.INFO
    setup_logging(level=level)
    run_memory_search(topic=topic, limit=limit, episodic_path=episodic_db)


@app.command("evaluate")
def evaluate(
    run_id: str = typer.Option(..., "--run-id", help="Run ID to evaluate."),
    stage: str = typer.Option(
        "", "--stage", "-s", help="Specific stage (default: all)."
    ),
    workspace: str = typer.Option(
        "runs", "--workspace", "-w", help="Workspace directory."
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Evaluate pipeline outputs against their schemas.

    Validates completeness, score ranges, and cross-field consistency.

    Example: research-pipeline evaluate --run-id <RUN_ID>
    """
    from research_pipeline.cli.cmd_evaluate import evaluate_cmd
    from research_pipeline.infra.logging import setup_logging

    level = logging.DEBUG if verbose else logging.INFO
    setup_logging(level=level)
    evaluate_cmd(run_id=run_id, stage=stage, workspace=workspace)


@app.command()
def feedback(
    verbose: bool = typer.Option(False, "--verbose", "-v"),
    workspace: Path | None = typer.Option(None, "--workspace", "-w"),
    run_id: str = typer.Option(
        ..., "--run-id", help="Run ID whose screened papers to give feedback on."
    ),
    accept: list[str] = typer.Option(
        [], "--accept", "-a", help="Paper IDs to accept (repeatable)."
    ),
    reject: list[str] = typer.Option(
        [], "--reject", "-r", help="Paper IDs to reject (repeatable)."
    ),
    reason: str = typer.Option(
        "", "--reason", help="Optional reason for the decisions."
    ),
    show: bool = typer.Option(False, "--show", "-s", help="Show feedback stats."),
    adjust: bool = typer.Option(
        False, "--adjust", help="Recompute adjusted BM25 weights."
    ),
) -> None:
    """Record user feedback on screened papers.

    Accepts or rejects paper IDs from a screening run. Accumulated
    feedback adjusts BM25 weights via ELO-style learning.

    \b
    Examples:
      research-pipeline feedback --run-id <ID> --accept 2401.12345 --accept 2401.12346
      research-pipeline feedback --run-id <ID> --reject 2401.12347 --reason "off-topic"
      research-pipeline feedback --run-id <ID> --show
      research-pipeline feedback --run-id <ID> --adjust
    """
    from research_pipeline.cli.cmd_feedback import feedback_cmd
    from research_pipeline.infra.logging import setup_logging

    level = logging.DEBUG if verbose else logging.INFO
    setup_logging(level=level)
    feedback_cmd(
        run_id=run_id,
        accept=accept,
        reject=reject,
        reason=reason,
        show=show,
        adjust=adjust,
        workspace=workspace,
    )


@app.command(name="eval-log")
def eval_log(
    verbose: bool = typer.Option(False, "--verbose", "-v"),
    workspace: Path | None = typer.Option(None, "--workspace", "-w"),
    run_id: str = typer.Option(
        ..., "--run-id", help="Run ID to inspect evaluation logs for."
    ),
    channel: str = typer.Option(
        "all",
        "--channel",
        "-c",
        help="Channel to inspect: traces, audit, snapshots, summary, all.",
    ),
    stage: str = typer.Option("", "--stage", "-s", help="Filter by pipeline stage."),
    limit: int = typer.Option(50, "--limit", "-n", help="Maximum records to display."),
) -> None:
    """Inspect three-channel evaluation logs for a pipeline run.

    Three channels capture different aspects of pipeline execution:

    \b
    - traces:    Execution flow (JSONL) — timing, causality
    - audit:     Structured DB (SQLite) — who/what/when
    - snapshots: Filesystem state — stage boundary captures

    \b
    Examples:
      research-pipeline eval-log --run-id <ID>
      research-pipeline eval-log --run-id <ID> --channel traces --stage screen
      research-pipeline eval-log --run-id <ID> --channel audit --limit 20
      research-pipeline eval-log --run-id <ID> --channel summary
    """
    from research_pipeline.cli.cmd_eval_log import eval_log_cmd
    from research_pipeline.infra.logging import setup_logging

    level = logging.DEBUG if verbose else logging.INFO
    setup_logging(level=level)
    eval_log_cmd(
        run_id=run_id,
        channel=channel,
        stage=stage,
        limit=limit,
        workspace=workspace,
    )


if __name__ == "__main__":
    app()


@app.command("aggregate")
def aggregate_command(
    run_id: str = typer.Option(..., "--run-id", help="Pipeline run ID."),
    min_pointers: int = typer.Option(
        0,
        "--min-pointers",
        help="Minimum evidence pointers per statement.",
    ),
    max_words: int = typer.Option(
        50,
        "--max-words",
        help="Maximum words per statement.",
    ),
    similarity_threshold: float = typer.Option(
        0.7,
        "--similarity-threshold",
        help="Threshold for merging similar statements (0-1).",
    ),
    no_strip_rhetoric: bool = typer.Option(
        False,
        "--no-strip-rhetoric",
        help="Disable rhetoric stripping.",
    ),
    output_format: str = typer.Option(
        "text",
        "--format",
        help="Output format: text or json.",
    ),
    config_path: str = typer.Option(
        "config.toml",
        "--config",
        help="Path to config file.",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output."),
) -> None:
    """Aggregate evidence from synthesis, stripping rhetoric.

    Examples:

    .. code-block:: bash

       research-pipeline aggregate --run-id <ID>
       research-pipeline aggregate --run-id <ID> --min-pointers 1
       research-pipeline aggregate --run-id <ID> --format json
    """
    from research_pipeline.cli.cmd_aggregate import aggregate_cmd
    from research_pipeline.infra.logging import setup_logging

    level = logging.DEBUG if verbose else logging.INFO
    setup_logging(level=level)
    aggregate_cmd(
        run_id=run_id,
        min_pointers=min_pointers,
        max_words=max_words,
        similarity_threshold=similarity_threshold,
        no_strip_rhetoric=no_strip_rhetoric,
        output_format=output_format,
        config_path=config_path,
    )


@app.command("export-html")
def export_html_command(
    run_id: str = typer.Option(
        "",
        "--run-id",
        help="Pipeline run ID (reads synthesis_report.json).",
    ),
    markdown_file: str = typer.Option(
        "",
        "--markdown",
        help="Path to a Markdown report to convert.",
    ),
    output: str = typer.Option(
        "",
        "--output",
        "-o",
        help="Output HTML file path.",
    ),
    title: str = typer.Option(
        "Research Report",
        "--title",
        help="Report title (used with --markdown mode).",
    ),
    config_path: str = typer.Option(
        "config.toml",
        "--config",
        help="Path to config file.",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output."),
) -> None:
    """Export synthesis report as self-contained HTML.

    Two modes:

    .. code-block:: bash

       research-pipeline export-html --run-id <ID>
       research-pipeline export-html --markdown report.md -o report.html
    """
    from research_pipeline.cli.cmd_export_html import export_html_cmd
    from research_pipeline.infra.logging import setup_logging

    level = logging.DEBUG if verbose else logging.INFO
    setup_logging(level=level)
    export_html_cmd(
        run_id=run_id,
        markdown_file=markdown_file,
        output=output,
        title=title,
        config_path=config_path,
    )
