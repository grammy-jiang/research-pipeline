"""CLI handler for the 'analyze-claims' command."""

from __future__ import annotations

import logging
from pathlib import Path

import typer

from research_pipeline.analysis.decomposer import decompose_paper
from research_pipeline.config.loader import load_config
from research_pipeline.models.summary import PaperSummary
from research_pipeline.storage.manifests import read_jsonl, write_jsonl
from research_pipeline.storage.workspace import get_stage_dir, init_run

logger = logging.getLogger(__name__)


def run_analyze_claims(
    config_path: Path | None = None,
    workspace: Path | None = None,
    run_id: str | None = None,
) -> None:
    """Execute claim decomposition on paper summaries.

    Args:
        config_path: Path to config TOML.
        workspace: Workspace directory.
        run_id: Run ID with summaries.
    """
    config = load_config(config_path)
    ws = workspace or Path(config.workspace)
    run_id_str, run_root = init_run(ws, run_id)

    # Read paper summaries
    summary_dir = get_stage_dir(run_root, "summarize")
    summary_path = summary_dir / "paper_summaries.jsonl"
    if not summary_path.exists():
        # Also try individual summary files
        summary_files = list(summary_dir.glob("*.summary.json"))
        if not summary_files:
            typer.echo(
                "Error: no paper summaries found. Run 'summarize' first.", err=True
            )
            raise typer.Exit(1)
        raw = []
        for sf in summary_files:
            import json

            raw.append(json.loads(sf.read_text(encoding="utf-8")))
    else:
        raw = read_jsonl(summary_path)

    summaries = [PaperSummary.model_validate(d) for d in raw]

    # Get markdown directory for evidence retrieval
    md_dir = get_stage_dir(run_root, "convert")

    results = []
    for summary in summaries:
        md_path = md_dir / f"{summary.arxiv_id}.md"
        if not md_path.exists():
            md_path = md_dir / f"{summary.arxiv_id}{summary.version}.md"

        markdown_path_str = str(md_path) if md_path.exists() else None

        decomp = decompose_paper(
            summary=summary,
            markdown_path=markdown_path_str,
        )
        results.append(decomp)

        supported = decomp.evidence_summary.get("supported", 0)
        total = decomp.total_claims
        logger.info(
            "Decomposed %s: %d claims, %d supported",
            summary.arxiv_id,
            total,
            supported,
        )

    # Write results
    analysis_dir = get_stage_dir(run_root, "summarize") / "claims"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    output_path = analysis_dir / "claim_decomposition.jsonl"
    write_jsonl(output_path, [r.model_dump(mode="json") for r in results])

    total_claims = sum(r.total_claims for r in results)
    total_supported = sum(r.evidence_summary.get("supported", 0) for r in results)

    typer.echo(f"Papers analyzed: {len(results)}")
    typer.echo(f"Total claims: {total_claims}")
    typer.echo(f"Supported: {total_supported}")
    typer.echo(f"Output: {output_path}")
