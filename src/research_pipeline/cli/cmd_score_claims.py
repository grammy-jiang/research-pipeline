"""CLI handler for the 'score-claims' command."""

from __future__ import annotations

import logging
from pathlib import Path

import typer

from research_pipeline.confidence.scorer import score_decomposition
from research_pipeline.config.loader import load_config
from research_pipeline.llm.providers import create_llm_provider
from research_pipeline.models.claim import ClaimDecomposition
from research_pipeline.storage.manifests import read_jsonl, write_jsonl
from research_pipeline.storage.workspace import get_stage_dir, init_run

logger = logging.getLogger(__name__)


def run_score_claims(
    config_path: Path | None = None,
    workspace: Path | None = None,
    run_id: str | None = None,
) -> None:
    """Score confidence for decomposed claims.

    Args:
        config_path: Path to config TOML.
        workspace: Workspace directory.
        run_id: Run ID with claim decompositions.
    """
    config = load_config(config_path)
    ws = workspace or Path(config.workspace)
    run_id_str, run_root = init_run(ws, run_id)

    # Read claim decompositions
    claims_dir = get_stage_dir(run_root, "summarize") / "claims"
    claims_path = claims_dir / "claim_decomposition.jsonl"
    if not claims_path.exists():
        typer.echo(
            "Error: no claim decompositions found. Run 'analyze-claims' first.",
            err=True,
        )
        raise typer.Exit(1)

    raw = read_jsonl(claims_path)
    decompositions = [ClaimDecomposition.model_validate(d) for d in raw]

    # Create LLM provider (optional)
    llm_provider = create_llm_provider(config.llm)
    if llm_provider:
        typer.echo("LLM available — using multi-sample consistency scoring")
    else:
        typer.echo("No LLM — using heuristic-only scoring")

    # Score each decomposition
    results = []
    for decomp in decompositions:
        scored = score_decomposition(decomp, llm_provider)
        results.append(scored)

        avg_conf = sum(c.confidence_score for c in scored.claims) / max(
            len(scored.claims), 1
        )
        logger.info(
            "Scored %s: %d claims, avg confidence %.3f",
            decomp.paper_id,
            len(scored.claims),
            avg_conf,
        )

    # Write results
    output_path = claims_dir / "scored_claims.jsonl"
    write_jsonl(output_path, [r.model_dump(mode="json") for r in results])

    total_claims = sum(len(r.claims) for r in results)
    avg_confidence = sum(c.confidence_score for r in results for c in r.claims) / max(
        total_claims, 1
    )

    typer.echo(f"Papers scored: {len(results)}")
    typer.echo(f"Total claims: {total_claims}")
    typer.echo(f"Average confidence: {avg_confidence:.3f}")
    typer.echo(f"Output: {output_path}")
