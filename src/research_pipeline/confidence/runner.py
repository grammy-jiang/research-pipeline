"""Confidence-scoring stage runner (#109).

Core stage logic for the `score-claims` step, shared by the CLI wrapper
(`cli/cmd_score_claims.py`) and the pipeline orchestrator. Typer-free: it uses
``logging`` for progress and raises :class:`FileNotFoundError` on missing input;
the CLI wrapper maps that to ``typer.Exit`` so the presentation concern stays in
the presentation layer.
"""

from __future__ import annotations

import logging
from pathlib import Path

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

    Raises:
        FileNotFoundError: If no claim decompositions exist for the run.
    """
    config = load_config(config_path)
    ws = workspace or Path(config.workspace)
    run_id_str, run_root = init_run(ws, run_id)

    # Read claim decompositions
    claims_dir = get_stage_dir(run_root, "summarize") / "claims"
    claims_path = claims_dir / "claim_decomposition.jsonl"
    if not claims_path.exists():
        raise FileNotFoundError(
            "no claim decompositions found. Run 'analyze-claims' first."
        )

    raw = read_jsonl(claims_path)
    decompositions = [ClaimDecomposition.model_validate(d) for d in raw]

    # Create LLM provider (optional)
    llm_provider = create_llm_provider(config.llm)
    if llm_provider:
        logger.info("LLM available — using multi-sample consistency scoring")
    else:
        logger.info("No LLM — using heuristic-only scoring")

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

    logger.info("Papers scored: %d", len(results))
    logger.info("Total claims: %d", total_claims)
    logger.info("Average confidence: %.3f", avg_confidence)
    logger.info("Output: %s", output_path)
