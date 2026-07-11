"""4-layer confidence architecture stage runner (#109).

Core stage logic for the `confidence-layers` step, shared by the CLI wrapper
(`cli/cmd_confidence_layers.py`) and the MCP `confidence_layers` tool. Typer-free:
progress via ``logging``; missing input raises :class:`FileNotFoundError` (the CLI
wrapper maps it to ``typer.Exit``); an empty claim set is a normal early return.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from research_pipeline.confidence.architecture import (
    ArchitectureConfig,
    batch_calibration_report,
    fit_platt_scaling,
    score_batch_layered,
)
from research_pipeline.config.loader import load_config
from research_pipeline.llm.providers import create_llm_provider
from research_pipeline.models.claim import ClaimDecomposition
from research_pipeline.storage.manifests import read_jsonl, write_jsonl
from research_pipeline.storage.workspace import get_stage_dir, init_run

logger = logging.getLogger(__name__)


def run_confidence_layers(
    config_path: Path | None = None,
    workspace: Path | None = None,
    run_id: str | None = None,
    l4_threshold: float = 0.50,
    damping: float = 0.8,
    calibrate: bool = False,
) -> None:
    """Score claims through the 4-layer confidence architecture.

    Processes claims through L1 (fast signal) → L2 (adaptive granularity)
    → L3 (calibration correction) → L4 (selective verification).

    Args:
        config_path: Path to config TOML.
        workspace: Workspace directory.
        run_id: Run ID with claim decompositions.
        l4_threshold: Confidence below which L4 verification triggers.
        damping: Fusion damping exponent (0-1).
        calibrate: Whether to fit Platt scaling from prior scored data.

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
    all_claims = [c for d in decompositions for c in d.claims]

    if not all_claims:
        logger.info("No claims to score.")
        return

    # Build architecture config
    arch_config = ArchitectureConfig(
        l4_threshold=l4_threshold,
        damping=damping,
    )

    # Optional: fit Platt scaling from prior scored data
    if calibrate:
        scored_path = claims_dir / "scored_claims.jsonl"
        if scored_path.exists():
            scored_raw = read_jsonl(scored_path)
            scored_decomps = [ClaimDecomposition.model_validate(d) for d in scored_raw]
            preds = [c.confidence_score for d in scored_decomps for c in d.claims]
            # Use evidence-based ground truth proxy
            from research_pipeline.confidence.scorer import compute_evidence_signal

            actuals = [
                compute_evidence_signal(c.evidence_class)
                for d in scored_decomps
                for c in d.claims
            ]
            if preds and actuals:
                params = fit_platt_scaling(preds, actuals)
                arch_config.platt_params = params
                logger.info("Platt scaling fitted: a=%.4f, b=%.4f", params.a, params.b)

    # Create LLM provider for L4
    llm_provider = create_llm_provider(config.llm)
    if llm_provider:
        logger.info("LLM available — L4 selective verification enabled")
    else:
        logger.info("No LLM — L4 verification disabled (heuristic-only)")

    # Score through 4-layer architecture
    logger.info("Scoring %d claims through 4-layer architecture...", len(all_claims))
    results = score_batch_layered(all_claims, arch_config, llm_provider)

    # Generate calibration report
    report = batch_calibration_report(results)

    # Write layered results
    output_path = claims_dir / "layered_confidence.jsonl"
    write_jsonl(output_path, [r.to_dict() for r in results])

    # Write calibration report
    report_path = claims_dir / "calibration_report.json"
    report_path.write_text(json.dumps(report.to_dict(), indent=2))

    # Summary
    scores = [r.final_score for r in results]
    avg = sum(scores) / len(scores) if scores else 0.0
    l4_count = sum(1 for r in results if r.l4.triggered)
    skip_count = sum(1 for r in results if r.l2.decision.value == "skip")
    decompose_count = sum(1 for r in results if r.l2.decision.value == "decompose")

    logger.info("4-Layer Confidence Results:")
    logger.info("  Claims scored: %d", len(results))
    logger.info("  Average confidence: %.4f", avg)
    logger.info("  L2 skipped (fast): %d", skip_count)
    logger.info("  L2 decomposed (deep): %d", decompose_count)
    logger.info("  L4 triggered (low-conf): %d", l4_count)
    logger.info(
        "  Calibration — ECE: %.4f, Brier: %.4f, AUROC: %.4f",
        report.ece,
        report.brier,
        report.auroc,
    )
    logger.info("  Output: %s", output_path)
    logger.info("  Report: %s", report_path)
