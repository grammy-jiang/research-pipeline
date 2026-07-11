from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from research_pipeline.mcp_server.schemas import (
    BlindingAuditInput,
    CoherenceInput,
    ConsolidationInput,
    DualMetricsInput,
    EvaluateInput,
    ToolResult,
)
from research_pipeline.mcp_server.tools._common import (
    _raise_tool_error,
    _report_progress,
    _resolve_workspace,
)

if TYPE_CHECKING:
    from mcp.server.fastmcp import Context


def coherence_tool(
    params: CoherenceInput,
    ctx: Context | None = None,
) -> ToolResult:
    """Evaluate multi-session coherence across pipeline runs.

    Computes factual consistency, temporal ordering, knowledge update
    fidelity, and contradiction detection across 2+ runs.
    """
    try:
        from research_pipeline.pipeline.coherence import run_coherence

        ws = _resolve_workspace(params.workspace)
        report = run_coherence(
            run_ids=params.run_ids,
            workspace=ws,
        )

        return ToolResult(
            success=True,
            message=(
                f"Coherence evaluated across {len(params.run_ids)} runs: "
                f"overall={report.score.overall:.2f}, "
                f"contradictions={len(report.contradictions)}"
            ),
            artifacts=report.to_dict(),
        )
    except Exception as exc:
        _raise_tool_error("coherence evaluation", exc)


def consolidation_tool(
    params: ConsolidationInput,
    ctx: Context | None = None,
) -> ToolResult:
    """Consolidate cross-run memory: compress episodes, promote rules, prune stale.

    Implements episodic → semantic consolidation following SEA/MLMF
    three-tier memory architecture.
    """
    try:
        from dataclasses import asdict

        from research_pipeline.pipeline.consolidation import run_consolidation

        ws = _resolve_workspace(params.workspace)
        result = run_consolidation(
            workspace=ws,
            run_ids=params.run_ids,
            capacity=params.capacity,
            threshold=params.threshold,
            min_support=params.min_support,
            dry_run=params.dry_run,
        )

        return ToolResult(
            success=True,
            message=(
                f"Consolidation complete: "
                f"{result.episodes_before}→{result.episodes_after} episodes, "
                f"{result.rules_created} new rules, "
                f"{result.entries_pruned} pruned"
            ),
            artifacts=asdict(result),
        )
    except Exception as exc:
        _raise_tool_error("consolidation", exc)


def blinding_audit_tool(
    params: BlindingAuditInput,
    ctx: Context | None = None,
) -> ToolResult:
    """Run epistemic blinding audit to detect LLM prior contamination.

    Implements A/B blinding protocol from arXiv 2604.06013: scans analysis
    outputs for identifying feature references and scores contamination.
    """
    try:
        from research_pipeline.evaluation.blinding import (
            run_blinding_audit_for_workspace,
        )

        ws = Path(params.workspace)
        result = run_blinding_audit_for_workspace(
            ws,
            run_id=params.run_id or None,
            contamination_threshold=params.threshold,
            store_results=params.store_results,
        )

        return ToolResult(
            success=True,
            message=(
                f"Blinding audit complete for run {result.run_id}: "
                f"score={result.aggregate_score:.3f}, "
                f"{len(result.high_contamination_papers)} flagged papers. "
                f"{result.recommendation}"
            ),
            artifacts=result.to_dict(),
        )
    except Exception as exc:
        _raise_tool_error("blinding audit", exc)


def dual_metrics_tool(
    params: DualMetricsInput,
    ctx: Context | None = None,
) -> ToolResult:
    """Evaluate pipeline runs using Pass@k + Pass[k] dual metrics.

    Implements Claw-Eval framework (arXiv 2604.06132): computes capability
    ceiling (Pass@k) and reliability floor (Pass[k]) with safety gates.
    """
    try:
        from research_pipeline.evaluation.dual_metrics import evaluate_runs

        ws = Path(params.workspace)
        run_ids = params.run_ids if params.run_ids else None
        result = evaluate_runs(
            ws,
            params.query,
            run_ids=run_ids,
            k=params.k,
            store_results=params.store_results,
        )

        return ToolResult(
            success=True,
            message=(
                f"Dual metrics for '{result.query}': "
                f"Pass@{result.k}={result.gated_pass_at_k:.3f}, "
                f"Pass[{result.k}]={result.gated_pass_bracket_k:.3f}, "
                f"gap={result.pass_at_k - result.pass_bracket_k:.3f}, "
                f"safety={result.safety_gate:.1f}, "
                f"n={result.n}, c={result.c}"
            ),
            artifacts=result.to_dict(),
        )
    except Exception as exc:
        _raise_tool_error("dual metrics", exc)


def evaluate_tool(params: EvaluateInput, ctx: Context | None = None) -> ToolResult:
    """Evaluate pipeline outputs against their schemas."""
    try:
        from research_pipeline.evaluation.schema_eval import (
            evaluate_run,
            evaluate_stage,
        )

        ws = Path(params.workspace)
        run_root = ws / params.run_id

        if not run_root.exists():
            return ToolResult(
                success=False,
                message=f"Run not found: {run_root}",
            )

        _report_progress(ctx, 0, 2, "Evaluating")

        if params.stage:
            report = evaluate_stage(run_root, params.stage)
            reports = [report]
        else:
            reports = evaluate_run(run_root)

        _report_progress(ctx, 1, 2, "Building results")

        all_passed = all(r.passed for r in reports)
        results = []
        for r in reports:
            checks = []
            for c in r.checks:
                checks.append(
                    {
                        "name": c.name,
                        "passed": c.passed,
                        "description": c.description,
                        "details": c.details,
                        "severity": c.severity,
                    }
                )
            results.append(
                {
                    "stage": r.stage,
                    "passed": r.passed,
                    "error_count": r.error_count,
                    "warning_count": r.warning_count,
                    "checks": checks,
                }
            )

        _report_progress(ctx, 2, 2, "Done")
        verdict = "PASS" if all_passed else "FAIL"
        return ToolResult(
            success=True,
            message=f"Evaluation: {verdict} ({len(reports)} stage(s) checked).",
            artifacts={
                "verdict": verdict,
                "all_passed": all_passed,
                "stages": results,
            },
        )
    except Exception as exc:
        _raise_tool_error("evaluate", exc)
