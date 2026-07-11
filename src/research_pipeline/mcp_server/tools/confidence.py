from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from research_pipeline.mcp_server.schemas import (
    AdaptiveStoppingInput,
    AnalyzeClaimsInput,
    ConfidenceLayersInput,
    ScoreClaimsInput,
    ToolResult,
)
from research_pipeline.mcp_server.tools._common import (
    _raise_tool_error,
    _report_progress,
)

if TYPE_CHECKING:
    from mcp.server.fastmcp import Context


def adaptive_stopping_tool(
    params: AdaptiveStoppingInput,
    ctx: Context | None = None,
) -> ToolResult:
    """Evaluate query-adaptive retrieval stopping criteria.

    Three strategies based on query type (HingeMem WWW '26):
    - recall: knee detection on cumulative relevance
    - precision: top-k saturation check
    - judgment: top-1 stability across batches
    Plus score plateau backstop and budget limits.
    """
    try:
        from research_pipeline.screening.adaptive_stopping import (
            BatchScores,
            QueryType,
            StoppingState,
            evaluate_stopping,
        )

        try:
            qtype = QueryType(params.query_type.lower())
        except ValueError:
            return ToolResult(
                success=False,
                message=f"Invalid query_type: {params.query_type}",
            )

        state = StoppingState(
            query_type=qtype,
            max_budget=params.max_budget,
            min_results=params.min_results,
            relevance_threshold=params.relevance_threshold,
        )
        for i, batch in enumerate(params.batch_scores):
            state.batches.append(BatchScores(i, [float(s) for s in batch]))

        decision = evaluate_stopping(state, query=params.query or None)

        return ToolResult(
            success=True,
            message=(
                f"Stopping: {'STOP' if decision.should_stop else 'CONTINUE'} "
                f"({decision.reason.value}) — {decision.details}"
            ),
            artifacts={
                "should_stop": decision.should_stop,
                "reason": decision.reason.value,
                "details": decision.details,
                "batches_processed": decision.batches_processed,
                "total_results": decision.total_results,
                "current_score": decision.current_score,
            },
        )

    except Exception as exc:
        _raise_tool_error("Adaptive stopping evaluation", exc)


def confidence_layers_tool(
    params: ConfidenceLayersInput,
    ctx: Context | None = None,
) -> ToolResult:
    """Score claims through the 4-layer confidence architecture.

    L1 (fast signal) → L2 (adaptive granularity) → L3 (DINCO calibration)
    → L4 (selective verification). Based on Atomic Calibration, AGSC,
    DINCO, and LoVeC research.
    """
    try:
        from pathlib import Path

        from research_pipeline.confidence.layers_runner import (
            run_confidence_layers,
        )

        run_confidence_layers(
            config_path=Path(params.config_path) if params.config_path else None,
            workspace=Path(params.workspace) if params.workspace else None,
            run_id=params.run_id,
            l4_threshold=params.l4_threshold,
            damping=params.damping,
            calibrate=params.calibrate,
        )

        return ToolResult(
            success=True,
            message=(
                f"4-layer confidence scoring completed for run {params.run_id}. "
                f"L4 threshold={params.l4_threshold}, damping={params.damping}."
            ),
            artifacts={
                "run_id": params.run_id,
                "l4_threshold": params.l4_threshold,
                "damping": params.damping,
                "calibrate": params.calibrate,
            },
        )

    except Exception as exc:
        _raise_tool_error("Confidence layers scoring", exc)


def analyze_claims_tool(
    params: AnalyzeClaimsInput, ctx: Context | None = None
) -> ToolResult:
    """Decompose paper summaries into atomic claims with evidence classification."""
    try:
        from research_pipeline.analysis.decomposer import decompose_paper
        from research_pipeline.config.loader import load_config
        from research_pipeline.models.summary import PaperSummary
        from research_pipeline.storage.manifests import read_jsonl, write_jsonl
        from research_pipeline.storage.workspace import get_stage_dir, init_run

        config = load_config()
        ws = Path(params.workspace) if params.workspace else Path(config.workspace)
        run_id_str, run_root = init_run(ws, params.run_id or None)

        _report_progress(ctx, 0, 3, "Loading summaries")

        summary_dir = get_stage_dir(run_root, "summarize")
        summary_path = summary_dir / "paper_summaries.jsonl"
        if summary_path.exists():
            raw = read_jsonl(summary_path)
        else:
            summary_files = list(summary_dir.glob("*.summary.json"))
            if not summary_files:
                return ToolResult(
                    success=False,
                    message=(
                        "No paper summaries found. Run tool_summarize_papers first."
                    ),
                )
            raw = []
            for sf in summary_files:
                raw.append(json.loads(sf.read_text(encoding="utf-8")))

        summaries = [PaperSummary.model_validate(d) for d in raw]
        md_dir = get_stage_dir(run_root, "convert")

        _report_progress(ctx, 1, 3, "Decomposing claims")

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

        _report_progress(ctx, 2, 3, "Writing results")

        claims_dir = summary_dir / "claims"
        claims_dir.mkdir(parents=True, exist_ok=True)
        output_path = claims_dir / "claim_decomposition.jsonl"
        write_jsonl(output_path, [r.model_dump(mode="json") for r in results])

        total_claims = sum(r.total_claims for r in results)
        total_supported = sum(r.evidence_summary.get("supported", 0) for r in results)

        _report_progress(ctx, 3, 3, "Done")
        return ToolResult(
            success=True,
            message=(
                f"Decomposed {len(results)} papers into {total_claims} claims "
                f"({total_supported} supported)."
            ),
            artifacts={
                "output": str(output_path),
                "papers": len(results),
                "total_claims": total_claims,
                "supported": total_supported,
            },
        )
    except Exception as exc:
        _raise_tool_error("analyze_claims", exc)


def score_claims_tool(
    params: ScoreClaimsInput, ctx: Context | None = None
) -> ToolResult:
    """Score confidence for decomposed claims."""
    try:
        from research_pipeline.confidence.scorer import score_decomposition
        from research_pipeline.config.loader import load_config
        from research_pipeline.llm.providers import create_llm_provider
        from research_pipeline.models.claim import ClaimDecomposition
        from research_pipeline.storage.manifests import read_jsonl, write_jsonl
        from research_pipeline.storage.workspace import get_stage_dir, init_run

        config = load_config()
        ws = Path(params.workspace) if params.workspace else Path(config.workspace)
        run_id_str, run_root = init_run(ws, params.run_id or None)

        claims_dir = get_stage_dir(run_root, "summarize") / "claims"
        claims_path = claims_dir / "claim_decomposition.jsonl"
        if not claims_path.exists():
            return ToolResult(
                success=False,
                message="No claim decompositions found. Run tool_analyze_claims first.",
            )

        _report_progress(ctx, 0, 3, "Loading decompositions")
        raw = read_jsonl(claims_path)
        decompositions = [ClaimDecomposition.model_validate(d) for d in raw]

        llm_provider = create_llm_provider(config.llm)

        _report_progress(ctx, 1, 3, "Scoring claims")
        results = []
        for decomp in decompositions:
            scored = score_decomposition(decomp, llm_provider)
            results.append(scored)

        _report_progress(ctx, 2, 3, "Writing results")
        output_path = claims_dir / "scored_claims.jsonl"
        write_jsonl(output_path, [r.model_dump(mode="json") for r in results])

        total_claims = sum(len(r.claims) for r in results)
        avg_confidence = sum(
            c.confidence_score for r in results for c in r.claims
        ) / max(total_claims, 1)

        _report_progress(ctx, 3, 3, "Done")
        return ToolResult(
            success=True,
            message=(
                f"Scored {total_claims} claims across {len(results)} papers. "
                f"Average confidence: {avg_confidence:.3f}."
            ),
            artifacts={
                "output": str(output_path),
                "papers": len(results),
                "total_claims": total_claims,
                "avg_confidence": round(avg_confidence, 3),
                "llm_available": llm_provider is not None,
            },
        )
    except Exception as exc:
        _raise_tool_error("score_claims", exc)
