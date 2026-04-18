"""CLI handler for the 'summarize' command."""

import json
import logging
from pathlib import Path

import typer

from research_pipeline.config.loader import load_config
from research_pipeline.models.conversion import ConvertManifestEntry
from research_pipeline.models.query_plan import QueryPlan
from research_pipeline.models.screening import RelevanceDecision
from research_pipeline.storage.manifests import read_jsonl
from research_pipeline.storage.workspace import get_stage_dir, init_run
from research_pipeline.summarization.per_paper import summarize_paper
from research_pipeline.summarization.synthesis import synthesize

logger = logging.getLogger(__name__)


def run_summarize(
    config_path: Path | None = None,
    workspace: Path | None = None,
    run_id: str | None = None,
    output_format: str = "markdown",
) -> None:
    """Execute the summarize stage: per-paper + cross-paper synthesis.

    Args:
        config_path: Path to config TOML.
        workspace: Workspace directory.
        run_id: Run ID with extracted data.
        output_format: Output format — markdown (default), json, bibtex,
            or structured-json.
    """
    config = load_config(config_path)
    ws = workspace or Path(config.workspace)
    run_id_str, run_root = init_run(ws, run_id)

    # Load plan for topic terms
    plan_path = get_stage_dir(run_root, "plan") / "query_plan.json"
    plan = (
        QueryPlan.model_validate(json.loads(plan_path.read_text(encoding="utf-8")))
        if plan_path.exists()
        else QueryPlan(topic_raw="", topic_normalized="")
    )

    # Load shortlist for titles
    shortlist_path = get_stage_dir(run_root, "screen") / "shortlist.json"
    shortlist: list[RelevanceDecision] = []
    if shortlist_path.exists():
        raw_sl = json.loads(shortlist_path.read_text(encoding="utf-8"))
        shortlist = [RelevanceDecision.model_validate(d) for d in raw_sl]

    title_map: dict[str, str] = {d.paper.arxiv_id: d.paper.title for d in shortlist}

    # Load convert manifest
    conv_path = get_stage_dir(run_root, "convert_root") / "convert_manifest.jsonl"
    if not conv_path.exists():
        typer.echo("Error: no convert manifest. Run 'convert' first.", err=True)
        raise typer.Exit(1)

    conv_entries = [
        ConvertManifestEntry.model_validate(d) for d in read_jsonl(conv_path)
    ]

    sum_dir = get_stage_dir(run_root, "summarize")
    summaries = []
    for entry in conv_entries:
        if entry.status not in ("converted", "skipped_exists"):
            continue
        md_path = Path(entry.markdown_path)
        if not md_path.exists():
            continue

        title = title_map.get(entry.arxiv_id, entry.arxiv_id)
        summary = summarize_paper(
            markdown_path=md_path,
            arxiv_id=entry.arxiv_id,
            version=entry.version,
            title=title,
            topic_terms=plan.must_terms + plan.nice_terms,
        )
        summaries.append(summary)
        out = sum_dir / f"{entry.arxiv_id}{entry.version}.summary.json"
        out.write_text(summary.model_dump_json(indent=2), encoding="utf-8")

    report = synthesize(summaries, plan.topic_raw)

    # Compute confidence metadata for synthesis claims using 4-layer architecture
    from research_pipeline.confidence.architecture import (
        ArchitectureConfig,
        score_claim_layered,
    )
    from research_pipeline.models.claim import AtomicClaim

    arch_config = ArchitectureConfig()
    confidence_results = []
    claim_idx = 0
    for agreement in report.agreements:
        claim = AtomicClaim(
            claim_id=f"SYN-A-{claim_idx:03d}",
            paper_id=",".join(agreement.supporting_papers[:3]),
            source_type="finding",
            statement=agreement.claim,
        )
        result = score_claim_layered(claim, config=arch_config)
        confidence_results.append(
            {
                "claim_id": claim.claim_id,
                "statement": claim.statement,
                "type": "agreement",
                "layered_confidence": result.final_score,
                "layers_used": result.layers_executed,
            }
        )
        claim_idx += 1

    for disagreement in report.disagreements:
        claim = AtomicClaim(
            claim_id=f"SYN-D-{claim_idx:03d}",
            paper_id=",".join(list(disagreement.positions.keys())[:3]),
            source_type="limitation",
            statement=disagreement.topic,
        )
        result = score_claim_layered(claim, config=arch_config)
        confidence_results.append(
            {
                "claim_id": claim.claim_id,
                "statement": claim.statement,
                "type": "disagreement",
                "layered_confidence": result.final_score,
                "layers_used": result.layers_executed,
            }
        )
        claim_idx += 1

    confidence_path = sum_dir / "synthesis_confidence.json"
    confidence_path.write_text(
        json.dumps(confidence_results, indent=2), encoding="utf-8"
    )
    logger.info(
        "Confidence scored %d synthesis claims → %s",
        len(confidence_results),
        confidence_path,
    )

    syn_path = sum_dir / "synthesis.json"
    syn_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")

    # Export additional formats if requested
    fmt = output_format.lower()
    if fmt in ("json", "bibtex"):
        from research_pipeline.summarization.export import export_report

        ext = "json" if fmt == "json" else "bib"
        export_path = sum_dir / f"synthesis_export.{ext}"
        export_report(report, export_path, fmt=fmt)
        typer.echo(f"Exported ({fmt}): {export_path}")
        logger.info("Exported synthesis as %s to %s", fmt, export_path)
    elif fmt == "structured-json":
        from research_pipeline.summarization.structured_output import (
            export_structured_json,
        )

        struct_path = sum_dir / "structured_evidence.json"
        export_structured_json(report, struct_path)
        typer.echo(f"Structured evidence: {struct_path}")
        logger.info("Exported structured evidence to %s", struct_path)

    typer.echo(f"Summarized: {len(summaries)} papers")
    typer.echo(f"Synthesis: {syn_path}")
    logger.info("Summarize stage complete: %d papers", len(summaries))
