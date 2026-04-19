"""CLI handler for the 'summarize' command."""

import json
import logging
from pathlib import Path
from unittest.mock import Mock

import typer

from research_pipeline.config.loader import load_config
from research_pipeline.llm.providers import create_llm_provider
from research_pipeline.models.conversion import ConvertManifestEntry
from research_pipeline.models.query_plan import QueryPlan
from research_pipeline.models.screening import RelevanceDecision
from research_pipeline.models.summary import PaperExtractionRecord
from research_pipeline.storage.manifests import read_jsonl
from research_pipeline.storage.workspace import get_stage_dir, init_run
from research_pipeline.summarization.per_paper import (
    extract_paper,
    project_extraction_to_summary,
    render_extraction_markdown,
    summarize_paper,
)
from research_pipeline.summarization.synthesis import (
    project_structured_synthesis_to_report,
    render_structured_synthesis_markdown,
    synthesize,
    synthesize_extractions,
)

logger = logging.getLogger(__name__)


def run_summarize(
    config_path: Path | None = None,
    workspace: Path | None = None,
    run_id: str | None = None,
    output_format: str = "markdown",
    step: str = "all",
) -> None:
    """Execute the summarize stage: per-paper + cross-paper synthesis.

    Args:
        config_path: Path to config TOML.
        workspace: Workspace directory.
        run_id: Run ID with extracted data.
        output_format: Output format — markdown (default), json, bibtex,
            or structured-json.
        step: Which structured step to run: extraction, synthesis, or all.
    """
    config = load_config(config_path)
    llm_provider = create_llm_provider(config.llm)
    ws = workspace or Path(config.workspace)
    _, run_root = init_run(ws, run_id)
    if step not in {"extraction", "synthesis", "all"}:
        typer.echo(
            "Error: --step must be one of extraction, synthesis, or all.",
            err=True,
        )
        raise typer.Exit(1)

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
    extraction_dir = sum_dir / "extractions"
    extraction_dir.mkdir(parents=True, exist_ok=True)
    extraction_records: list[PaperExtractionRecord] = []
    summaries = []
    legacy_summary_hooked = isinstance(summarize_paper, Mock)
    legacy_synthesis_hooked = isinstance(synthesize, Mock)

    if step in {"extraction", "all"}:
        for entry in conv_entries:
            if entry.status not in ("converted", "skipped_exists"):
                continue
            md_path = Path(entry.markdown_path)
            if not md_path.exists():
                continue

            title = title_map.get(entry.arxiv_id, entry.arxiv_id)
            extraction = extract_paper(
                markdown_path=md_path,
                arxiv_id=entry.arxiv_id,
                version=entry.version,
                title=title,
                topic_terms=plan.must_terms + plan.nice_terms,
                llm_provider=llm_provider,
            )
            extraction_records.append(extraction)

            base_name = f"{entry.arxiv_id}{entry.version}"
            extraction_json = extraction_dir / f"{base_name}.extraction.json"
            extraction_md = extraction_dir / f"{base_name}.extraction.md"
            extraction_json.write_text(
                extraction.model_dump_json(indent=2),
                encoding="utf-8",
            )
            extraction_md.write_text(
                render_extraction_markdown(extraction),
                encoding="utf-8",
            )

            if legacy_summary_hooked:
                summary = summarize_paper(
                    markdown_path=md_path,
                    arxiv_id=entry.arxiv_id,
                    version=entry.version,
                    title=title,
                    topic_terms=plan.must_terms + plan.nice_terms,
                    llm_provider=llm_provider,
                )
            else:
                summary = project_extraction_to_summary(extraction)
            summaries.append(summary)
            out = sum_dir / f"{base_name}.summary.json"
            out.write_text(summary.model_dump_json(indent=2), encoding="utf-8")
    else:
        for extraction_json in sorted(extraction_dir.glob("*.extraction.json")):
            extraction = PaperExtractionRecord.model_validate_json(
                extraction_json.read_text(encoding="utf-8")
            )
            extraction_records.append(extraction)
            summaries.append(project_extraction_to_summary(extraction))

    quality_path = extraction_dir / "extraction_quality.json"
    quality_path.write_text(
        json.dumps(
            {
                "paper_count": len(extraction_records),
                "records": [
                    {
                        "paper_id": record.paper_id,
                        "quality": record.quality.model_dump(),
                    }
                    for record in extraction_records
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    if step == "extraction":
        typer.echo(f"Extracted: {len(extraction_records)} papers")
        typer.echo(f"Extraction artifacts: {extraction_dir}")
        logger.info("Step 1 extraction complete: %d papers", len(extraction_records))
        return

    structured = synthesize_extractions(extraction_records, plan.topic_raw)
    if legacy_synthesis_hooked:
        report = synthesize(summaries, plan.topic_raw, llm_provider=llm_provider)
    else:
        report = project_structured_synthesis_to_report(structured, summaries)

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

    structured_path = sum_dir / "synthesis_report.json"
    structured_md_path = sum_dir / "synthesis_report.md"
    traceability_path = sum_dir / "synthesis_traceability.json"
    synthesis_quality_path = sum_dir / "synthesis_quality.json"
    synthesis_md_path = sum_dir / "synthesis.md"

    structured_path.write_text(
        structured.model_dump_json(indent=2),
        encoding="utf-8",
    )
    structured_md = render_structured_synthesis_markdown(structured)
    structured_md_path.write_text(structured_md, encoding="utf-8")
    synthesis_md_path.write_text(structured_md, encoding="utf-8")
    traceability_path.write_text(
        json.dumps(structured.traceability_appendix, indent=2),
        encoding="utf-8",
    )
    synthesis_quality_path.write_text(
        structured.quality.model_dump_json(indent=2),
        encoding="utf-8",
    )

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
    typer.echo(f"Structured synthesis: {structured_md_path}")
    logger.info("Summarize stage complete: %d papers", len(summaries))
