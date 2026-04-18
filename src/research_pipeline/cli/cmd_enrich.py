"""CLI handler for the 'enrich' command.

Enriches candidate papers with missing abstracts and metadata by
querying Semantic Scholar (by DOI or title).
"""

import json
import logging
from pathlib import Path

import typer

from research_pipeline.config.loader import load_config
from research_pipeline.models.candidate import CandidateRecord
from research_pipeline.sources.enrichment import enrich_candidates
from research_pipeline.storage.manifests import read_jsonl, write_jsonl
from research_pipeline.storage.workspace import get_stage_dir

logger = logging.getLogger(__name__)


def enrich_command(
    run_id: str = typer.Option(..., help="Run ID to enrich candidates for"),
    stage: str = typer.Option(
        "candidates",
        help="Stage to read candidates from: 'candidates' or 'screened'",
    ),
    config_path: Path = typer.Option(
        Path("config.toml"),
        "--config",
        help="Path to config file",
    ),
) -> None:
    """Enrich candidates with missing abstracts/metadata from Semantic Scholar."""
    config = load_config(config_path)
    run_dir = Path(config.workspace) / run_id

    if not run_dir.exists():
        logger.error("Run directory not found: %s", run_dir)
        raise typer.Exit(code=1)

    if stage == "screened":
        stage_dir = get_stage_dir(run_dir, "screen")
        jsonl_file = stage_dir / "screened.jsonl"
    else:
        stage_dir = get_stage_dir(run_dir, "search")
        jsonl_file = stage_dir / "candidates.jsonl"

    if not jsonl_file.exists():
        logger.error("Candidates file not found: %s", jsonl_file)
        raise typer.Exit(code=1)

    records = [CandidateRecord.model_validate(d) for d in read_jsonl(jsonl_file)]
    logger.info(
        "Loaded %d candidates from %s",
        len(records),
        jsonl_file.name,
    )

    missing_abstract = sum(1 for r in records if not r.abstract)
    missing_citations = sum(1 for r in records if r.citation_count is None)
    logger.info(
        "Missing: %d abstracts, %d citation counts",
        missing_abstract,
        missing_citations,
    )

    s2_api_key = getattr(config, "semantic_scholar_api_key", "") or ""

    enriched_count = enrich_candidates(
        records,
        s2_api_key=s2_api_key,
    )

    output_file = stage_dir / f"{jsonl_file.stem}_enriched.jsonl"
    write_jsonl(output_file, [r.model_dump(mode="json") for r in records])

    summary = {
        "total_candidates": len(records),
        "enriched_count": enriched_count,
        "missing_abstracts_before": missing_abstract,
        "missing_abstracts_after": sum(1 for r in records if not r.abstract),
        "missing_citations_before": missing_citations,
        "missing_citations_after": sum(1 for r in records if r.citation_count is None),
    }

    summary_file = stage_dir / "enrichment_summary.json"
    summary_file.write_text(json.dumps(summary, indent=2))

    logger.info(
        "Enrichment complete: %d/%d candidates updated. Output: %s",
        enriched_count,
        len(records),
        output_file,
    )
