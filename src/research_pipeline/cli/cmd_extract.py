"""CLI handler for the 'extract' command."""

import json
import logging
from pathlib import Path

import typer

from research_pipeline.config.loader import load_config
from research_pipeline.extraction.bibliography import extract_bibliography
from research_pipeline.extraction.extractor import extract_from_markdown
from research_pipeline.models.conversion import ConvertManifestEntry
from research_pipeline.storage.manifests import read_jsonl, write_jsonl
from research_pipeline.storage.workspace import get_stage_dir, init_run

logger = logging.getLogger(__name__)


def _discover_convert_manifest(
    run_root: Path,
) -> list[ConvertManifestEntry]:
    """Discover and merge conversion manifests from all available sources.

    Checks for manifests in this priority order:
    1. ``convert/convert_manifest.jsonl`` (standard single-tier)
    2. Two-tier: ``convert_rough_manifest.jsonl`` +
       ``convert_fine_manifest.jsonl`` (fine overrides rough)

    When two-tier manifests are found, they are merged into a unified
    ``convert/convert_manifest.jsonl`` for downstream stages.

    Args:
        run_root: Root directory of the pipeline run.

    Returns:
        List of ConvertManifestEntry.

    Raises:
        typer.Exit: If no conversion manifest is found anywhere.
    """
    # Standard path from single-tier convert
    std_path = get_stage_dir(run_root, "convert_root") / "convert_manifest.jsonl"
    if std_path.exists():
        raw = read_jsonl(std_path)
        return [ConvertManifestEntry.model_validate(d) for d in raw]

    # Two-tier: look for rough and/or fine manifests
    rough_manifest = "convert_rough_manifest.jsonl"
    rough_path = get_stage_dir(run_root, "convert_rough") / rough_manifest
    fine_path = get_stage_dir(run_root, "convert_fine") / "convert_fine_manifest.jsonl"

    has_rough = rough_path.exists()
    has_fine = fine_path.exists()

    if not has_rough and not has_fine:
        typer.echo(
            "Error: no convert manifest found. Run 'convert', 'convert-rough', "
            "or 'convert-fine' first.",
            err=True,
        )
        raise typer.Exit(1)

    # Merge: rough first, fine overrides by arxiv_id
    entries_by_id: dict[str, ConvertManifestEntry] = {}

    if has_rough:
        rough_raw = read_jsonl(rough_path)
        for d in rough_raw:
            entry = ConvertManifestEntry.model_validate(d)
            if entry.status in ("converted", "skipped_exists"):
                entries_by_id[entry.arxiv_id] = entry

    if has_fine:
        fine_raw = read_jsonl(fine_path)
        for d in fine_raw:
            entry = ConvertManifestEntry.model_validate(d)
            if entry.status in ("converted", "skipped_exists"):
                entries_by_id[entry.arxiv_id] = entry

    merged = list(entries_by_id.values())

    # Write unified manifest so downstream stages can find it
    unified_records = [e.model_dump(mode="json") for e in merged]
    write_jsonl(std_path, unified_records)

    tier_info = []
    if has_rough:
        tier_info.append("rough")
    if has_fine:
        tier_info.append("fine")
    logger.info(
        "Merged %d entries from %s tier(s) into %s",
        len(merged),
        "+".join(tier_info),
        std_path,
    )
    typer.echo(f"Auto-merged {len(merged)} entries from {'+'.join(tier_info)} tier(s)")

    return merged


def run_extract(
    config_path: Path | None = None,
    workspace: Path | None = None,
    run_id: str | None = None,
) -> None:
    """Execute the extract stage: structured content extraction.

    Args:
        config_path: Path to config TOML.
        workspace: Workspace directory.
        run_id: Run ID with converted Markdown.
    """
    config = load_config(config_path)
    ws = workspace or Path(config.workspace)
    run_id_str, run_root = init_run(ws, run_id)

    conv_entries = _discover_convert_manifest(run_root)

    extract_dir = get_stage_dir(run_root, "extract")
    count = 0
    bib_count = 0
    for entry in conv_entries:
        if entry.status not in ("converted", "skipped_exists"):
            continue
        md_path = Path(entry.markdown_path)
        if not md_path.exists():
            typer.echo(f"Warning: Markdown not found: {md_path}", err=True)
            continue

        extraction = extract_from_markdown(md_path, entry.arxiv_id, entry.version)
        out_path = extract_dir / f"{entry.arxiv_id}{entry.version}.extract.json"
        out_path.write_text(extraction.model_dump_json(indent=2), encoding="utf-8")
        count += 1

        # Extract bibliography entries
        md_text = md_path.read_text(encoding="utf-8")
        bib_entries = extract_bibliography(md_text)
        if bib_entries:
            bib_records = [
                {
                    "raw_text": e.raw_text,
                    "title": e.title,
                    "authors": e.authors,
                    "year": e.year,
                    "arxiv_id": e.arxiv_id,
                    "doi": e.doi,
                }
                for e in bib_entries
            ]
            bib_path = (
                extract_dir / f"{entry.arxiv_id}{entry.version}.bibliography.json"
            )
            bib_path.write_text(
                json.dumps(bib_records, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            bib_count += 1
            logger.info(
                "Extracted %d bibliography entries from %s",
                len(bib_entries),
                entry.arxiv_id,
            )

    typer.echo(f"Extracted: {count} papers ({bib_count} with bibliography)")
    logger.info(
        "Extract stage complete: %d papers, %d bibliographies", count, bib_count
    )
