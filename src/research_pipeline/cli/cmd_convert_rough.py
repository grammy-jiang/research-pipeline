"""CLI handler for the 'convert-rough' command.

Converts ALL downloaded PDFs using pymupdf4llm (fast, CPU-only).
This is Tier 2 in the two-tier conversion strategy.
"""

import logging
from pathlib import Path

import typer

from research_pipeline.config.loader import load_config
from research_pipeline.conversion.registry import (
    _ensure_builtins_registered,
    get_backend,
)
from research_pipeline.models.download import DownloadManifestEntry
from research_pipeline.storage.manifests import read_jsonl, write_jsonl
from research_pipeline.storage.workspace import get_stage_dir, init_run

logger = logging.getLogger(__name__)


def run_convert_rough(
    force: bool = False,
    config_path: Path | None = None,
    workspace: Path | None = None,
    run_id: str | None = None,
) -> None:
    """Execute rough conversion: all PDFs → Markdown via pymupdf4llm.

    Args:
        force: Re-convert even if output exists.
        config_path: Path to config TOML.
        workspace: Workspace directory.
        run_id: Run ID with downloaded PDFs.
    """
    config = load_config(config_path)
    ws = workspace or Path(config.workspace)
    _run_id, run_root = init_run(ws, run_id)

    dl_manifest_path = (
        get_stage_dir(run_root, "download_root") / "download_manifest.jsonl"
    )
    if not dl_manifest_path.exists():
        typer.echo("Error: no download manifest found. Run 'download' first.", err=True)
        raise typer.Exit(1)

    raw = read_jsonl(dl_manifest_path)
    entries = [DownloadManifestEntry.model_validate(d) for d in raw]

    _ensure_builtins_registered()
    converter = get_backend("pymupdf4llm")

    rough_dir = get_stage_dir(run_root, "convert_rough")
    rough_dir.mkdir(parents=True, exist_ok=True)

    results = []
    total = sum(1 for e in entries if e.status in ("downloaded", "skipped_exists"))
    done = 0
    for entry in entries:
        if entry.status not in ("downloaded", "skipped_exists"):
            continue
        done += 1
        pdf_path = Path(entry.local_path)
        if not pdf_path.exists():
            logger.warning("PDF not found: %s", pdf_path)
            continue

        logger.info(
            "Rough-converting PDF %d/%d (%s%s)...",
            done,
            total,
            entry.arxiv_id,
            entry.version,
        )
        result = converter.convert(pdf_path, rough_dir, force=force)
        results.append(result)

    manifest_path = rough_dir / "convert_rough_manifest.jsonl"
    records = [r.model_dump(mode="json") for r in results]
    # Add tier field to each record
    for rec in records:
        rec["tier"] = "rough"
    write_jsonl(records, manifest_path)

    converted = sum(1 for r in results if r.status == "converted")
    skipped = sum(1 for r in results if r.status == "skipped_exists")
    failed = sum(1 for r in results if r.status == "failed")

    typer.echo(f"Rough: {converted} converted, {skipped} skipped, {failed} failed")
    typer.echo(f"Manifest: {manifest_path}")
    logger.info(
        "Rough conversion complete: %d converted, %d skipped, %d failed",
        converted,
        skipped,
        failed,
    )
