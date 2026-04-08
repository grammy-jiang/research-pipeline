"""CLI handler for the 'convert-fine' command.

Converts agent-selected PDFs using the configured high-quality backend.
This is Tier 3 in the two-tier conversion strategy.
"""

import logging
from pathlib import Path

import typer

from research_pipeline.cli.cmd_convert import _create_converter
from research_pipeline.config.loader import load_config
from research_pipeline.models.download import DownloadManifestEntry
from research_pipeline.storage.manifests import read_jsonl, write_jsonl
from research_pipeline.storage.workspace import get_stage_dir, init_run

logger = logging.getLogger(__name__)


def run_convert_fine(
    paper_ids: list[str],
    force: bool = False,
    config_path: Path | None = None,
    workspace: Path | None = None,
    run_id: str | None = None,
    backend: str | None = None,
) -> None:
    """Execute fine conversion: selected PDFs → Markdown via high-quality backend.

    Args:
        paper_ids: arXiv IDs of papers to convert.
        force: Re-convert even if output exists.
        config_path: Path to config TOML.
        workspace: Workspace directory.
        run_id: Run ID with downloaded PDFs.
        backend: Backend name override.
    """
    if not paper_ids:
        typer.echo("Error: --paper-ids is required.", err=True)
        raise typer.Exit(1)

    config = load_config(config_path)
    if backend:
        config.conversion.backend = backend
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

    # Filter to requested paper IDs
    id_set = set(paper_ids)
    selected = [
        e
        for e in entries
        if e.arxiv_id in id_set and e.status in ("downloaded", "skipped_exists")
    ]

    if not selected:
        typer.echo("No matching downloaded papers found for given IDs.", err=True)
        raise typer.Exit(1)

    converter = _create_converter(config)

    fine_dir = get_stage_dir(run_root, "convert_fine")
    fine_dir.mkdir(parents=True, exist_ok=True)

    results = []
    total = len(selected)
    for idx, entry in enumerate(selected, 1):
        pdf_path = Path(entry.local_path)
        if not pdf_path.exists():
            logger.warning("PDF not found: %s", pdf_path)
            continue

        backend_name = getattr(converter, "name", config.conversion.backend)
        logger.info(
            "Fine-converting PDF %d/%d (%s%s) [%s]...",
            idx,
            total,
            entry.arxiv_id,
            entry.version,
            backend_name,
        )
        result = converter.convert(pdf_path, fine_dir, force=force)
        results.append(result)

    manifest_path = fine_dir / "convert_fine_manifest.jsonl"
    records = [r.model_dump(mode="json") for r in results]
    for rec in records:
        rec["tier"] = "fine"
    write_jsonl(manifest_path, records)

    converted = sum(1 for r in results if r.status == "converted")
    skipped = sum(1 for r in results if r.status == "skipped_exists")
    failed = sum(1 for r in results if r.status == "failed")

    typer.echo(f"Fine: {converted} converted, {skipped} skipped, {failed} failed")
    typer.echo(f"Manifest: {manifest_path}")
    logger.info(
        "Fine conversion complete: %d converted, %d skipped, %d failed",
        converted,
        skipped,
        failed,
    )
