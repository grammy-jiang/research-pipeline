"""CLI handler for the 'download' command."""

import json
import logging
from pathlib import Path

import typer

from research_pipeline.arxiv.rate_limit import ArxivRateLimiter
from research_pipeline.config.loader import load_config
from research_pipeline.download.pdf import download_batch
from research_pipeline.infra.http import create_session
from research_pipeline.models.download import DownloadManifestEntry
from research_pipeline.models.screening import (
    parse_shortlist_lenient,
)
from research_pipeline.storage.manifests import read_jsonl, write_jsonl
from research_pipeline.storage.workspace import get_stage_dir, init_run

logger = logging.getLogger(__name__)


def run_download(
    force: bool = False,
    config_path: Path | None = None,
    workspace: Path | None = None,
    run_id: str | None = None,
    retry_failed: bool = False,
) -> None:
    """Execute the download stage: download shortlisted PDFs.

    Args:
        force: Re-download even if cached.
        config_path: Path to config TOML.
        workspace: Workspace directory.
        run_id: Run ID with screened shortlist.
        retry_failed: When True, re-attempt only previously failed downloads
            (reads existing manifest; ignores papers that already succeeded).
    """
    config = load_config(config_path)
    ws = workspace or Path(config.workspace)
    run_id_str, run_root = init_run(ws, run_id)

    manifest_path = get_stage_dir(run_root, "download_root") / "download_manifest.jsonl"

    if retry_failed:
        # Retry-failed mode: read existing manifest and re-attempt failures only
        if not manifest_path.exists():
            typer.echo(
                "Error: no download manifest found. Run 'download' first.", err=True
            )
            raise typer.Exit(1)

        existing_entries = [
            DownloadManifestEntry.model_validate(r) for r in read_jsonl(manifest_path)
        ]
        failed_entries = [e for e in existing_entries if e.status == "failed"]
        if not failed_entries:
            typer.echo("No failed downloads found in manifest. Nothing to retry.")
            return

        # Build paper list from the failed manifest entries
        papers = [
            {
                "arxiv_id": e.arxiv_id,
                "version": e.version,
                "pdf_url": e.pdf_url,
            }
            for e in failed_entries
        ]
        # Keep successful entries; we will merge in the retry results below
        successful_entries = [e for e in existing_entries if e.status != "failed"]
        typer.echo(f"Retrying {len(papers)} previously failed download(s)…")
    else:
        shortlist_path = get_stage_dir(run_root, "screen") / "shortlist.json"
        if not shortlist_path.exists():
            typer.echo("Error: no shortlist found. Run 'screen' first.", err=True)
            raise typer.Exit(1)

        raw = json.loads(shortlist_path.read_text(encoding="utf-8"))
        shortlist = [parse_shortlist_lenient(d) for d in raw]

        papers = [
            {
                "arxiv_id": d.paper.arxiv_id,
                "version": d.paper.version,
                "pdf_url": d.paper.pdf_url,
            }
            for d in shortlist
            if d.download
        ]
        successful_entries = []

    rate_limiter = ArxivRateLimiter(min_interval=config.arxiv.min_interval_seconds)
    session = create_session(config.contact_email)

    pdf_dir = get_stage_dir(run_root, "download")
    pdf_dir.mkdir(parents=True, exist_ok=True)

    entries = download_batch(
        papers,
        output_dir=pdf_dir,
        session=session,
        rate_limiter=rate_limiter,
        max_downloads=config.download.max_per_run,
    )

    # Merge retry results back with previously-successful entries when applicable
    all_entries = successful_entries + list(entries)

    manifest_path = get_stage_dir(run_root, "download_root") / "download_manifest.jsonl"
    write_jsonl(manifest_path, [e.model_dump(mode="json") for e in all_entries])

    downloaded = sum(1 for e in entries if e.status == "downloaded")
    skipped = sum(1 for e in entries if e.status == "skipped_exists")
    failed = sum(1 for e in entries if e.status == "failed")

    typer.echo(f"Downloaded: {downloaded}, Skipped: {skipped}, Failed: {failed}")
    typer.echo(f"Manifest: {manifest_path}")
    logger.info(
        "Download stage complete: %d downloaded, %d skipped, %d failed",
        downloaded,
        skipped,
        failed,
    )
