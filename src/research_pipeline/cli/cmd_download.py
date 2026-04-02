"""CLI handler for the 'download' command."""

import json
import logging
from pathlib import Path

import typer

from research_pipeline.arxiv.rate_limit import ArxivRateLimiter
from research_pipeline.config.loader import load_config
from research_pipeline.download.pdf import download_batch
from research_pipeline.infra.http import create_session
from research_pipeline.models.screening import RelevanceDecision
from research_pipeline.storage.manifests import write_jsonl
from research_pipeline.storage.workspace import get_stage_dir, init_run

logger = logging.getLogger(__name__)


def run_download(
    force: bool = False,
    config_path: Path | None = None,
    workspace: Path | None = None,
    run_id: str | None = None,
) -> None:
    """Execute the download stage: download shortlisted PDFs.

    Args:
        force: Re-download even if cached.
        config_path: Path to config TOML.
        workspace: Workspace directory.
        run_id: Run ID with screened shortlist.
    """
    config = load_config(config_path)
    ws = workspace or Path(config.workspace)
    run_id_str, run_root = init_run(ws, run_id)

    shortlist_path = get_stage_dir(run_root, "screen") / "shortlist.json"
    if not shortlist_path.exists():
        typer.echo("Error: no shortlist found. Run 'screen' first.", err=True)
        raise typer.Exit(1)

    raw = json.loads(shortlist_path.read_text(encoding="utf-8"))
    shortlist = [RelevanceDecision.model_validate(d) for d in raw]

    papers = [
        {
            "arxiv_id": d.paper.arxiv_id,
            "version": d.paper.version,
            "pdf_url": d.paper.pdf_url,
        }
        for d in shortlist
        if d.download
    ]

    rate_limiter = ArxivRateLimiter(min_interval=config.arxiv.min_interval_seconds)
    session = create_session(config.contact_email)

    pdf_dir = get_stage_dir(run_root, "download") / "pdf"
    pdf_dir.mkdir(parents=True, exist_ok=True)

    entries = download_batch(
        papers,
        output_dir=pdf_dir,
        session=session,
        rate_limiter=rate_limiter,
        max_downloads=config.download.max_per_run,
    )

    manifest_path = get_stage_dir(run_root, "download_root") / "download_manifest.jsonl"
    write_jsonl(manifest_path, [e.model_dump(mode="json") for e in entries])

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
