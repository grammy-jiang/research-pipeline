"""CLI handler for the 'extract' command."""

import logging
from pathlib import Path

import typer

from research_pipeline.config.loader import load_config
from research_pipeline.extraction.extractor import extract_from_markdown
from research_pipeline.models.conversion import ConvertManifestEntry
from research_pipeline.storage.manifests import read_jsonl
from research_pipeline.storage.workspace import get_stage_dir, init_run

logger = logging.getLogger(__name__)


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

    conv_path = get_stage_dir(run_root, "convert_root") / "convert_manifest.jsonl"
    if not conv_path.exists():
        typer.echo("Error: no convert manifest found. Run 'convert' first.", err=True)
        raise typer.Exit(1)

    raw = read_jsonl(conv_path)
    conv_entries = [ConvertManifestEntry.model_validate(d) for d in raw]

    extract_dir = get_stage_dir(run_root, "extract")
    count = 0
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

    typer.echo(f"Extracted: {count} papers")
    logger.info("Extract stage complete: %d papers", count)
