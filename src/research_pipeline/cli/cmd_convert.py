"""CLI handler for the 'convert' command."""

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


def _backend_kwargs(
    backend_name: str,
    config: "PipelineConfig",  # noqa: F821
) -> dict[str, object]:
    """Build constructor kwargs for a converter backend from config."""
    if backend_name == "docling":
        return {"timeout_seconds": config.conversion.timeout_seconds}
    if backend_name == "marker":
        mc = config.conversion.marker
        kwargs: dict[str, object] = {"force_ocr": mc.force_ocr}
        if mc.use_llm:
            kwargs["use_llm"] = True
            if mc.llm_service:
                kwargs["llm_service"] = mc.llm_service
            if mc.llm_api_key:
                kwargs["llm_api_key"] = mc.llm_api_key
        return kwargs
    # pymupdf4llm and others: no special kwargs
    return {}


def _create_converter(config: "PipelineConfig") -> "ConverterBackend":  # noqa: F821
    """Create a converter backend from pipeline config."""
    _ensure_builtins_registered()
    backend_name = config.conversion.backend
    kwargs = _backend_kwargs(backend_name, config)
    return get_backend(backend_name, **kwargs)


def run_convert(
    force: bool = False,
    config_path: Path | None = None,
    workspace: Path | None = None,
    run_id: str | None = None,
    backend: str | None = None,
) -> None:
    """Execute the convert stage: PDF → Markdown.

    Args:
        force: Re-convert even if output exists.
        config_path: Path to config TOML.
        workspace: Workspace directory.
        run_id: Run ID with downloaded PDFs.
        backend: Converter backend name override.
    """
    config = load_config(config_path)
    if backend:
        config.conversion.backend = backend
    ws = workspace or Path(config.workspace)
    run_id_str, run_root = init_run(ws, run_id)

    dl_manifest_path = (
        get_stage_dir(run_root, "download_root") / "download_manifest.jsonl"
    )
    if not dl_manifest_path.exists():
        typer.echo("Error: no download manifest found. Run 'download' first.", err=True)
        raise typer.Exit(1)

    raw = read_jsonl(dl_manifest_path)
    entries = [DownloadManifestEntry.model_validate(d) for d in raw]

    converter = _create_converter(config)
    md_dir = get_stage_dir(run_root, "convert")

    results = []
    for entry in entries:
        if entry.status not in ("downloaded", "skipped_exists"):
            continue
        pdf_path = Path(entry.local_path)
        if not pdf_path.exists():
            typer.echo(f"Warning: PDF not found: {pdf_path}", err=True)
            continue
        result = converter.convert(pdf_path, md_dir, force=force)
        results.append(result)

    conv_path = get_stage_dir(run_root, "convert_root") / "convert_manifest.jsonl"
    write_jsonl(conv_path, [r.model_dump(mode="json") for r in results])

    converted = sum(1 for r in results if r.status == "converted")
    skipped = sum(1 for r in results if r.status == "skipped_exists")
    failed = sum(1 for r in results if r.status == "failed")

    typer.echo(f"Converted: {converted}, Skipped: {skipped}, Failed: {failed}")
    typer.echo(f"Manifest: {conv_path}")
    logger.info(
        "Convert stage complete: %d converted, %d skipped, %d failed",
        converted,
        skipped,
        failed,
    )
