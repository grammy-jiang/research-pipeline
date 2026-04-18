"""CLI handler for the 'convert' command."""

import logging
from pathlib import Path

import typer

from research_pipeline.config.loader import load_config
from research_pipeline.config.models import PipelineConfig
from research_pipeline.conversion.base import ConverterBackend
from research_pipeline.conversion.fallback import FallbackConverter
from research_pipeline.conversion.registry import (
    _ensure_builtins_registered,
    get_backend,
)
from research_pipeline.models.download import DownloadManifestEntry
from research_pipeline.storage.manifests import read_jsonl, write_jsonl
from research_pipeline.storage.workspace import get_stage_dir, init_run

logger = logging.getLogger(__name__)


def _backend_kwargs_list(
    backend_name: str,
    config: PipelineConfig,
) -> list[dict[str, object]]:
    """Build constructor kwargs for each account of a converter backend.

    Returns a list of kwargs dicts — one per account. If no multi-account
    config is set, returns a single-element list with the default credentials.
    """
    if backend_name == "docling":
        return [{"timeout_seconds": config.conversion.timeout_seconds}]
    if backend_name == "marker":
        mc = config.conversion.marker
        kwargs: dict[str, object] = {"force_ocr": mc.force_ocr}
        if mc.use_llm:
            kwargs["use_llm"] = True
            if mc.llm_service:
                kwargs["llm_service"] = mc.llm_service
            if mc.llm_api_key:
                kwargs["llm_api_key"] = mc.llm_api_key
        return [kwargs]
    if backend_name == "mathpix":
        mp = config.conversion.mathpix
        if mp.accounts:
            return [
                {"app_id": acct.app_id, "app_key": acct.app_key} for acct in mp.accounts
            ]
        return [{"app_id": mp.app_id, "app_key": mp.app_key}]
    if backend_name == "datalab":
        dl = config.conversion.datalab
        if dl.accounts:
            return [
                {"api_key": acct.api_key, "mode": acct.mode} for acct in dl.accounts
            ]
        return [{"api_key": dl.api_key, "mode": dl.mode}]
    if backend_name == "llamaparse":
        lp = config.conversion.llamaparse
        if lp.accounts:
            return [
                {"api_key": acct.api_key, "tier": acct.tier} for acct in lp.accounts
            ]
        return [{"api_key": lp.api_key, "tier": lp.tier}]
    if backend_name == "mistral_ocr":
        mo = config.conversion.mistral_ocr
        if mo.accounts:
            return [
                {"api_key": acct.api_key, "model": acct.model} for acct in mo.accounts
            ]
        return [{"api_key": mo.api_key, "model": mo.model}]
    if backend_name == "openai_vision":
        ov = config.conversion.openai_vision
        if ov.accounts:
            return [
                {"api_key": acct.api_key, "model": acct.model} for acct in ov.accounts
            ]
        return [{"api_key": ov.api_key, "model": ov.model}]
    if backend_name == "mineru":
        mn = config.conversion.mineru
        return [
            {
                "parse_method": mn.parse_method,
                "timeout_seconds": mn.timeout_seconds,
            }
        ]
    # pymupdf4llm and others: no special kwargs
    return [{}]


def _create_converter(config: PipelineConfig) -> ConverterBackend:
    """Create a converter backend from pipeline config.

    If ``fallback_backends`` is configured, creates a FallbackConverter wrapping
    all backends (primary + fallbacks) with all their accounts. Otherwise,
    creates the primary backend only (with multi-account fallback if configured).
    """
    _ensure_builtins_registered()

    primary = config.conversion.backend
    backend_names = [primary, *list(config.conversion.fallback_backends)]

    all_backends: list[ConverterBackend] = []
    for name in backend_names:
        kwargs_list = _backend_kwargs_list(name, config)
        for kwargs in kwargs_list:
            all_backends.append(get_backend(name, **kwargs))

    if len(all_backends) == 1:
        return all_backends[0]
    return FallbackConverter(all_backends)


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
