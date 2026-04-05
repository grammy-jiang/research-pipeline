"""CLI handler for the 'convert-file' command (standalone PDF → Markdown)."""

import logging
from pathlib import Path

import typer

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
    return {}


def run_convert_file(
    pdf_path: Path,
    output_dir: Path | None = None,
    backend: str | None = None,
) -> None:
    """Convert a single PDF file to Markdown (standalone, no workspace needed).

    Args:
        pdf_path: Path to the source PDF file.
        output_dir: Directory to write output. Defaults to same directory as PDF.
        backend: Converter backend name. Defaults to config value.
    """
    from research_pipeline.config.loader import load_config
    from research_pipeline.conversion.registry import (
        _ensure_builtins_registered,
        get_backend,
    )

    _ensure_builtins_registered()

    pdf_path = pdf_path.expanduser().resolve()
    if not pdf_path.exists():
        typer.echo(f"Error: PDF file not found: {pdf_path}", err=True)
        raise typer.Exit(1)

    if pdf_path.suffix.lower() != ".pdf":
        typer.echo(f"Warning: file does not have .pdf extension: {pdf_path}", err=True)

    out = (output_dir or pdf_path.parent).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    config = load_config()
    backend_name = backend or config.conversion.backend
    kwargs = _backend_kwargs(backend_name, config)

    try:
        converter = get_backend(backend_name, **kwargs)
    except ValueError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from None

    logger.info("Converting %s → %s/ (backend=%s)", pdf_path, out, backend_name)

    try:
        result = converter.convert(pdf_path, out)
    except ImportError:
        typer.echo(
            f"Error: Backend {backend_name!r} is not installed. "
            f"Install with: pipx inject research-pipeline <package>",
            err=True,
        )
        raise typer.Exit(1) from None

    md_path = out / f"{pdf_path.stem}.md"
    if result.status in ("converted", "skipped_exists"):
        typer.echo(f"OK: {md_path}")
    else:
        typer.echo(f"Failed ({result.status}): {result.error or 'unknown'}", err=True)
        raise typer.Exit(1)
