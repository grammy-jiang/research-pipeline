"""CLI handler for the 'convert-file' command (standalone PDF → Markdown)."""

import logging
from pathlib import Path

import typer

logger = logging.getLogger(__name__)


def run_convert_file(
    pdf_path: Path,
    output_dir: Path | None = None,
) -> None:
    """Convert a single PDF file to Markdown (standalone, no workspace needed).

    Args:
        pdf_path: Path to the source PDF file.
        output_dir: Directory to write output. Defaults to same directory as PDF.
    """
    from arxiv_paper_pipeline.conversion.docling_backend import DoclingBackend

    pdf_path = pdf_path.expanduser().resolve()
    if not pdf_path.exists():
        typer.echo(f"Error: PDF file not found: {pdf_path}", err=True)
        raise typer.Exit(1)

    if pdf_path.suffix.lower() != ".pdf":
        typer.echo(f"Warning: file does not have .pdf extension: {pdf_path}", err=True)

    out = (output_dir or pdf_path.parent).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    backend = DoclingBackend()
    logger.info("Converting %s → %s/", pdf_path, out)

    try:
        result = backend.convert(pdf_path, out)
    except ImportError:
        typer.echo(
            "Error: Docling is not installed. "
            "Install with: pipx inject arxiv-paper-pipeline docling",
            err=True,
        )
        raise typer.Exit(1) from None

    md_path = out / f"{pdf_path.stem}.md"
    if result.status in ("converted", "skipped_exists"):
        typer.echo(f"OK: {md_path}")
    else:
        typer.echo(f"Failed ({result.status}): {result.error or 'unknown'}", err=True)
        raise typer.Exit(1)
