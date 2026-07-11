"""PyMuPDF4LLM-based PDF-to-Markdown converter backend.

A fast, lightweight converter (~10-50x faster than Docling) that trades
equation rendering quality for speed. Best for pre-screening or when
LaTeX equations are not critical.

Keywords: multi-account, account rotation, quota rotation.
"""

from __future__ import annotations

import logging

from research_pipeline.conversion.base import ConversionContext, ConverterBackend
from research_pipeline.conversion.registry import register_backend
from research_pipeline.models.conversion import ConvertManifestEntry

logger = logging.getLogger(__name__)


@register_backend("pymupdf4llm")
class PyMuPDF4LLMBackend(ConverterBackend):
    """PDF-to-Markdown conversion using pymupdf4llm (AGPL license).

    Very fast CPU-only converter. Good for quick screening.
    Does not render LaTeX equations.
    """

    def __init__(self, *, page_chunks: bool = False) -> None:
        self.page_chunks = page_chunks
        self._version: str | None = None

    @property
    def version(self) -> str:
        """Get the installed pymupdf4llm version."""
        if self._version is None:
            try:
                from importlib.metadata import version

                self._version = version("pymupdf4llm")
            except Exception:
                self._version = "unknown"
        return self._version

    @property
    def converter_name(self) -> str:
        return "pymupdf4llm"

    @property
    def converter_version(self) -> str:
        return self.version

    def _config_string(self) -> str:
        return f"page_chunks={self.page_chunks}"

    def _run(self, ctx: ConversionContext) -> tuple[str, list[str]]:
        import pymupdf4llm

        markdown_text = pymupdf4llm.to_markdown(
            str(ctx.pdf_path), page_chunks=self.page_chunks
        )

        ctx.md_path.write_text(markdown_text, encoding="utf-8")
        logger.info("Converted %s → %s", ctx.pdf_path.name, ctx.md_path.name)
        return markdown_text, []

    def _on_convert_error(
        self, exc: Exception, ctx: ConversionContext
    ) -> ConvertManifestEntry:
        if isinstance(exc, ImportError):
            msg = (
                "pymupdf4llm is not installed. Install with: "
                "pip install 'research-pipeline[pymupdf4llm]'"
            )
            logger.error(msg)
            return ctx.entry(
                "failed", markdown_path=str(ctx.md_path), warnings=[msg], error=msg
            )
        logger.error("pymupdf4llm conversion failed for %s: %s", ctx.pdf_path.name, exc)
        return ctx.entry(
            "failed",
            markdown_path=str(ctx.md_path),
            warnings=[str(exc)],
            error=str(exc),
        )
