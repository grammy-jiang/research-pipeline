"""Docling-based PDF-to-Markdown converter backend.

Keywords: multi-account, account rotation, quota rotation.
"""

from __future__ import annotations

import logging

from research_pipeline.conversion.base import ConversionContext, ConverterBackend
from research_pipeline.conversion.registry import register_backend
from research_pipeline.models.conversion import ConvertManifestEntry

logger = logging.getLogger(__name__)


@register_backend("docling")
class DoclingBackend(ConverterBackend):
    """PDF-to-Markdown conversion using Docling (MIT license)."""

    def __init__(self, timeout_seconds: int = 300) -> None:
        self.timeout_seconds = timeout_seconds
        self._version: str | None = None

    @property
    def version(self) -> str:
        """Get the installed Docling version."""
        if self._version is None:
            try:
                from importlib.metadata import version

                self._version = version("docling")
            except Exception:
                self._version = "unknown"
        return self._version

    @property
    def converter_name(self) -> str:
        return "docling"

    @property
    def converter_version(self) -> str:
        return self.version

    def _config_string(self) -> str:
        return f"timeout={self.timeout_seconds}"

    def _run(self, ctx: ConversionContext) -> tuple[str, list[str]]:
        from docling.document_converter import (
            DocumentConverter,
        )

        converter = DocumentConverter()
        result = converter.convert(str(ctx.pdf_path))
        markdown_text = result.document.export_to_markdown()

        ctx.md_path.write_text(markdown_text, encoding="utf-8")
        logger.info("Converted %s → %s", ctx.pdf_path.name, ctx.md_path.name)
        return markdown_text, []

    def _on_convert_error(
        self, exc: Exception, ctx: ConversionContext
    ) -> ConvertManifestEntry:
        if isinstance(exc, ImportError):
            msg = (
                "Docling is not installed. Install with: "
                "pip install 'research-pipeline[docling]'"
            )
            logger.error(msg)
            return ctx.entry(
                "failed", markdown_path=str(ctx.md_path), warnings=[], error=msg
            )
        logger.error("Conversion failed for %s: %s", ctx.pdf_path.name, exc)
        return ctx.entry(
            "failed", markdown_path=str(ctx.md_path), warnings=[], error=str(exc)
        )
