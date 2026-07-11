"""Marker-based PDF-to-Markdown converter backend.

Keywords: multi-account, account rotation, quota rotation.
"""

from __future__ import annotations

import logging

from research_pipeline.conversion.base import ConversionContext, ConverterBackend
from research_pipeline.conversion.registry import register_backend
from research_pipeline.models.conversion import ConvertManifestEntry

logger = logging.getLogger(__name__)


@register_backend("marker")
class MarkerBackend(ConverterBackend):
    """PDF-to-Markdown conversion using Marker (GPL-3.0 code, Open Rail-M models).

    Supports optional LLM-assisted conversion for improved accuracy on
    complex layouts and equations.
    """

    def __init__(
        self,
        *,
        force_ocr: bool = False,
        use_llm: bool = False,
        llm_service: str | None = None,
        llm_api_key: str | None = None,
    ) -> None:
        self.force_ocr = force_ocr
        self.use_llm = use_llm
        self.llm_service = llm_service
        self.llm_api_key = llm_api_key
        self._version: str | None = None

    @property
    def version(self) -> str:
        """Get the installed Marker version."""
        if self._version is None:
            try:
                from importlib.metadata import version

                self._version = version("marker-pdf")
            except Exception:
                self._version = "unknown"
        return self._version

    @property
    def converter_name(self) -> str:
        return "marker"

    @property
    def converter_version(self) -> str:
        return self.version

    def _config_string(self) -> str:
        return (
            f"force_ocr={self.force_ocr}|use_llm={self.use_llm}"
            f"|llm_service={self.llm_service or 'none'}"
        )

    def _run(self, ctx: ConversionContext) -> tuple[str, list[str]]:
        from marker.converters.pdf import (  # type: ignore[import-not-found]
            PdfConverter,
        )
        from marker.models import (  # type: ignore[import-not-found]
            create_model_dict,
        )

        models = create_model_dict()

        converter_kwargs: dict[str, object] = {}
        if self.force_ocr:
            converter_kwargs["force_ocr"] = True
        if self.use_llm and self.llm_service:
            converter_kwargs["use_llm"] = True
            converter_kwargs["llm_service"] = self.llm_service

        converter = PdfConverter(artifact_dict=models, **converter_kwargs)
        rendered = converter(str(ctx.pdf_path))
        markdown_text = rendered.markdown

        ctx.md_path.write_text(markdown_text, encoding="utf-8")
        logger.info("Converted %s → %s", ctx.pdf_path.name, ctx.md_path.name)
        return markdown_text, []

    def _on_convert_error(
        self, exc: Exception, ctx: ConversionContext
    ) -> ConvertManifestEntry:
        if isinstance(exc, ImportError):
            msg = (
                "Marker is not installed. Install with: "
                "pip install 'research-pipeline[marker]'"
            )
            logger.error(msg)
            return ctx.entry(
                "failed", markdown_path=str(ctx.md_path), warnings=[msg], error=msg
            )
        logger.error("Marker conversion failed for %s: %s", ctx.pdf_path.name, exc)
        return ctx.entry(
            "failed",
            markdown_path=str(ctx.md_path),
            warnings=[str(exc)],
            error=str(exc),
        )
