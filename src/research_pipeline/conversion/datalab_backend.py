"""Datalab (hosted Marker) cloud backend for PDF-to-Markdown conversion.

Uses the Datalab REST API (https://documentation.datalab.to/) — the hosted
version of Marker. Requires ``DATALAB_API_KEY`` from
https://www.datalab.to/app/keys.
Free tier: $5 credit on signup (~1,250 pages on fast mode).

Keywords: multi-account, account rotation, quota rotation.
"""

from __future__ import annotations

import logging

from research_pipeline.conversion.base import ConversionContext, ConverterBackend
from research_pipeline.conversion.registry import register_backend

logger = logging.getLogger(__name__)


@register_backend("datalab")
class DatalabBackend(ConverterBackend):
    """Datalab (hosted Marker) cloud PDF-to-Markdown converter.

    Excellent quality, ~15 s per 250 pages.
    $4/1000 pages (fast/balanced), $6/1000 pages (accurate).
    $5 free credit on signup.
    """

    def __init__(
        self,
        *,
        api_key: str = "",
        mode: str = "balanced",
    ) -> None:
        if not api_key:
            raise ValueError(
                "Datalab backend requires 'api_key'. "
                "Set RESEARCH_PIPELINE_DATALAB_API_KEY or configure in config.toml."
            )
        if mode not in ("fast", "balanced", "accurate"):
            raise ValueError(
                f"Invalid Datalab mode {mode!r}. Must be fast, balanced, or accurate."
            )
        self.api_key = api_key
        self.mode = mode

    @property
    def converter_name(self) -> str:
        return "datalab"

    @property
    def converter_version(self) -> str:
        return "cloud"

    def _config_string(self) -> str:
        return f"mode={self.mode}"

    def _run(self, ctx: ConversionContext) -> tuple[str, list[str]]:
        from datalab_sdk import (  # type: ignore[import-not-found]
            ConvertOptions,
            DatalabClient,
        )

        logger.info(
            "Converting %s via Datalab API (mode=%s)...",
            ctx.pdf_path.name,
            self.mode,
        )
        client = DatalabClient(api_key=self.api_key)
        options = ConvertOptions(
            output_format="markdown",
            mode=self.mode,
        )
        result = client.convert(str(ctx.pdf_path), options=options)
        markdown_text: str = result.markdown

        ctx.md_path.write_text(markdown_text, encoding="utf-8")
        logger.info(
            "Converted %s → %s via Datalab", ctx.pdf_path.name, ctx.md_path.name
        )

        warnings: list[str] = []
        quality = getattr(result, "parse_quality_score", None)
        if quality is not None and quality < 3.0:
            warnings.append(f"Low parse quality score: {quality:.1f}/5.0")

        return markdown_text, warnings
