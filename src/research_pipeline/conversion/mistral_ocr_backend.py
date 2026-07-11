"""Mistral OCR cloud backend for PDF-to-Markdown conversion.

Uses the Mistral Document AI OCR API (https://docs.mistral.ai/capabilities/document_ai/)
with the ``mistral-ocr-latest`` model via direct HTTP requests. Requires
``MISTRAL_API_KEY`` from https://console.mistral.ai.
Per-token pricing with free API credits for new accounts.

Keywords: multi-account, account rotation, quota rotation.
"""

from __future__ import annotations

import base64
import logging

import requests

from research_pipeline.conversion.base import ConversionContext, ConverterBackend
from research_pipeline.conversion.registry import register_backend

logger = logging.getLogger(__name__)

_MISTRAL_OCR_URL = "https://api.mistral.ai/v1/ocr"
_REQUEST_TIMEOUT = 120  # seconds


@register_backend("mistral_ocr")
class MistralOcrBackend(ConverterBackend):
    """Mistral OCR cloud PDF-to-Markdown converter.

    Uses ``mistral-ocr-latest`` model via the Mistral Document AI API.
    Communicates via direct HTTPS requests — no external SDK required.
    Very good quality, fast, per-token pricing.
    """

    def __init__(
        self,
        *,
        api_key: str = "",
        model: str = "mistral-ocr-latest",
    ) -> None:
        if not api_key:
            raise ValueError(
                "Mistral OCR backend requires 'api_key'. "
                "Set RESEARCH_PIPELINE_MISTRAL_API_KEY or configure in config.toml."
            )
        self.api_key = api_key
        self.model = model

    @property
    def converter_name(self) -> str:
        return "mistral_ocr"

    @property
    def converter_version(self) -> str:
        return "cloud"

    def _config_string(self) -> str:
        return f"model={self.model}"

    def _run(self, ctx: ConversionContext) -> tuple[str, list[str]]:
        logger.info("Converting %s via Mistral OCR API...", ctx.pdf_path.name)

        pdf_bytes = ctx.pdf_path.read_bytes()
        pdf_b64 = base64.standard_b64encode(pdf_bytes).decode("ascii")

        response = requests.post(
            _MISTRAL_OCR_URL,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "document": {
                    "type": "document_url",
                    "document_url": f"data:application/pdf;base64,{pdf_b64}",
                },
                "include_image_base64": False,
            },
            timeout=_REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        ocr_data = response.json()

        pages = [p["markdown"] for p in ocr_data.get("pages", [])]
        markdown_text = "\n\n---\n\n".join(pages)

        ctx.md_path.write_text(markdown_text, encoding="utf-8")
        logger.info(
            "Converted %s → %s via Mistral OCR (%d pages)",
            ctx.pdf_path.name,
            ctx.md_path.name,
            len(pages),
        )

        return markdown_text, []
