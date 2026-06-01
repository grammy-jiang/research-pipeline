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
from pathlib import Path

import requests

from research_pipeline.conversion.base import ConverterBackend
from research_pipeline.conversion.registry import register_backend
from research_pipeline.infra.clock import utc_now
from research_pipeline.infra.hashing import sha256_file, sha256_str
from research_pipeline.models.conversion import ConvertManifestEntry

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

    def fingerprint(self) -> str:
        config_hash = sha256_str(f"model={self.model}")[:8]
        return f"mistral_ocr/cloud/{config_hash}"

    def convert(
        self, pdf_path: Path, output_dir: Path, *, force: bool = False
    ) -> ConvertManifestEntry:
        output_dir.mkdir(parents=True, exist_ok=True)
        md_filename = pdf_path.stem + ".md"
        md_path = output_dir / md_filename
        pdf_hash = sha256_file(pdf_path)

        stem = pdf_path.stem
        arxiv_id = stem
        version = "v1"
        if stem[-2] == "v" and stem[-1].isdigit():
            arxiv_id = stem[:-2]
            version = stem[-2:]

        if force and md_path.exists():
            logger.info("Force mode: removing existing %s", md_path)
            md_path.unlink()

        config_hash = sha256_str(f"model={self.model}")[:8]

        if md_path.exists():
            logger.info("Markdown already exists, skipping: %s", md_path)
            return ConvertManifestEntry(
                arxiv_id=arxiv_id,
                version=version,
                pdf_path=str(pdf_path),
                pdf_sha256=pdf_hash,
                markdown_path=str(md_path),
                converter_name="mistral_ocr",
                converter_version="cloud",
                converter_config_hash=config_hash,
                converted_at=utc_now(),
                warnings=[],
                status="skipped_exists",
            )

        try:
            logger.info("Converting %s via Mistral OCR API...", pdf_path.name)

            pdf_bytes = pdf_path.read_bytes()
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

            md_path.write_text(markdown_text, encoding="utf-8")
            logger.info(
                "Converted %s → %s via Mistral OCR (%d pages)",
                pdf_path.name,
                md_path.name,
                len(pages),
            )

            return ConvertManifestEntry(
                arxiv_id=arxiv_id,
                version=version,
                pdf_path=str(pdf_path),
                pdf_sha256=pdf_hash,
                markdown_path=str(md_path),
                converter_name="mistral_ocr",
                converter_version="cloud",
                converter_config_hash=config_hash,
                converted_at=utc_now(),
                warnings=[],
                status="converted",
            )
        except Exception as exc:
            logger.error("Mistral OCR conversion failed for %s: %s", pdf_path.name, exc)
            return ConvertManifestEntry(
                arxiv_id=arxiv_id,
                version=version,
                pdf_path=str(pdf_path),
                pdf_sha256=pdf_hash,
                markdown_path="",
                converter_name="mistral_ocr",
                converter_version="cloud",
                converter_config_hash=config_hash,
                converted_at=utc_now(),
                warnings=[str(exc)],
                status="failed",
                error=str(exc),
            )
