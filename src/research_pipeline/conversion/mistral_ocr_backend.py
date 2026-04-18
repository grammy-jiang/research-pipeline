"""Mistral OCR cloud backend for PDF-to-Markdown conversion.

Uses the Mistral Document AI OCR API (https://docs.mistral.ai/capabilities/document_ai/)
with the ``mistral-ocr-latest`` model. Requires ``MISTRAL_API_KEY`` from
https://console.mistral.ai.
Per-token pricing with free API credits for new accounts.
"""

from __future__ import annotations

import logging
from pathlib import Path

from research_pipeline.conversion.base import ConverterBackend
from research_pipeline.conversion.registry import register_backend
from research_pipeline.infra.clock import utc_now
from research_pipeline.infra.hashing import sha256_file, sha256_str
from research_pipeline.models.conversion import ConvertManifestEntry

logger = logging.getLogger(__name__)


@register_backend("mistral_ocr")
class MistralOcrBackend(ConverterBackend):
    """Mistral OCR cloud PDF-to-Markdown converter.

    Uses ``mistral-ocr-latest`` model via the Mistral Document AI API.
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
            from mistralai.client import Mistral  # type: ignore[import-not-found]

            logger.info("Converting %s via Mistral OCR API...", pdf_path.name)
            client = Mistral(api_key=self.api_key)

            # Upload PDF to Mistral file API first, then use the file
            # reference in the OCR call. For PDFs, use document_url with
            # a file upload approach.
            import base64

            pdf_bytes = pdf_path.read_bytes()
            pdf_b64 = base64.standard_b64encode(pdf_bytes).decode("ascii")

            ocr_response = client.ocr.process(
                model=self.model,
                document={
                    "type": "document_url",
                    "document_url": f"data:application/pdf;base64,{pdf_b64}",
                },
                include_image_base64=False,
            )

            # Concatenate page markdowns
            pages = []
            for page in ocr_response.pages:
                pages.append(page.markdown)
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
