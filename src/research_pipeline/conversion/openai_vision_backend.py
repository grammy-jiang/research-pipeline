"""OpenAI GPT-4o vision backend for PDF-to-Markdown conversion.

Uses the OpenAI Chat Completions API with vision to convert PDF pages
to Markdown. Requires ``OPENAI_API_KEY``.
Per-token pricing, no dedicated free tier for volume.
"""

from __future__ import annotations

import base64
import logging
from pathlib import Path

from research_pipeline.conversion.base import ConverterBackend
from research_pipeline.conversion.registry import register_backend
from research_pipeline.infra.clock import utc_now
from research_pipeline.infra.hashing import sha256_file, sha256_str
from research_pipeline.models.conversion import ConvertManifestEntry

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a document OCR assistant. Convert the provided PDF page image "
    "to clean Markdown. Preserve all text, headings, lists, tables, code "
    "blocks, and math equations (use LaTeX $...$ and $$...$$). "
    "Do not add commentary. Output only the Markdown."
)


@register_backend("openai_vision")
class OpenAIVisionBackend(ConverterBackend):
    """OpenAI GPT-4o vision PDF-to-Markdown converter.

    Converts each page as an image via the Chat Completions API.
    Very good quality, ~2-10 s per page, per-token pricing.
    """

    def __init__(
        self,
        *,
        api_key: str = "",
        model: str = "gpt-4o",
    ) -> None:
        if not api_key:
            raise ValueError(
                "OpenAI vision backend requires 'api_key'. "
                "Set RESEARCH_PIPELINE_OPENAI_API_KEY or configure in config.toml."
            )
        self.api_key = api_key
        self.model = model

    def fingerprint(self) -> str:
        config_hash = sha256_str(f"model={self.model}")[:8]
        return f"openai_vision/cloud/{config_hash}"

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
                converter_name="openai_vision",
                converter_version="cloud",
                converter_config_hash=config_hash,
                converted_at=utc_now(),
                warnings=[],
                status="skipped_exists",
            )

        try:
            import fitz  # type: ignore[import-not-found]  # PyMuPDF
            from openai import OpenAI  # type: ignore[import-not-found]

            logger.info("Converting %s via OpenAI vision API...", pdf_path.name)
            client = OpenAI(api_key=self.api_key)
            doc = fitz.open(str(pdf_path))
            pages_md: list[str] = []

            for page_num in range(len(doc)):
                page = doc[page_num]
                # Render page to PNG at 150 DPI for good quality/size balance
                pix = page.get_pixmap(dpi=150)
                img_bytes = pix.tobytes("png")
                img_b64 = base64.standard_b64encode(img_bytes).decode("ascii")

                response = client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": _SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{img_b64}",
                                    },
                                },
                            ],
                        },
                    ],
                    max_tokens=4096,
                    temperature=0,
                )
                page_md = response.choices[0].message.content or ""
                pages_md.append(page_md)
                logger.debug(
                    "Converted page %d/%d via OpenAI vision",
                    page_num + 1,
                    len(doc),
                )

            doc.close()
            markdown_text = "\n\n---\n\n".join(pages_md)

            md_path.write_text(markdown_text, encoding="utf-8")
            logger.info(
                "Converted %s → %s via OpenAI vision (%d pages)",
                pdf_path.name,
                md_path.name,
                len(pages_md),
            )

            return ConvertManifestEntry(
                arxiv_id=arxiv_id,
                version=version,
                pdf_path=str(pdf_path),
                pdf_sha256=pdf_hash,
                markdown_path=str(md_path),
                converter_name="openai_vision",
                converter_version="cloud",
                converter_config_hash=config_hash,
                converted_at=utc_now(),
                warnings=[],
                status="converted",
            )
        except Exception as exc:
            logger.error(
                "OpenAI vision conversion failed for %s: %s", pdf_path.name, exc
            )
            return ConvertManifestEntry(
                arxiv_id=arxiv_id,
                version=version,
                pdf_path=str(pdf_path),
                pdf_sha256=pdf_hash,
                markdown_path="",
                converter_name="openai_vision",
                converter_version="cloud",
                converter_config_hash=config_hash,
                converted_at=utc_now(),
                warnings=[str(exc)],
                status="failed",
                error=str(exc),
            )
