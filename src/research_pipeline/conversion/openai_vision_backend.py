"""OpenAI GPT-4o vision backend for PDF-to-Markdown conversion.

Uses the OpenAI Chat Completions API with vision to convert PDF pages
to Markdown. Requires ``OPENAI_API_KEY``.
Per-token pricing, no dedicated free tier for volume.

Keywords: multi-account, account rotation, quota rotation.
"""

from __future__ import annotations

import base64
import logging

from research_pipeline.conversion.base import ConversionContext, ConverterBackend
from research_pipeline.conversion.registry import register_backend

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

    @property
    def converter_name(self) -> str:
        return "openai_vision"

    @property
    def converter_version(self) -> str:
        return "cloud"

    def _config_string(self) -> str:
        return f"model={self.model}"

    def _run(self, ctx: ConversionContext) -> tuple[str, list[str]]:
        import fitz
        from openai import OpenAI  # type: ignore[import-not-found]

        logger.info("Converting %s via OpenAI vision API...", ctx.pdf_path.name)
        client = OpenAI(api_key=self.api_key)
        doc = fitz.open(str(ctx.pdf_path))
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

        ctx.md_path.write_text(markdown_text, encoding="utf-8")
        logger.info(
            "Converted %s → %s via OpenAI vision (%d pages)",
            ctx.pdf_path.name,
            ctx.md_path.name,
            len(pages_md),
        )

        return markdown_text, []
