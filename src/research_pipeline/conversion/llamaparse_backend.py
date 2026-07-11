"""LlamaParse cloud backend for PDF-to-Markdown conversion.

Uses the LlamaCloud API (https://developers.llamaindex.ai) for
document parsing. Requires ``LLAMA_CLOUD_API_KEY`` from
https://cloud.llamaindex.ai.
Free tier: 1,000 pages/day.

Keywords: multi-account, account rotation, quota rotation.
"""

from __future__ import annotations

import logging

from research_pipeline.conversion.base import ConversionContext, ConverterBackend
from research_pipeline.conversion.registry import register_backend

logger = logging.getLogger(__name__)


@register_backend("llamaparse")
class LlamaParseBackend(ConverterBackend):
    """LlamaParse cloud PDF-to-Markdown converter.

    Good quality with LlamaIndex integration.
    ~5-15 s per document, 1K free pages/day.
    """

    def __init__(
        self,
        *,
        api_key: str = "",
        tier: str = "agentic",
    ) -> None:
        if not api_key:
            raise ValueError(
                "LlamaParse backend requires 'api_key'. "
                "Set RESEARCH_PIPELINE_LLAMAPARSE_API_KEY or configure in config.toml."
            )
        self.api_key = api_key
        self.tier = tier

    @property
    def converter_name(self) -> str:
        return "llamaparse"

    @property
    def converter_version(self) -> str:
        return "cloud"

    def _config_string(self) -> str:
        return f"llamaparse:{self.tier}"

    def _run(self, ctx: ConversionContext) -> tuple[str, list[str]]:
        import asyncio

        from llama_cloud import AsyncLlamaCloud  # type: ignore[import-not-found]

        logger.info(
            "Converting %s via LlamaParse API (tier=%s)...",
            ctx.pdf_path.name,
            self.tier,
        )

        async def _parse() -> str:
            client = AsyncLlamaCloud(api_key=self.api_key)
            file_obj = await client.files.create(
                file=str(ctx.pdf_path),
                purpose="parse",
            )
            result = await client.parsing.parse(
                file_id=file_obj.id,
                tier=self.tier,
                version="latest",
                expand=["markdown"],
            )
            pages = []
            for page in result.markdown.pages:
                pages.append(page.markdown)
            return "\n\n---\n\n".join(pages)

        # Run async in sync context
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as pool:
                markdown_text = pool.submit(asyncio.run, _parse()).result()
        else:
            markdown_text = asyncio.run(_parse())

        ctx.md_path.write_text(markdown_text, encoding="utf-8")
        logger.info(
            "Converted %s → %s via LlamaParse",
            ctx.pdf_path.name,
            ctx.md_path.name,
        )

        return markdown_text, []
