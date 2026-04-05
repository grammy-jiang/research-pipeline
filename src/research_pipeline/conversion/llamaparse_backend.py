"""LlamaParse cloud backend for PDF-to-Markdown conversion.

Uses the LlamaCloud API (https://developers.llamaindex.ai) for
document parsing. Requires ``LLAMA_CLOUD_API_KEY`` from
https://cloud.llamaindex.ai.
Free tier: 1,000 pages/day.
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

    def fingerprint(self) -> str:
        config_hash = sha256_str(f"llamaparse:{self.tier}")[:8]
        return f"llamaparse/cloud/{config_hash}"

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

        config_hash = sha256_str(f"llamaparse:{self.tier}")[:8]

        if md_path.exists():
            logger.info("Markdown already exists, skipping: %s", md_path)
            return ConvertManifestEntry(
                arxiv_id=arxiv_id,
                version=version,
                pdf_path=str(pdf_path),
                pdf_sha256=pdf_hash,
                markdown_path=str(md_path),
                converter_name="llamaparse",
                converter_version="cloud",
                converter_config_hash=config_hash,
                converted_at=utc_now(),
                warnings=[],
                status="skipped_exists",
            )

        try:
            import asyncio

            from llama_cloud import AsyncLlamaCloud

            logger.info(
                "Converting %s via LlamaParse API (tier=%s)...",
                pdf_path.name,
                self.tier,
            )

            async def _parse() -> str:
                client = AsyncLlamaCloud(api_key=self.api_key)
                file_obj = await client.files.create(
                    file=str(pdf_path),
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

            md_path.write_text(markdown_text, encoding="utf-8")
            logger.info(
                "Converted %s → %s via LlamaParse",
                pdf_path.name,
                md_path.name,
            )

            return ConvertManifestEntry(
                arxiv_id=arxiv_id,
                version=version,
                pdf_path=str(pdf_path),
                pdf_sha256=pdf_hash,
                markdown_path=str(md_path),
                converter_name="llamaparse",
                converter_version="cloud",
                converter_config_hash=config_hash,
                converted_at=utc_now(),
                warnings=[],
                status="converted",
            )
        except Exception as exc:
            logger.error("LlamaParse conversion failed for %s: %s", pdf_path.name, exc)
            return ConvertManifestEntry(
                arxiv_id=arxiv_id,
                version=version,
                pdf_path=str(pdf_path),
                pdf_sha256=pdf_hash,
                markdown_path="",
                converter_name="llamaparse",
                converter_version="cloud",
                converter_config_hash=config_hash,
                converted_at=utc_now(),
                warnings=[str(exc)],
                status="failed",
                error=str(exc),
            )
