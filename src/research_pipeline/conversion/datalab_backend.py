"""Datalab (hosted Marker) cloud backend for PDF-to-Markdown conversion.

Uses the Datalab REST API (https://documentation.datalab.to/) — the hosted
version of Marker. Requires ``DATALAB_API_KEY`` from
https://www.datalab.to/app/keys.
Free tier: $5 credit on signup (~1,250 pages on fast mode).
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

    def fingerprint(self) -> str:
        config_hash = sha256_str(f"mode={self.mode}")[:8]
        return f"datalab/cloud/{config_hash}"

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

        config_hash = sha256_str(f"mode={self.mode}")[:8]

        if md_path.exists():
            logger.info("Markdown already exists, skipping: %s", md_path)
            return ConvertManifestEntry(
                arxiv_id=arxiv_id,
                version=version,
                pdf_path=str(pdf_path),
                pdf_sha256=pdf_hash,
                markdown_path=str(md_path),
                converter_name="datalab",
                converter_version="cloud",
                converter_config_hash=config_hash,
                converted_at=utc_now(),
                warnings=[],
                status="skipped_exists",
            )

        try:
            from datalab_sdk import ConvertOptions, DatalabClient

            logger.info(
                "Converting %s via Datalab API (mode=%s)...",
                pdf_path.name,
                self.mode,
            )
            client = DatalabClient(api_key=self.api_key)
            options = ConvertOptions(
                output_format="markdown",
                mode=self.mode,
            )
            result = client.convert(str(pdf_path), options=options)
            markdown_text: str = result.markdown

            md_path.write_text(markdown_text, encoding="utf-8")
            logger.info("Converted %s → %s via Datalab", pdf_path.name, md_path.name)

            warnings: list[str] = []
            quality = getattr(result, "parse_quality_score", None)
            if quality is not None and quality < 3.0:
                warnings.append(f"Low parse quality score: {quality:.1f}/5.0")

            return ConvertManifestEntry(
                arxiv_id=arxiv_id,
                version=version,
                pdf_path=str(pdf_path),
                pdf_sha256=pdf_hash,
                markdown_path=str(md_path),
                converter_name="datalab",
                converter_version="cloud",
                converter_config_hash=config_hash,
                converted_at=utc_now(),
                warnings=warnings,
                status="converted",
            )
        except Exception as exc:
            logger.error("Datalab conversion failed for %s: %s", pdf_path.name, exc)
            return ConvertManifestEntry(
                arxiv_id=arxiv_id,
                version=version,
                pdf_path=str(pdf_path),
                pdf_sha256=pdf_hash,
                markdown_path="",
                converter_name="datalab",
                converter_version="cloud",
                converter_config_hash=config_hash,
                converted_at=utc_now(),
                warnings=[str(exc)],
                status="failed",
                error=str(exc),
            )
