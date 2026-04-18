"""Marker-based PDF-to-Markdown converter backend."""

from __future__ import annotations

import logging
from pathlib import Path

from research_pipeline.conversion.base import ConverterBackend
from research_pipeline.conversion.registry import register_backend
from research_pipeline.infra.clock import utc_now
from research_pipeline.infra.hashing import sha256_file, sha256_str
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

    def fingerprint(self) -> str:
        """Return converter fingerprint."""
        config_parts = [
            f"force_ocr={self.force_ocr}",
            f"use_llm={self.use_llm}",
            f"llm_service={self.llm_service or 'none'}",
        ]
        config_hash = sha256_str("|".join(config_parts))[:8]
        return f"marker/{self.version}/{config_hash}"

    def convert(
        self, pdf_path: Path, output_dir: Path, *, force: bool = False
    ) -> ConvertManifestEntry:
        """Convert a PDF to Markdown using Marker.

        Args:
            pdf_path: Path to the source PDF.
            output_dir: Directory to write Markdown output.
            force: Re-convert even if output already exists.

        Returns:
            Conversion manifest entry.
        """
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

        config_hash = sha256_str(
            f"force_ocr={self.force_ocr}|use_llm={self.use_llm}"
            f"|llm_service={self.llm_service or 'none'}"
        )[:8]

        if md_path.exists():
            logger.info("Markdown already exists, skipping: %s", md_path)
            return ConvertManifestEntry(
                arxiv_id=arxiv_id,
                version=version,
                pdf_path=str(pdf_path),
                pdf_sha256=pdf_hash,
                markdown_path=str(md_path),
                converter_name="marker",
                converter_version=self.version,
                converter_config_hash=config_hash,
                converted_at=utc_now(),
                warnings=[],
                status="skipped_exists",
            )

        try:
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
            rendered = converter(str(pdf_path))
            markdown_text = rendered.markdown

            md_path.write_text(markdown_text, encoding="utf-8")
            logger.info("Converted %s → %s", pdf_path.name, md_path.name)

            return ConvertManifestEntry(
                arxiv_id=arxiv_id,
                version=version,
                pdf_path=str(pdf_path),
                pdf_sha256=pdf_hash,
                markdown_path=str(md_path),
                converter_name="marker",
                converter_version=self.version,
                converter_config_hash=config_hash,
                converted_at=utc_now(),
                warnings=[],
                status="converted",
            )

        except ImportError:
            msg = (
                "Marker is not installed. Install with: "
                "pip install 'research-pipeline[marker]'"
            )
            logger.error(msg)
            return ConvertManifestEntry(
                arxiv_id=arxiv_id,
                version=version,
                pdf_path=str(pdf_path),
                pdf_sha256=pdf_hash,
                markdown_path=str(md_path),
                converter_name="marker",
                converter_version=self.version,
                converter_config_hash=config_hash,
                converted_at=utc_now(),
                warnings=[msg],
                status="failed",
                error=msg,
            )
        except Exception as exc:
            logger.error("Marker conversion failed for %s: %s", pdf_path.name, exc)
            return ConvertManifestEntry(
                arxiv_id=arxiv_id,
                version=version,
                pdf_path=str(pdf_path),
                pdf_sha256=pdf_hash,
                markdown_path=str(md_path),
                converter_name="marker",
                converter_version=self.version,
                converter_config_hash=config_hash,
                converted_at=utc_now(),
                warnings=[str(exc)],
                status="failed",
                error=str(exc),
            )
