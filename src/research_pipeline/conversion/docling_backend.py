"""Docling-based PDF-to-Markdown converter backend."""

import logging
from pathlib import Path

from research_pipeline.conversion.base import ConverterBackend
from research_pipeline.conversion.registry import register_backend
from research_pipeline.infra.clock import utc_now
from research_pipeline.infra.hashing import sha256_file, sha256_str
from research_pipeline.models.conversion import ConvertManifestEntry

logger = logging.getLogger(__name__)


@register_backend("docling")
class DoclingBackend(ConverterBackend):
    """PDF-to-Markdown conversion using Docling (MIT license)."""

    def __init__(self, timeout_seconds: int = 300) -> None:
        self.timeout_seconds = timeout_seconds
        self._version: str | None = None

    @property
    def version(self) -> str:
        """Get the installed Docling version."""
        if self._version is None:
            try:
                from importlib.metadata import version

                self._version = version("docling")
            except Exception:
                self._version = "unknown"
        return self._version

    def fingerprint(self) -> str:
        """Return converter fingerprint."""
        config_hash = sha256_str(f"timeout={self.timeout_seconds}")[:8]
        return f"docling/{self.version}/{config_hash}"

    def convert(
        self, pdf_path: Path, output_dir: Path, *, force: bool = False
    ) -> ConvertManifestEntry:
        """Convert a PDF to Markdown using Docling.

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

        # Parse arXiv ID and version from filename
        stem = pdf_path.stem
        arxiv_id = stem
        version = "v1"
        if stem[-2] == "v" and stem[-1].isdigit():
            arxiv_id = stem[:-2]
            version = stem[-2:]

        # Remove existing output when force is set
        if force and md_path.exists():
            logger.info("Force mode: removing existing %s", md_path)
            md_path.unlink()

        # Check if already converted with same fingerprint
        if md_path.exists():
            logger.info("Markdown already exists, skipping: %s", md_path)
            return ConvertManifestEntry(
                arxiv_id=arxiv_id,
                version=version,
                pdf_path=str(pdf_path),
                pdf_sha256=pdf_hash,
                markdown_path=str(md_path),
                converter_name="docling",
                converter_version=self.version,
                converter_config_hash=sha256_str(f"timeout={self.timeout_seconds}")[:8],
                converted_at=utc_now(),
                warnings=[],
                status="skipped_exists",
            )

        try:
            from docling.document_converter import (  # type: ignore[import-not-found]
                DocumentConverter,
            )

            converter = DocumentConverter()
            result = converter.convert(str(pdf_path))
            markdown_text = result.document.export_to_markdown()

            md_path.write_text(markdown_text, encoding="utf-8")
            logger.info("Converted %s → %s", pdf_path.name, md_path.name)

            return ConvertManifestEntry(
                arxiv_id=arxiv_id,
                version=version,
                pdf_path=str(pdf_path),
                pdf_sha256=pdf_hash,
                markdown_path=str(md_path),
                converter_name="docling",
                converter_version=self.version,
                converter_config_hash=sha256_str(f"timeout={self.timeout_seconds}")[:8],
                converted_at=utc_now(),
                warnings=[],
                status="converted",
            )

        except ImportError:
            msg = (
                "Docling is not installed. Install with: "
                "pip install 'research-pipeline[docling]'"
            )
            logger.error(msg)
            return ConvertManifestEntry(
                arxiv_id=arxiv_id,
                version=version,
                pdf_path=str(pdf_path),
                pdf_sha256=pdf_hash,
                markdown_path=str(md_path),
                converter_name="docling",
                converter_version=self.version,
                converter_config_hash=sha256_str(f"timeout={self.timeout_seconds}")[:8],
                converted_at=utc_now(),
                warnings=[],
                status="failed",
                error=msg,
            )

        except Exception as exc:
            logger.error("Conversion failed for %s: %s", pdf_path.name, exc)
            return ConvertManifestEntry(
                arxiv_id=arxiv_id,
                version=version,
                pdf_path=str(pdf_path),
                pdf_sha256=pdf_hash,
                markdown_path=str(md_path),
                converter_name="docling",
                converter_version=self.version,
                converter_config_hash=sha256_str(f"timeout={self.timeout_seconds}")[:8],
                converted_at=utc_now(),
                warnings=[],
                status="failed",
                error=str(exc),
            )
