"""PyMuPDF4LLM-based PDF-to-Markdown converter backend.

A fast, lightweight converter (~10-50x faster than Docling) that trades
equation rendering quality for speed. Best for pre-screening or when
LaTeX equations are not critical.
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


@register_backend("pymupdf4llm")
class PyMuPDF4LLMBackend(ConverterBackend):
    """PDF-to-Markdown conversion using pymupdf4llm (AGPL license).

    Very fast CPU-only converter. Good for quick screening.
    Does not render LaTeX equations.
    """

    def __init__(self, *, page_chunks: bool = False) -> None:
        self.page_chunks = page_chunks
        self._version: str | None = None

    @property
    def version(self) -> str:
        """Get the installed pymupdf4llm version."""
        if self._version is None:
            try:
                import pymupdf4llm

                self._version = getattr(pymupdf4llm, "__version__", "unknown")
            except ImportError:
                self._version = "not_installed"
        return self._version

    def fingerprint(self) -> str:
        """Return converter fingerprint."""
        config_hash = sha256_str(f"page_chunks={self.page_chunks}")[:8]
        return f"pymupdf4llm/{self.version}/{config_hash}"

    def convert(
        self, pdf_path: Path, output_dir: Path, *, force: bool = False
    ) -> ConvertManifestEntry:
        """Convert a PDF to Markdown using pymupdf4llm.

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

        config_hash = sha256_str(f"page_chunks={self.page_chunks}")[:8]

        if md_path.exists():
            logger.info("Markdown already exists, skipping: %s", md_path)
            return ConvertManifestEntry(
                arxiv_id=arxiv_id,
                version=version,
                pdf_path=str(pdf_path),
                pdf_sha256=pdf_hash,
                markdown_path=str(md_path),
                converter_name="pymupdf4llm",
                converter_version=self.version,
                converter_config_hash=config_hash,
                converted_at=utc_now(),
                warnings=[],
                status="skipped_exists",
            )

        try:
            import pymupdf4llm

            markdown_text = pymupdf4llm.to_markdown(
                str(pdf_path), page_chunks=self.page_chunks
            )

            md_path.write_text(markdown_text, encoding="utf-8")
            logger.info("Converted %s → %s", pdf_path.name, md_path.name)

            return ConvertManifestEntry(
                arxiv_id=arxiv_id,
                version=version,
                pdf_path=str(pdf_path),
                pdf_sha256=pdf_hash,
                markdown_path=str(md_path),
                converter_name="pymupdf4llm",
                converter_version=self.version,
                converter_config_hash=config_hash,
                converted_at=utc_now(),
                warnings=[],
                status="converted",
            )

        except ImportError:
            msg = (
                "pymupdf4llm is not installed. Install with: "
                "pip install 'research-pipeline[pymupdf4llm]'"
            )
            logger.error(msg)
            return ConvertManifestEntry(
                arxiv_id=arxiv_id,
                version=version,
                pdf_path=str(pdf_path),
                pdf_sha256=pdf_hash,
                markdown_path=str(md_path),
                converter_name="pymupdf4llm",
                converter_version=self.version,
                converter_config_hash=config_hash,
                converted_at=utc_now(),
                warnings=[msg],
                status="failed",
                error=msg,
            )
        except Exception as exc:
            logger.error("pymupdf4llm conversion failed for %s: %s", pdf_path.name, exc)
            return ConvertManifestEntry(
                arxiv_id=arxiv_id,
                version=version,
                pdf_path=str(pdf_path),
                pdf_sha256=pdf_hash,
                markdown_path=str(md_path),
                converter_name="pymupdf4llm",
                converter_version=self.version,
                converter_config_hash=config_hash,
                converted_at=utc_now(),
                warnings=[str(exc)],
                status="failed",
                error=str(exc),
            )
