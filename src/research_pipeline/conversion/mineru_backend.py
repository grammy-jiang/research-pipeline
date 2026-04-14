"""MinerU (magic-pdf) PDF-to-Markdown converter backend.

A high-quality scientific PDF parser achieving TEDS 93.42% for table
recognition. Supports auto/ocr/txt parse modes. The Python package
is ``magic-pdf`` and provides the ``magic_pdf`` module.
"""

from __future__ import annotations

import logging
import subprocess  # nosec B404
from pathlib import Path

from research_pipeline.conversion.base import ConverterBackend
from research_pipeline.conversion.registry import register_backend
from research_pipeline.infra.clock import utc_now
from research_pipeline.infra.hashing import sha256_file, sha256_str
from research_pipeline.models.conversion import ConvertManifestEntry

logger = logging.getLogger(__name__)


@register_backend("mineru")
class MinerUBackend(ConverterBackend):
    """PDF-to-Markdown conversion using MinerU / magic-pdf.

    MinerU is an open-source scientific PDF parser with excellent table
    recognition (TEDS 93.42%). Supports three parse modes:

    * ``"auto"`` — automatic layout detection (recommended)
    * ``"ocr"`` — force OCR for scanned documents
    * ``"txt"`` — text extraction only (fastest)
    """

    def __init__(
        self,
        *,
        parse_method: str = "auto",
        timeout_seconds: int = 600,
    ) -> None:
        self.parse_method = parse_method
        self.timeout_seconds = timeout_seconds
        self._version: str | None = None

    @property
    def version(self) -> str:
        """Get the installed magic-pdf version."""
        if self._version is None:
            try:
                from importlib.metadata import version

                self._version = version("magic-pdf")
            except Exception:
                self._version = "unknown"
        return self._version

    def fingerprint(self) -> str:
        """Return converter fingerprint."""
        config_hash = sha256_str(
            f"parse_method={self.parse_method}" f"|timeout={self.timeout_seconds}"
        )[:8]
        return f"mineru/{self.version}/{config_hash}"

    def convert(
        self, pdf_path: Path, output_dir: Path, *, force: bool = False
    ) -> ConvertManifestEntry:
        """Convert a PDF to Markdown using MinerU (magic-pdf).

        Attempts the Python API first, falling back to the ``magic-pdf``
        CLI if the API call fails.

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

        # Parse arxiv_id and version from filename
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
            f"parse_method={self.parse_method}" f"|timeout={self.timeout_seconds}"
        )[:8]

        if md_path.exists():
            logger.info("Markdown already exists, skipping: %s", md_path)
            return ConvertManifestEntry(
                arxiv_id=arxiv_id,
                version=version,
                pdf_path=str(pdf_path),
                pdf_sha256=pdf_hash,
                markdown_path=str(md_path),
                converter_name="mineru",
                converter_version=self.version,
                converter_config_hash=config_hash,
                converted_at=utc_now(),
                warnings=[],
                status="skipped_exists",
            )

        try:
            markdown_text = self._convert_python_api(pdf_path, output_dir)
            md_path.write_text(markdown_text, encoding="utf-8")
            logger.info("Converted %s → %s (Python API)", pdf_path.name, md_path.name)

            return ConvertManifestEntry(
                arxiv_id=arxiv_id,
                version=version,
                pdf_path=str(pdf_path),
                pdf_sha256=pdf_hash,
                markdown_path=str(md_path),
                converter_name="mineru",
                converter_version=self.version,
                converter_config_hash=config_hash,
                converted_at=utc_now(),
                warnings=[],
                status="converted",
            )

        except ImportError:
            msg = (
                "magic-pdf is not installed. Install with: "
                "pip install 'research-pipeline[mineru]'"
            )
            logger.error(msg)
            return ConvertManifestEntry(
                arxiv_id=arxiv_id,
                version=version,
                pdf_path=str(pdf_path),
                pdf_sha256=pdf_hash,
                markdown_path=str(md_path),
                converter_name="mineru",
                converter_version=self.version,
                converter_config_hash=config_hash,
                converted_at=utc_now(),
                warnings=[msg],
                status="failed",
                error=msg,
            )
        except Exception as exc:
            logger.warning(
                "MinerU Python API failed for %s: %s — trying CLI fallback",
                pdf_path.name,
                exc,
            )
            return self._convert_cli_fallback(
                pdf_path,
                md_path,
                pdf_hash,
                arxiv_id,
                version,
                config_hash,
            )

    def _convert_python_api(self, pdf_path: Path, output_dir: Path) -> str:
        """Attempt conversion via the magic_pdf Python API.

        Raises:
            ImportError: If magic-pdf is not installed.
            Exception: If the conversion fails.
        """
        from magic_pdf.data.data_reader_writer import (
            FileBasedDataWriter,  # type: ignore[import-untyped]
        )
        from magic_pdf.pipe.UNIPipe import UNIPipe  # type: ignore[import-untyped]

        pdf_bytes = pdf_path.read_bytes()
        image_dir = output_dir / (pdf_path.stem + "_images")
        image_dir.mkdir(parents=True, exist_ok=True)
        image_writer = FileBasedDataWriter(str(image_dir))

        pipe = UNIPipe(pdf_bytes, [], image_writer)
        pipe.pipe_classify()
        pipe.pipe_analyze()
        pipe.pipe_parse()
        markdown_text: str = pipe.pipe_mk_markdown(str(image_dir), drop_mode="none")
        return markdown_text

    def _convert_cli_fallback(
        self,
        pdf_path: Path,
        md_path: Path,
        pdf_hash: str,
        arxiv_id: str,
        version: str,
        config_hash: str,
    ) -> ConvertManifestEntry:
        """Fall back to the ``magic-pdf`` CLI tool."""
        try:
            cli_output_dir = md_path.parent / (pdf_path.stem + "_cli_out")
            cli_output_dir.mkdir(parents=True, exist_ok=True)

            result = subprocess.run(  # nosec B603 B607
                [
                    "magic-pdf",
                    "-p",
                    str(pdf_path),
                    "-o",
                    str(cli_output_dir),
                    "-m",
                    self.parse_method,
                ],
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"magic-pdf CLI exited with code {result.returncode}: "
                    f"{result.stderr.strip()}"
                )

            # magic-pdf CLI writes to <output>/<stem>/<parse_method>/content.md
            # or <output>/<stem>/<parse_method>/<stem>.md — search for .md files
            md_files = list(cli_output_dir.rglob("*.md"))
            if not md_files:
                raise RuntimeError("magic-pdf CLI produced no Markdown output")

            # Pick the largest .md file as the main output
            best = max(md_files, key=lambda p: p.stat().st_size)
            markdown_text = best.read_text(encoding="utf-8")
            md_path.write_text(markdown_text, encoding="utf-8")
            logger.info("Converted %s → %s (CLI fallback)", pdf_path.name, md_path.name)

            return ConvertManifestEntry(
                arxiv_id=arxiv_id,
                version=version,
                pdf_path=str(pdf_path),
                pdf_sha256=pdf_hash,
                markdown_path=str(md_path),
                converter_name="mineru",
                converter_version=self.version,
                converter_config_hash=config_hash,
                converted_at=utc_now(),
                warnings=["Converted via CLI fallback"],
                status="converted",
            )
        except Exception as exc:
            error_msg = f"MinerU conversion failed: {exc}"
            logger.error("MinerU conversion failed for %s: %s", pdf_path.name, exc)
            return ConvertManifestEntry(
                arxiv_id=arxiv_id,
                version=version,
                pdf_path=str(pdf_path),
                pdf_sha256=pdf_hash,
                markdown_path=str(md_path),
                converter_name="mineru",
                converter_version=self.version,
                converter_config_hash=config_hash,
                converted_at=utc_now(),
                warnings=[error_msg],
                status="failed",
                error=error_msg,
            )
