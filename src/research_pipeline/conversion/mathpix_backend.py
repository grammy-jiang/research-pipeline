"""Mathpix-based converter backend (optional, placeholder)."""

import logging
from pathlib import Path

from research_pipeline.conversion.base import ConverterBackend
from research_pipeline.models.conversion import ConvertManifestEntry

logger = logging.getLogger(__name__)


class MathpixBackend(ConverterBackend):
    """Mathpix PDF-to-Markdown converter (not yet implemented)."""

    def fingerprint(self) -> str:
        return "mathpix/not_implemented/0"

    def convert(self, pdf_path: Path, output_dir: Path) -> ConvertManifestEntry:
        raise NotImplementedError(
            "Mathpix backend is a future extension. Use 'docling' for V1."
        )
