"""Abstract converter backend interface."""

import logging
from abc import ABC, abstractmethod
from pathlib import Path

from research_pipeline.models.conversion import ConvertManifestEntry

logger = logging.getLogger(__name__)


class ConverterBackend(ABC):
    """Abstract interface for PDF-to-Markdown converter backends."""

    @abstractmethod
    def convert(self, pdf_path: Path, output_dir: Path) -> ConvertManifestEntry:
        """Convert a single PDF to Markdown.

        Args:
            pdf_path: Path to the source PDF.
            output_dir: Directory to write Markdown output.

        Returns:
            Manifest entry recording the conversion result.
        """

    @abstractmethod
    def fingerprint(self) -> str:
        """Return a deterministic fingerprint for this converter configuration.

        Returns:
            String of the form ``name/version/config_hash``.
        """
