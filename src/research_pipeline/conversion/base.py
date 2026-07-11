"""Abstract converter backend interface."""

import logging
from abc import ABC, abstractmethod
from pathlib import Path

from research_pipeline.models.conversion import ConvertManifestEntry

logger = logging.getLogger(__name__)

# Programming-error exception types that a backend's ``convert()`` boundary must
# NOT swallow into a ``status="failed"`` manifest entry (#124): these signal a
# real bug (a wrong key, a None attribute, a type mismatch), not an operational
# conversion failure, so they are re-raised to surface rather than hide them.
# Operational failures (OSError, RequestException, subprocess errors, timeouts,
# library errors) still become ``status="failed"`` as before.
PROGRAMMING_ERRORS: tuple[type[BaseException], ...] = (
    KeyError,
    AttributeError,
    TypeError,
    NameError,
)


class ConverterBackend(ABC):
    """Abstract interface for PDF-to-Markdown converter backends."""

    @abstractmethod
    def convert(
        self, pdf_path: Path, output_dir: Path, *, force: bool = False
    ) -> ConvertManifestEntry:
        """Convert a single PDF to Markdown.

        Args:
            pdf_path: Path to the source PDF.
            output_dir: Directory to write Markdown output.
            force: Re-convert even if output already exists.

        Returns:
            Manifest entry recording the conversion result.
        """

    @abstractmethod
    def fingerprint(self) -> str:
        """Return a deterministic fingerprint for this converter configuration.

        Returns:
            String of the form ``name/version/config_hash``.
        """
