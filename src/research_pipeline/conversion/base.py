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


def parse_arxiv_stem(stem: str) -> tuple[str, str]:
    """Split a PDF file stem into its arXiv id and version (#124).

    A trailing ``v<N>`` (e.g. ``2401.00001v2``) is peeled off as the version;
    a stem without one defaults to ``v1``. This six-line block was copy-pasted
    byte-for-byte into all nine converter backends' ``convert()`` methods.

    Args:
        stem: The PDF path stem (``pdf_path.stem``).

    Returns:
        ``(arxiv_id, version)``. The version keeps its ``v`` prefix.
    """
    if len(stem) >= 2 and stem[-2] == "v" and stem[-1].isdigit():
        return stem[:-2], stem[-2:]
    return stem, "v1"


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
