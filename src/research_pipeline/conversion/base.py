"""Abstract converter backend interface."""

from __future__ import annotations

import logging
from abc import ABC
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from research_pipeline.infra.clock import utc_now
from research_pipeline.infra.hashing import sha256_file, sha256_str
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


@dataclass(frozen=True)
class ConversionContext:
    """Per-conversion state shared between the template ``convert()`` and a
    backend's ``_run``/``_on_convert_error`` hooks.

    It is also the single source of truth for the ``ConvertManifestEntry`` field
    set: every entry (converted, skipped, failed) is built through ``entry()`` so
    the common columns cannot drift between backends.
    """

    pdf_path: Path
    md_path: Path
    pdf_hash: str
    arxiv_id: str
    version: str
    config_hash: str
    converter_name: str
    converter_version: str

    def entry(
        self,
        status: Literal["converted", "skipped_exists", "failed"],
        *,
        markdown_path: str,
        warnings: list[str],
        error: str | None = None,
    ) -> ConvertManifestEntry:
        """Build a manifest entry for this conversion with the given outcome."""
        return ConvertManifestEntry(
            arxiv_id=self.arxiv_id,
            version=self.version,
            pdf_path=str(self.pdf_path),
            pdf_sha256=self.pdf_hash,
            markdown_path=markdown_path,
            converter_name=self.converter_name,
            converter_version=self.converter_version,
            converter_config_hash=self.config_hash,
            converted_at=utc_now(),
            warnings=warnings,
            status=status,
            error=error,
        )


class ConverterBackend(ABC):  # noqa: B024
    """Abstract interface for PDF-to-Markdown converter backends.

    Subclasses implement four hooks — ``converter_name``, ``converter_version``,
    ``_config_string`` and ``_run`` — and inherit the ``convert()`` template
    (mkdir → hash → skip-exists → run → failure policy) and ``fingerprint()``.
    A backend with a non-default failure policy overrides ``_on_convert_error``.

    The four hooks are concrete-and-``NotImplementedError`` rather than
    ``@abstractmethod`` on purpose: composite/wrapper subclasses such as
    ``FallbackConverter`` (and the test stubs) fully override ``convert()`` and
    ``fingerprint()`` and never touch the hooks, so requiring them abstractly
    would make those legitimately-instantiable classes abstract.
    """

    @property
    def converter_name(self) -> str:
        """Registry name recorded in the manifest (e.g. ``"docling"``)."""
        raise NotImplementedError

    @property
    def converter_version(self) -> str:
        """Version label recorded in the manifest (e.g. ``"cloud"``)."""
        raise NotImplementedError

    def _config_string(self) -> str:
        """Return the canonical config string hashed into the fingerprint.

        This same string feeds both ``converter_config_hash`` and
        ``fingerprint()``, so the two can never disagree.
        """
        raise NotImplementedError

    def _run(self, ctx: ConversionContext) -> tuple[str, list[str]]:
        """Perform the conversion.

        Write the Markdown to ``ctx.md_path`` and return
        ``(markdown_text, warnings)``. Raise on failure — the ``convert()``
        template routes the exception to ``_on_convert_error`` (unless it is a
        programming error, which is re-raised).
        """
        raise NotImplementedError

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
        output_dir.mkdir(parents=True, exist_ok=True)
        md_path = output_dir / (pdf_path.stem + ".md")
        pdf_hash = sha256_file(pdf_path)
        arxiv_id, version = parse_arxiv_stem(pdf_path.stem)

        if force and md_path.exists():
            logger.info("Force mode: removing existing %s", md_path)
            md_path.unlink()

        config_hash = sha256_str(self._config_string())[:8]
        ctx = ConversionContext(
            pdf_path=pdf_path,
            md_path=md_path,
            pdf_hash=pdf_hash,
            arxiv_id=arxiv_id,
            version=version,
            config_hash=config_hash,
            converter_name=self.converter_name,
            converter_version=self.converter_version,
        )

        if md_path.exists():
            logger.info("Markdown already exists, skipping: %s", md_path)
            return ctx.entry("skipped_exists", markdown_path=str(md_path), warnings=[])

        try:
            markdown_text, warnings = self._run(ctx)
            return ctx.entry("converted", markdown_path=str(md_path), warnings=warnings)
        except Exception as exc:
            if isinstance(exc, PROGRAMMING_ERRORS):
                raise
            return self._on_convert_error(exc, ctx)

    def _on_convert_error(
        self, exc: Exception, ctx: ConversionContext
    ) -> ConvertManifestEntry:
        """Build the ``status="failed"`` entry for an operational error.

        Default = cloud-backend policy: log the failure and record an
        empty-path entry whose ``warnings`` and ``error`` are ``str(exc)``.
        Local backends override this to add an ``ImportError`` install hint.
        """
        logger.error(
            "%s conversion failed for %s: %s",
            self.converter_name,
            ctx.pdf_path.name,
            exc,
        )
        return ctx.entry(
            "failed", markdown_path="", warnings=[str(exc)], error=str(exc)
        )

    def fingerprint(self) -> str:
        """Return a deterministic fingerprint for this converter configuration.

        Returns:
            String of the form ``name/version/config_hash``.
        """
        return (
            f"{self.converter_name}/{self.converter_version}/"
            f"{sha256_str(self._config_string())[:8]}"
        )
