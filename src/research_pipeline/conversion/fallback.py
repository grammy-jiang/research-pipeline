"""Fallback converter with multi-account rotation and cross-service failover.

Wraps multiple converter backends and tries them in order. When a backend
fails (quota exceeded, rate limited, or any error), automatically moves to
the next backend. Backends are ordered: all accounts of service A first,
then all accounts of service B, etc.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from research_pipeline.conversion.base import ConverterBackend
from research_pipeline.models.conversion import ConvertManifestEntry

logger = logging.getLogger(__name__)

# Patterns that indicate quota/rate-limit exhaustion (case-insensitive).
_QUOTA_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"429",
        r"rate.?limit",
        r"quota.?exceed",
        r"too many requests",
        r"usage.?limit",
        r"exceeded.*limit",
        r"limit.*exceeded",
        r"insufficient.?credits",
        r"billing",
        r"plan.*limit",
    ]
]


def _is_quota_error(error_text: str) -> bool:
    """Check whether an error message indicates a quota/rate-limit issue."""
    return any(pat.search(error_text) for pat in _QUOTA_PATTERNS)


class FallbackConverter(ConverterBackend):
    """Converter that tries multiple backends in order with automatic failover.

    On each conversion:
    1. Try the first backend in the list.
    2. If it fails with a quota/rate-limit error, log and try the next.
    3. If it fails with a non-quota error, also try the next backend.
    4. Return the first successful result, or the last failure if all fail.

    Args:
        backends: Ordered list of converter backends to try.
    """

    def __init__(self, backends: list[ConverterBackend]) -> None:
        if not backends:
            raise ValueError("FallbackConverter requires at least one backend")
        self.backends = backends

    def fingerprint(self) -> str:
        parts = [b.fingerprint() for b in self.backends]
        return f"fallback({','.join(parts)})"

    def convert(
        self, pdf_path: Path, output_dir: Path, *, force: bool = False
    ) -> ConvertManifestEntry:
        last_result: ConvertManifestEntry | None = None

        for i, backend in enumerate(self.backends):
            fp = backend.fingerprint()
            logger.info(
                "Trying backend %d/%d (%s) for %s",
                i + 1,
                len(self.backends),
                fp,
                pdf_path.name,
            )

            result = backend.convert(pdf_path, output_dir, force=force)

            if result.status != "failed":
                if i > 0:
                    logger.info(
                        "Backend %s succeeded after %d failed attempt(s)",
                        fp,
                        i,
                    )
                return result

            # Conversion failed — decide whether to try next
            error_msg = result.error or ""
            if _is_quota_error(error_msg):
                logger.warning(
                    "Quota/rate-limit error on %s: %s — trying next backend",
                    fp,
                    error_msg,
                )
            else:
                logger.warning(
                    "Backend %s failed for %s: %s — trying next backend",
                    fp,
                    pdf_path.name,
                    error_msg,
                )
            last_result = result

        # All backends exhausted
        logger.error("All %d backends failed for %s", len(self.backends), pdf_path.name)
        assert last_result is not None  # nosec B101
        return last_result
