"""Briefing source adapter protocol and helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import requests

from research_pipeline.briefing.models import BriefingSourceConfig, IntelligenceEvent

USER_AGENT = "research-pipeline-daily-ai-intelligence/1.0 (+https://github.com/grammy-jiang/research-pipeline)"

# Default HTTP timeout (seconds) for briefing source fetches (#124).
DEFAULT_HTTP_TIMEOUT = 20


class BriefingSource(Protocol):
    """Protocol implemented by all briefing source adapters."""

    def poll(self) -> list[IntelligenceEvent]:
        """Poll a source and return normalized events."""


def build_session() -> requests.Session:
    """Build a polite requests session."""
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    return session


def read_fixture_text(
    source: BriefingSourceConfig, base_dir: Path | None = None
) -> str | None:
    """Read fixture text for offline source tests or deterministic runs."""
    if source.fixture_path is None:
        return None
    path = Path(source.fixture_path)
    if not path.is_absolute() and base_dir is not None:
        path = base_dir / path
    return path.read_text(encoding="utf-8")
