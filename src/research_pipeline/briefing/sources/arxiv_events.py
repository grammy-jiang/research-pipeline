"""arXiv Atom adapter mapped into briefing events for Phase F."""

from __future__ import annotations

from pathlib import Path

from research_pipeline.briefing.models import (
    AccessMethod,
    BriefingSourceConfig,
    IntelligenceEvent,
)
from research_pipeline.briefing.sources.rss_atom import RssAtomSource


class ArxivEventsSource(RssAtomSource):
    """Map arXiv Atom entries into generic briefing events."""

    def __init__(
        self,
        source: BriefingSourceConfig,
        *,
        fixture_base_dir: Path | None = None,
    ) -> None:
        super().__init__(source, fixture_base_dir=fixture_base_dir)

    def poll(self) -> list[IntelligenceEvent]:
        """Poll arXiv Atom entries and relabel them as paper events."""
        return [
            event.model_copy(
                update={
                    "collection_method": AccessMethod.ARXIV,
                    "item_type": "arxiv_paper",
                }
            )
            for event in super().poll()
        ]
