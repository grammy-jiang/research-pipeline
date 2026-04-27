"""Hugging Face daily papers adapter for Phase F source expansion."""

from __future__ import annotations

from pathlib import Path

from research_pipeline.briefing.models import (
    AccessMethod,
    BriefingSourceConfig,
    IntelligenceEvent,
)
from research_pipeline.briefing.sources.rss_atom import RssAtomSource


class HuggingFacePapersSource(RssAtomSource):
    """Normalize Hugging Face paper feed entries as academic-source events."""

    def __init__(
        self,
        source: BriefingSourceConfig,
        *,
        fixture_base_dir: Path | None = None,
    ) -> None:
        super().__init__(source, fixture_base_dir=fixture_base_dir)

    def poll(self) -> list[IntelligenceEvent]:
        """Poll feed entries and relabel them as Hugging Face paper items."""
        return [
            event.model_copy(
                update={
                    "collection_method": AccessMethod.HUGGINGFACE_PAPERS,
                    "item_type": "huggingface_paper",
                }
            )
            for event in super().poll()
        ]
