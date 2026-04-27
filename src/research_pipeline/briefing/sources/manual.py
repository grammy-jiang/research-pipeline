"""Manual curated source adapter."""

from __future__ import annotations

from research_pipeline.briefing.models import (
    AccessMethod,
    BriefingSourceConfig,
    IntelligenceEvent,
)
from research_pipeline.briefing.normalize import (
    canonicalize_url,
    content_hash_for,
    dedup_key_for,
    event_id_for,
    topic_id_for_title,
    utc_now_iso,
)


class ManualSource:
    """Emit registry-provided curated events."""

    def __init__(self, source: BriefingSourceConfig) -> None:
        self.source = source

    def poll(self) -> list[IntelligenceEvent]:
        """Normalize manual items into events."""
        events: list[IntelligenceEvent] = []
        for item in self.source.manual_items[: self.source.max_events_per_run]:
            canonical_url = canonicalize_url(str(item.url))
            source_native_id = item.source_native_id or canonical_url
            dedup_key = dedup_key_for(
                source=self.source,
                title=item.title,
                canonical_url=canonical_url,
                source_native_id=source_native_id,
            )
            events.append(
                IntelligenceEvent(
                    event_id=event_id_for(self.source, canonical_url, source_native_id),
                    source_name=self.source.source_name,
                    source_id=self.source.source_id,
                    source_type=self.source.source_class,
                    item_type=item.item_type,
                    canonical_url=canonical_url,
                    title=item.title,
                    retrieved_at=utc_now_iso(),
                    collection_method=AccessMethod.MANUAL,
                    content_hash=content_hash_for(
                        item.title,
                        canonical_url,
                        item.published_at,
                        item.summary_hint,
                    ),
                    dedup_key=dedup_key,
                    published_at=item.published_at,
                    source_native_id=source_native_id,
                    summary_hint=item.summary_hint,
                    excerpt=item.summary_hint,
                    topics=(topic_id_for_title(item.title),),
                    artifact_links=(canonical_url,),
                    confidence="medium",
                )
            )
        return events
