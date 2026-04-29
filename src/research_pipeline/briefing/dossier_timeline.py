"""Phase E — evidence timeline construction for hot-topic dossiers.

Builds a deterministic, deduplicated timeline of evidence rows merging
cluster events with optional prior topic-memory references. Pure
function: no I/O, no network.
"""

from __future__ import annotations

from research_pipeline.briefing.dossier import EvidenceTimelineEntry
from research_pipeline.briefing.models import BriefingCluster, TopicMemory


def build_evidence_timeline(
    cluster: BriefingCluster,
    *,
    topic_memories: list[TopicMemory] | None = None,
) -> tuple[EvidenceTimelineEntry, ...]:
    """Return ordered, deduplicated evidence timeline entries.

    Sources merged in order:
    1. Cluster events (each must have a canonical_url).
    2. Optional prior topic-memory entries (added once per unique URL).

    Deduplication is by ``evidence_url``. Output is sorted by ``date``
    ascending then by ``evidence_url`` for determinism.
    """
    seen: set[str] = set()
    entries: list[EvidenceTimelineEntry] = []

    for event in cluster.events:
        url = str(event.canonical_url)
        if not url or url in seen:
            continue
        seen.add(url)
        date = event.published_at or event.retrieved_at[:10]
        entries.append(
            EvidenceTimelineEntry(
                date=date,
                evidence_url=url,
                source_class=event.source_type.value,
                note=event.title,
                origin="cluster_event",
            )
        )

    for memory in topic_memories or []:
        if memory.last_reported_at and memory.obsidian_note:
            url = memory.obsidian_note
            if not (url.startswith("http://") or url.startswith("https://")):
                # Fall back to topic-id obsidian URI form to satisfy URL gate.
                url = f"obsidian://open?vault=Research&file={memory.topic_id}"
            if url in seen:
                continue
            seen.add(url)
            entries.append(
                EvidenceTimelineEntry(
                    date=memory.last_reported_at[:10],
                    evidence_url=url,
                    source_class="primary_artifact",
                    note=f"Prior topic-memory entry for {memory.name}",
                    origin="topic_memory",
                )
            )

    entries.sort(key=lambda entry: (entry.date, entry.evidence_url))
    return tuple(entries)
