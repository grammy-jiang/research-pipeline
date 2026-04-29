"""Topic-memory lookup and reviewable alias suggestions."""

from __future__ import annotations

from research_pipeline.briefing.models import (
    BriefingCluster,
    TopicAliasSuggestion,
    TopicMemory,
)
from research_pipeline.briefing.normalize import (
    normalize_title,
    stable_hash,
    utc_now_iso,
)
from research_pipeline.briefing.topic_memory_store import TopicMemoryStore


def lookup_recent_topic_context(
    store: TopicMemoryStore, cluster: BriefingCluster
) -> list[TopicMemory]:
    """Find existing topic memories by ID, then fallback heuristics.

    If explicit topic IDs are provided and resolved, those matches are
    authoritative and fallback matching is skipped.
    """
    memories: dict[str, TopicMemory] = {}
    for topic_id in cluster.topic_ids:
        memory = store.get(topic_id)
        if memory is not None:
            memories[memory.topic_id] = memory

    # Current cluster evidence via explicit topic IDs is authoritative.
    if memories:
        return list(memories.values())

    normalized_title = normalize_title(cluster.title)
    for memory in store.list_memories():
        alias_set = {normalize_title(alias) for alias in memory.aliases}
        if normalized_title and (
            normalized_title == normalize_title(memory.name)
            or normalized_title in alias_set
        ):
            memories[memory.topic_id] = memory
            continue
        if cluster.cluster_id in memory.canonical_clusters:
            memories[memory.topic_id] = memory

    return list(memories.values())


def suggest_aliases(
    cluster: BriefingCluster, memories: list[TopicMemory]
) -> list[TopicAliasSuggestion]:
    """Suggest aliases without applying durable topic merges."""
    suggestions: list[TopicAliasSuggestion] = []
    title = normalize_title(cluster.title)
    for memory in memories:
        aliases = {normalize_title(alias) for alias in memory.aliases}
        if title and title != normalize_title(memory.name) and title not in aliases:
            suggestions.append(
                TopicAliasSuggestion(
                    suggestion_id=stable_hash(
                        memory.topic_id, title, prefix="alias_suggestion_"
                    ),
                    created_at=utc_now_iso(),
                    topic_id=memory.topic_id,
                    suggested_alias=title,
                    reason=(
                        "Cluster title matched an existing topic lookup but differs "
                        "from the canonical topic name."
                    ),
                )
            )
    return suggestions
