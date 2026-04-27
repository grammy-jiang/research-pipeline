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
from research_pipeline.briefing.topic_memory import TopicMemoryStore


def lookup_recent_topic_context(
    store: TopicMemoryStore, cluster: BriefingCluster
) -> list[TopicMemory]:
    """Find existing topic memories by topic ID or normalized title tokens."""
    memories: list[TopicMemory] = []
    for topic_id in cluster.topic_ids:
        memory = store.get(topic_id)
        if memory is not None:
            memories.append(memory)
    return memories


def suggest_aliases(
    cluster: BriefingCluster, memories: list[TopicMemory]
) -> list[TopicAliasSuggestion]:
    """Suggest aliases without applying durable topic merges."""
    suggestions: list[TopicAliasSuggestion] = []
    title = normalize_title(cluster.title)
    for memory in memories:
        if (
            title
            and title != normalize_title(memory.name)
            and title not in memory.aliases
        ):
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
