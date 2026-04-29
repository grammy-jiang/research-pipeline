"""Deterministic validation for Phase B topic-memory state."""

from __future__ import annotations

from datetime import datetime

from research_pipeline.briefing.memory_lookup import lookup_recent_topic_context
from research_pipeline.briefing.models import BriefingCluster, TopicMemory
from research_pipeline.briefing.topic_memory_store import TopicMemoryStore
from research_pipeline.briefing.validate import ValidationResult


def validate_topic_memory(
    clusters: list[BriefingCluster],
    *,
    topic_memory: TopicMemoryStore | None,
) -> ValidationResult:
    """Validate that Phase B memory-backed ranking state is consistent."""
    errors: list[str] = []
    warnings: list[str] = []
    queue_size = 0

    if topic_memory is None:
        for cluster in clusters:
            if _requires_memory(cluster):
                errors.append(
                    "cluster "
                    f"{cluster.cluster_id} has memory-based novelty but "
                    "topic_memory store is unavailable"
                )
        return ValidationResult(
            passed=not errors,
            errors=tuple(errors),
            warnings=tuple(warnings),
            metrics={
                "cluster_count": len(clusters),
                "queue_size": queue_size,
            },
        )

    try:
        queue_size = len(topic_memory.list_alias_suggestions(status=None))
    except Exception as exc:
        errors.append(f"invalid alias review queue: {exc}")

    for cluster in clusters:
        memories = lookup_recent_topic_context(topic_memory, cluster)
        explicit_matches = [
            memory for memory in memories if memory.topic_id in cluster.topic_ids
        ]
        has_explicit_match = bool(explicit_matches)

        if not memories:
            if _requires_memory(cluster):
                errors.append(
                    f"cluster {cluster.cluster_id} has no matching topic memory"
                )
            continue

        if not has_explicit_match and len(memories) > 1:
            errors.append(
                f"cluster {cluster.cluster_id} matches multiple topic memories "
                "via fallback"
            )
            continue

        if cluster.topic_ids and not has_explicit_match and _requires_memory(cluster):
            errors.append(
                f"cluster {cluster.cluster_id} current evidence contradicts "
                "stale memory fallback"
            )
            continue

        memory = explicit_matches[0] if explicit_matches else memories[0]
        errors.extend(_validate_cluster_memory_state(cluster, memory))

    return ValidationResult(
        passed=not errors,
        errors=tuple(errors),
        warnings=tuple(warnings),
        metrics={
            "cluster_count": len(clusters),
            "queue_size": queue_size,
        },
    )


def _validate_cluster_memory_state(
    cluster: BriefingCluster,
    memory: TopicMemory,
) -> list[str]:
    errors: list[str] = []
    if cluster.novelty_type == "cooling" and not _is_cooling_consistent(memory):
        errors.append(
            f"cluster {cluster.cluster_id} marked cooling without fatigued prior memory"
        )
    if cluster.novelty_type == "resurfaced" and not _is_resurfaced_consistent(
        memory, cluster.last_seen_at
    ):
        errors.append(
            f"cluster {cluster.cluster_id} marked resurfaced without dormant "
            "prior memory"
        )
    if cluster.fatigue_penalty > 0.0 and memory.fatigue_score <= 0.0:
        errors.append(
            f"cluster {cluster.cluster_id} has fatigue penalty without fatigue history"
        )
    if cluster.resurfaced_boost > 0.0 and not _is_resurfaced_consistent(
        memory, cluster.last_seen_at
    ):
        errors.append(
            f"cluster {cluster.cluster_id} has resurfaced boost without dormant "
            "prior memory"
        )
    return errors


def _requires_memory(cluster: BriefingCluster) -> bool:
    return (
        cluster.novelty_type in {"cooling", "resurfaced"}
        or cluster.fatigue_penalty > 0.0
        or cluster.resurfaced_boost > 0.0
    )


def _is_cooling_consistent(memory: TopicMemory) -> bool:
    return memory.fatigue_score >= 0.7 or memory.report_count_7d >= 3


def _is_resurfaced_consistent(memory: TopicMemory, run_date: str) -> bool:
    if memory.status == "resurfaced":
        return True
    if memory.last_reported_at is None:
        return False
    return _days_since(memory.last_reported_at, run_date) >= 14


def _days_since(start: str, end: str) -> int:
    start_date = datetime.fromisoformat(start[:10]).date()
    end_date = datetime.fromisoformat(end[:10]).date()
    return (end_date - start_date).days
