"""Deterministic topic lifecycle classification for daily briefing clusters."""

from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from research_pipeline.briefing.models import BriefingCluster, TopicMemory
from research_pipeline.briefing.normalize import normalize_title
from research_pipeline.briefing.topic_memory_store import TopicMemoryStore

LifecycleStatus = Literal["new", "active", "cooling", "dormant", "resurfaced"]


def classify_cluster_lifecycle(
    store: TopicMemoryStore | None,
    cluster: BriefingCluster,
    run_date: str,
) -> LifecycleStatus:
    """Classify the lifecycle status for a cluster without mutating memory state."""
    if store is None:
        return "new"

    memory = _select_memory(store, cluster)
    if memory is None:
        return "new"

    days_since_report = _days_since(memory.last_reported_at, run_date)

    if days_since_report >= 14 and cluster.primary_artifact_present:
        return "resurfaced"

    if days_since_report >= 30 and not cluster.primary_artifact_present:
        return "dormant"

    if memory.fatigue_score >= 0.7 or memory.report_count_7d >= 3:
        return "cooling"

    return "active"


def _select_memory(
    store: TopicMemoryStore,
    cluster: BriefingCluster,
) -> TopicMemory | None:
    for topic_id in cluster.topic_ids:
        memory = store.get(topic_id)
        if memory is not None:
            return memory

    normalized_title = normalize_title(cluster.title)
    if not normalized_title:
        return None

    title_matches: list[TopicMemory] = []
    for memory in store.list_memories():
        if normalize_title(memory.name) == normalized_title:
            title_matches.append(memory)
            continue
        if normalized_title in {normalize_title(alias) for alias in memory.aliases}:
            title_matches.append(memory)

    if not title_matches:
        return None

    return max(title_matches, key=lambda item: item.last_reported_at or "")


def _days_since(last_reported_at: str | None, run_date: str) -> int:
    if last_reported_at is None:
        return 0
    prior = _parse_date(last_reported_at)
    current = _parse_date(run_date)
    if prior is None or current is None:
        return 0
    return max(0, (current - prior).days)


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value[:10]).date()
    except ValueError:
        return None
