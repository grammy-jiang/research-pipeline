"""Deterministic briefing ranking.

Keywords: SPECTER2, semantic re-ranking, embedding scoring.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from research_pipeline.briefing.models import BriefingCluster, SourceClass
from research_pipeline.briefing.topic_memory import TopicMemoryStore

SOURCE_CLASS_WEIGHTS: dict[SourceClass, float] = {
    SourceClass.PRIMARY_ARTIFACT: 3.0,
    SourceClass.IMPLEMENTATION_SOURCE: 2.7,
    SourceClass.ACADEMIC_SOURCE: 2.4,
    SourceClass.NEWSLETTER: 1.4,
    SourceClass.TECHNICAL_DISCUSSION: 1.0,
    SourceClass.MEDIA_NEWS: 0.6,
    SourceClass.VIDEO_AUDIO: 0.5,
    SourceClass.SOCIAL_SIGNAL: 0.2,
}

HYPE_WORDS = {
    "breakthrough",
    "game-changing",
    "revolutionary",
    "shocking",
    "insane",
    "mind-blowing",
    "killer",
}


@dataclass(frozen=True)
class RankingOptions:
    """Inputs controlling deterministic ranking."""

    watchlist_terms: tuple[str, ...] = ()
    max_items: int = 10
    feedback_weights: dict[str, float] | None = None
    topic_memory: TopicMemoryStore | None = None
    # Caveat (#123): min_rank_score is a fixed tuned constant for the daily
    # brief, not relative to the run's score distribution. It is overridable
    # here via RankingOptions; a distribution-relative cutoff is future work.
    min_rank_score: float = 4.0


def rank_clusters(
    clusters: list[BriefingCluster],
    *,
    source_weights: dict[str, tuple[float, float]] | None = None,
    options: RankingOptions | None = None,
) -> list[BriefingCluster]:
    """Score and sort clusters with the Phase A tie-breaker contract."""
    opts = options or RankingOptions()
    source_weights = source_weights or {}
    ranked = [_score_cluster(cluster, source_weights, opts) for cluster in clusters]
    eligible = [
        cluster
        for cluster in ranked
        if cluster.rank_score >= opts.min_rank_score
        and not _low_information_cluster(cluster)
    ]
    return sorted(
        eligible,
        key=lambda cluster: (
            -cluster.rank_score,
            -max(SOURCE_CLASS_WEIGHTS.get(cls, 0.0) for cls in cluster.source_classes),
            -_published_sort_value(cluster),
            -cluster.authority_score,
            cluster.title.lower(),
            cluster.cluster_id,
        ),
    )[: opts.max_items]


def _published_sort_key(cluster: BriefingCluster) -> str:
    dates = [event.published_at or "" for event in cluster.events]
    return max(dates) if dates else cluster.last_seen_at


def _published_sort_value(cluster: BriefingCluster) -> float:
    published = _published_sort_key(cluster)
    if not published:
        return 0.0
    normalized = published.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(normalized).timestamp()
    except ValueError:
        # Fall back to lexical ordering to keep behavior deterministic.
        return 0.0


def _score_cluster(
    cluster: BriefingCluster,
    source_weights: dict[str, tuple[float, float]],
    options: RankingOptions,
) -> BriefingCluster:
    class_weight = max(
        SOURCE_CLASS_WEIGHTS.get(source_class, 0.0)
        for source_class in cluster.source_classes
    )
    trust_weight = sum(
        source_weights.get(event.source_id, (1.0, 0.0))[0] for event in cluster.events
    ) / max(1, len(cluster.events))
    noise_weight = sum(
        source_weights.get(event.source_id, (1.0, 0.0))[1] for event in cluster.events
    ) / max(1, len(cluster.events))
    primary_bonus = 1.0 if cluster.primary_artifact_present else 0.0
    watchlist_bonus = _watchlist_bonus(cluster, options.watchlist_terms)
    hype_penalty = _hype_penalty(cluster)
    low_info_penalty = 2.0 if _low_information_cluster(cluster) else 0.0
    feedback = _feedback_bonus(cluster, options.feedback_weights or {})
    fatigue, resurfaced = _memory_adjustments(cluster, options.topic_memory)
    novelty_type = _novelty_type(cluster, options.topic_memory, fatigue, resurfaced)
    suggested_action = _suggested_action(cluster, fatigue, hype_penalty)
    score = (
        class_weight
        + trust_weight
        - noise_weight
        + primary_bonus
        + watchlist_bonus
        + feedback
        + resurfaced
        - hype_penalty
        - low_info_penalty
        - cluster.duplicate_penalty
        - fatigue
    )
    explanation = (
        f"class={class_weight:.2f}; trust={trust_weight:.2f}; "
        f"noise={noise_weight:.2f}; primary={primary_bonus:.2f}; "
        f"watchlist={watchlist_bonus:.2f}; feedback={feedback:.2f}; "
        f"fatigue={fatigue:.2f}; hype={hype_penalty:.2f}; "
        f"low_info={low_info_penalty:.2f}; "
        f"duplicate={cluster.duplicate_penalty:.2f}"
    )
    return cluster.model_copy(
        update={
            "authority_score": class_weight + trust_weight,
            "engineering_usefulness_score": primary_bonus + watchlist_bonus,
            "personal_interest_score": feedback,
            "hype_penalty": hype_penalty,
            "fatigue_penalty": fatigue,
            "resurfaced_boost": resurfaced,
            "novelty_type": novelty_type,
            "suggested_action": suggested_action,
            "rank_score": round(score, 6),
            "ranking_explanation": explanation,
        }
    )


def _watchlist_bonus(cluster: BriefingCluster, terms: tuple[str, ...]) -> float:
    text = " ".join(
        [cluster.title, *(event.summary_hint for event in cluster.events)]
    ).lower()
    return min(1.5, 0.5 * sum(1 for term in terms if term.lower() in text))


def _hype_penalty(cluster: BriefingCluster) -> float:
    text = " ".join(
        [cluster.title, *(event.summary_hint for event in cluster.events)]
    ).lower()
    penalty = 0.0 if cluster.primary_artifact_present else 1.0
    if any(word in text for word in HYPE_WORDS):
        penalty += 0.5
    if (
        SourceClass.SOCIAL_SIGNAL in cluster.source_classes
        and not cluster.primary_artifact_present
    ):
        penalty += 1.0
    return penalty


def _feedback_bonus(cluster: BriefingCluster, weights: dict[str, float]) -> float:
    total = weights.get(f"cluster:{cluster.cluster_id}", 0.0)
    for topic_id in cluster.topic_ids:
        total += weights.get(f"topic:{topic_id}", 0.0)
    for event in cluster.events:
        total += weights.get(f"source:{event.source_id}", 0.0)
        total += weights.get(f"event:{event.event_id}", 0.0)
    return total


def explicit_feedback_components(
    cluster: BriefingCluster,
    weights: dict[str, float],
) -> tuple[float, float, float]:
    """Decompose explicit feedback weights for a cluster.

    Returns ``(topic_adjustment, source_adjustment, negative_penalty)``
    where ``negative_penalty`` is a non-negative magnitude expressing the
    *amount* of negative explicit feedback.  This satisfies the Phase D
    ranking contract::

        rank_score = phase_c_rank_score
                     + explicit_topic_adjustment
                     + explicit_source_adjustment
                     - explicit_negative_penalty

    Only ``topic:``, ``cluster:``, ``source:``, and ``event:`` keys
    contribute — never behavioural signals.
    """
    topic_adj = 0.0
    for topic_id in cluster.topic_ids:
        topic_adj += weights.get(f"topic:{topic_id}", 0.0)
    topic_adj += weights.get(f"cluster:{cluster.cluster_id}", 0.0)

    source_adj = 0.0
    for event in cluster.events:
        source_adj += weights.get(f"source:{event.source_id}", 0.0)
        source_adj += weights.get(f"event:{event.event_id}", 0.0)

    pos_topic = max(topic_adj, 0.0)
    pos_source = max(source_adj, 0.0)
    neg_penalty = max(-topic_adj, 0.0) + max(-source_adj, 0.0)
    return pos_topic, pos_source, neg_penalty


def _memory_adjustments(
    cluster: BriefingCluster, store: TopicMemoryStore | None
) -> tuple[float, float]:
    if store is None:
        return 0.0, 0.0
    fatigue = 0.0
    resurfaced = 0.0
    for topic_id in cluster.topic_ids:
        memory = store.get(topic_id)
        if memory is None:
            continue
        fatigue = max(fatigue, memory.fatigue_score)
        if memory.status == "resurfaced":
            resurfaced = max(resurfaced, 1.0)
    return fatigue, resurfaced


def _novelty_type(
    cluster: BriefingCluster,
    store: TopicMemoryStore | None,
    fatigue: float,
    resurfaced: float,
) -> Literal["new", "active", "cooling", "dormant", "resurfaced"]:
    if resurfaced > 0.0:
        return "resurfaced"
    if store is None:
        return "new"
    prior_seen = any(store.get(topic_id) is not None for topic_id in cluster.topic_ids)
    if not prior_seen:
        return "new"
    if fatigue >= 0.7:
        return "cooling"
    return "active"


def _suggested_action(
    cluster: BriefingCluster, fatigue: float, hype_penalty: float
) -> Literal["read", "try", "watch", "ignore"]:
    # Caveat (#123): the 1.5 hype/fatigue cutoffs are fixed tuned constants
    # for the daily brief, not relative to the run's score distribution; a
    # distribution-relative cutoff is future work.
    if hype_penalty > 1.5 or fatigue > 1.5:
        return "ignore"
    if SourceClass.IMPLEMENTATION_SOURCE in cluster.source_classes:
        return "try"
    if SourceClass.TECHNICAL_DISCUSSION in cluster.source_classes:
        return "watch"
    return "read"


def _low_information_cluster(cluster: BriefingCluster) -> bool:
    summaries = [
        (event.summary_hint or event.excerpt or "").strip().lower()
        for event in cluster.events
    ]
    text = " ".join(summaries)
    if not text:
        return True
    boilerplate = {
        "previous release",
        "bug fixes",
        "minor update",
        "miscellaneous updates",
        "duplicate release mention",
    }
    return any(item in text for item in boilerplate) and len(text.split()) < 12
