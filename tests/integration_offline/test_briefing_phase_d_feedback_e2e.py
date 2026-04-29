"""D08 — Phase D feedback loop end-to-end (offline)."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from research_pipeline.briefing.feedback_audit import (
    audit_promotion_record,
    safe_rollback,
)
from research_pipeline.briefing.feedback_store import BriefingFeedbackStore
from research_pipeline.briefing.models import (
    AccessMethod,
    BriefingCluster,
    FeedbackSignal,
    IntelligenceEvent,
    SourceClass,
    SourcePolicy,
)
from research_pipeline.briefing.preference_update import apply_preference_updates
from research_pipeline.briefing.rank import (
    RankingOptions,
    explicit_feedback_components,
    rank_clusters,
)
from research_pipeline.briefing.weekly import render_feedback_section


def _cluster(cluster_id: str, *, topic_id: str, source_id: str) -> BriefingCluster:
    event = IntelligenceEvent(
        event_id=f"{cluster_id}:e1",
        source_name=f"Source {source_id}",
        source_id=source_id,
        source_type=SourceClass.PRIMARY_ARTIFACT,
        source_policy=SourcePolicy.PUBLIC_OFFICIAL,
        item_type="release",
        canonical_url=f"https://example.com/{cluster_id}",
        title=f"Event {cluster_id}",
        retrieved_at="2026-06-12T00:00:00Z",
        collection_method=AccessMethod.API,
        content_hash=f"content-{cluster_id}",
        dedup_key=f"dedup-{cluster_id}",
        published_at="2026-06-12T10:00:00Z",
        summary_hint="Detailed reproducibility package with benchmarks",
    )
    return BriefingCluster(
        cluster_id=cluster_id,
        title=f"Cluster {cluster_id}",
        primary_event_id=event.event_id,
        event_ids=(event.event_id,),
        topic_ids=(topic_id,),
        canonical_urls=(event.canonical_url,),
        first_seen_at="2026-06-12T00:00:00Z",
        last_seen_at="2026-06-12T00:00:00Z",
        source_classes=(SourceClass.PRIMARY_ARTIFACT,),
        primary_artifact_present=True,
        events=(event,),
    )


def test_phase_d_explicit_feedback_loop(tmp_path: Path) -> None:
    """Record → apply → re-rank → rollback → weekly section."""
    db = tmp_path / "feedback.db"
    cluster = _cluster("c1", topic_id="topic_alpha", source_id="src_a")

    # 1. Baseline rank with no feedback weights.
    baseline = rank_clusters([cluster], options=RankingOptions(min_rank_score=0.0))[0]
    baseline_score = baseline.rank_score

    # 2. Record three explicit positive feedback events.
    store = BriefingFeedbackStore(db)
    try:
        for i in range(3):
            store.record(
                target_type="topic",
                target_id="topic_alpha",
                signal=FeedbackSignal.MORE_LIKE_THIS,
                reason=f"explicit-{i}",
            )

        # 3. Apply preference updates with a review record.
        adjustments = apply_preference_updates(
            store, min_feedback=3, review_record="weekly-review-2026-06-12"
        )
        assert len(adjustments) == 1
        record = adjustments[0]
        ok, issues = audit_promotion_record(record)
        assert ok, issues

        # 4. Re-rank with derived feedback weights.
        weights = store.weights_by_target()
        boosted = rank_clusters(
            [cluster],
            options=RankingOptions(min_rank_score=0.0, feedback_weights=weights),
        )[0]
        topic_adj, _src_adj, neg_penalty = explicit_feedback_components(
            cluster, weights
        )
        assert topic_adj > 0
        assert neg_penalty == 0
        assert boosted.rank_score > baseline_score
        assert round(boosted.rank_score - baseline_score, 6) == round(topic_adj, 6)

        # 5. Render weekly feedback section.
        section = render_feedback_section(store)
        assert "topic:topic_alpha" in section
        assert "more_like_this: 3" in section
    finally:
        store.close()

    # 6. Rollback the durable change.
    receipt = safe_rollback(db, str(record["adjustment_id"]))
    assert receipt["rolled_back"] is True
    conn = sqlite3.connect(str(db))
    try:
        rows = conn.execute("SELECT * FROM preference_adjustments").fetchall()
    finally:
        conn.close()
    assert rows == []


def test_phase_d_conflicting_feedback_does_not_promote(tmp_path: Path) -> None:
    """Conflicting positive+negative feedback on same target → no-op."""
    db = tmp_path / "feedback.db"
    store = BriefingFeedbackStore(db)
    try:
        for _ in range(2):
            store.record(
                target_type="topic",
                target_id="topic_x",
                signal=FeedbackSignal.MORE_LIKE_THIS,
            )
        for _ in range(2):
            store.record(
                target_type="topic",
                target_id="topic_x",
                signal=FeedbackSignal.LESS_LIKE_THIS,
            )
        result = apply_preference_updates(store, min_feedback=3)
    finally:
        store.close()
    assert result == []
    conn = sqlite3.connect(str(db))
    try:
        rows = conn.execute("SELECT * FROM preference_adjustments").fetchall()
    finally:
        conn.close()
    assert rows == []
