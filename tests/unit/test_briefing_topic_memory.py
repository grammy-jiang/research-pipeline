from __future__ import annotations

import pytest
from pydantic import ValidationError

from research_pipeline.briefing import (
    TopicAliasSuggestion,
    TopicMemory,
    TopicMemoryWriteRecord,
)


def test_topic_memory_roundtrip_preserves_values() -> None:
    memory = TopicMemory(
        topic_id="topic_llm_memory",
        name="LLM memory",
        aliases=("agent memory",),
        first_seen_at="2026-04-01",
        last_seen_at="2026-04-29",
        status="resurfaced",
        summary="Recurring topic with renewed evidence.",
        key_entities=("MCP", "memory"),
        canonical_clusters=("cluster_a", "cluster_b"),
        interest_score=1.5,
        fatigue_score=0.75,
        last_reported_at="2026-04-29",
        report_count_7d=1,
        report_count_30d=3,
    )

    restored = TopicMemory.model_validate_json(memory.model_dump_json())

    assert restored == memory


def test_topic_memory_write_record_requires_metadata_and_source_ids() -> None:
    record = TopicMemoryWriteRecord(
        timestamp="2026-04-29T10:00:00Z",
        topic_id="topic_llm_memory",
        trigger="cluster_reported",
        effect="fatigue_penalty_increased",
        rollback="restore previous topic row",
        source_cluster_ids=("cluster_a",),
        source_event_ids=("event_1", "event_2"),
        owner="B01_topic_memory_models",
        review_required=False,
    )

    restored = TopicMemoryWriteRecord.model_validate_json(record.model_dump_json())

    assert restored == record


def test_topic_memory_write_record_rejects_missing_source_ids() -> None:
    with pytest.raises(ValidationError, match="at least one source cluster or event"):
        TopicMemoryWriteRecord(
            timestamp="2026-04-29T10:00:00Z",
            topic_id="topic_llm_memory",
            trigger="cluster_reported",
            effect="fatigue_penalty_increased",
            rollback="restore previous topic row",
            owner="brief run",
            review_required=True,
        )


def test_topic_alias_suggestion_requires_review_record_after_decision() -> None:
    with pytest.raises(ValidationError, match="review_record is required"):
        TopicAliasSuggestion(
            suggestion_id="alias_1",
            created_at="2026-04-29T10:00:00Z",
            topic_id="topic_llm_memory",
            suggested_alias="context memory",
            reason="same canonical URL and title tokens",
            status="approved",
            review_record="",
        )


def test_topic_alias_suggestion_allows_pending_without_review_record() -> None:
    suggestion = TopicAliasSuggestion(
        suggestion_id="alias_2",
        created_at="2026-04-29T10:00:00Z",
        topic_id="topic_llm_memory",
        suggested_alias="context memory",
        reason="same canonical URL and title tokens",
        status="pending",
    )

    assert suggestion.review_record == ""
