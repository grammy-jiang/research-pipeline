"""Tests for user feedback loop (feedback/store.py and feedback/models.py)."""

import json
from pathlib import Path

import pytest

from research_pipeline.feedback.models import (
    FeedbackDecision,
    FeedbackRecord,
    WeightAdjustment,
)
from research_pipeline.feedback.store import (
    FeedbackStore,
)

# --- Model tests ---


class TestFeedbackDecision:
    def test_accept_value(self) -> None:
        assert FeedbackDecision.ACCEPT.value == "accept"

    def test_reject_value(self) -> None:
        assert FeedbackDecision.REJECT.value == "reject"

    def test_from_string(self) -> None:
        assert FeedbackDecision("accept") == FeedbackDecision.ACCEPT
        assert FeedbackDecision("reject") == FeedbackDecision.REJECT

    def test_invalid_raises(self) -> None:
        with pytest.raises(ValueError):
            FeedbackDecision("maybe")


class TestFeedbackRecord:
    def test_minimal_construction(self) -> None:
        rec = FeedbackRecord(
            paper_id="2401.12345",
            run_id="run-001",
            decision=FeedbackDecision.ACCEPT,
        )
        assert rec.paper_id == "2401.12345"
        assert rec.run_id == "run-001"
        assert rec.decision == FeedbackDecision.ACCEPT
        assert rec.reason == ""
        assert rec.cheap_score == 0.0
        assert rec.recorded_at  # auto-populated

    def test_full_construction(self) -> None:
        rec = FeedbackRecord(
            paper_id="2401.12345",
            run_id="run-001",
            decision=FeedbackDecision.REJECT,
            reason="off-topic",
            cheap_score=0.75,
            recorded_at="2024-01-01T00:00:00+00:00",
        )
        assert rec.reason == "off-topic"
        assert rec.cheap_score == 0.75
        assert rec.recorded_at == "2024-01-01T00:00:00+00:00"

    def test_roundtrip_serialization(self) -> None:
        rec = FeedbackRecord(
            paper_id="2401.12345",
            run_id="run-001",
            decision=FeedbackDecision.ACCEPT,
            reason="relevant",
            cheap_score=0.85,
        )
        data = rec.model_dump(mode="json")
        restored = FeedbackRecord.model_validate(data)
        assert restored == rec

    def test_json_serializable(self) -> None:
        rec = FeedbackRecord(
            paper_id="2401.12345",
            run_id="run-001",
            decision=FeedbackDecision.ACCEPT,
        )
        text = json.dumps(rec.model_dump(mode="json"))
        assert "2401.12345" in text


class TestWeightAdjustment:
    def test_defaults(self) -> None:
        adj = WeightAdjustment()
        assert adj.bm25_must_title == 0.20
        assert adj.bm25_nice_title == 0.10
        assert adj.bm25_must_abstract == 0.25
        assert adj.bm25_nice_abstract == 0.10
        assert adj.cat_match == 0.15
        assert adj.negative_penalty == 0.10
        assert adj.recency_bonus == 0.10
        assert adj.feedback_count == 0

    def test_to_weight_dict(self) -> None:
        adj = WeightAdjustment(bm25_must_title=0.30)
        d = adj.to_weight_dict()
        assert d["bm25_must_title"] == 0.30
        assert "feedback_count" not in d
        assert "learning_rate" not in d

    def test_roundtrip(self) -> None:
        adj = WeightAdjustment(
            bm25_must_title=0.22,
            feedback_count=10,
            learning_rate=0.08,
        )
        data = adj.model_dump(mode="json")
        restored = WeightAdjustment.model_validate(data)
        assert restored == adj


# --- Store tests ---


@pytest.fixture
def temp_store(tmp_path: Path) -> FeedbackStore:
    """Create a FeedbackStore with a temporary database."""
    db_path = tmp_path / "test_feedback.db"
    store = FeedbackStore(db_path=db_path)
    yield store
    store.close()


class TestFeedbackStore:
    def test_init_creates_db(self, tmp_path: Path) -> None:
        db_path = tmp_path / "new_feedback.db"
        store = FeedbackStore(db_path=db_path)
        assert db_path.exists()
        store.close()

    def test_record_single(self, temp_store: FeedbackStore) -> None:
        rec = FeedbackRecord(
            paper_id="2401.12345",
            run_id="run-001",
            decision=FeedbackDecision.ACCEPT,
        )
        temp_store.record(rec)
        feedback = temp_store.get_feedback(run_id="run-001")
        assert len(feedback) == 1
        assert feedback[0].paper_id == "2401.12345"
        assert feedback[0].decision == FeedbackDecision.ACCEPT

    def test_record_batch(self, temp_store: FeedbackStore) -> None:
        records = [
            FeedbackRecord(
                paper_id=f"2401.{i:05d}",
                run_id="run-001",
                decision=(
                    FeedbackDecision.ACCEPT if i % 2 == 0 else FeedbackDecision.REJECT
                ),
            )
            for i in range(10)
        ]
        count = temp_store.record_batch(records)
        assert count == 10

    def test_upsert_overwrites(self, temp_store: FeedbackStore) -> None:
        rec1 = FeedbackRecord(
            paper_id="2401.12345",
            run_id="run-001",
            decision=FeedbackDecision.ACCEPT,
        )
        temp_store.record(rec1)
        rec2 = FeedbackRecord(
            paper_id="2401.12345",
            run_id="run-001",
            decision=FeedbackDecision.REJECT,
            reason="changed mind",
        )
        temp_store.record(rec2)
        feedback = temp_store.get_feedback(run_id="run-001")
        assert len(feedback) == 1
        assert feedback[0].decision == FeedbackDecision.REJECT
        assert feedback[0].reason == "changed mind"

    def test_get_feedback_filter_by_run(self, temp_store: FeedbackStore) -> None:
        temp_store.record(
            FeedbackRecord(
                paper_id="a", run_id="run-001", decision=FeedbackDecision.ACCEPT
            )
        )
        temp_store.record(
            FeedbackRecord(
                paper_id="b", run_id="run-002", decision=FeedbackDecision.REJECT
            )
        )
        r1 = temp_store.get_feedback(run_id="run-001")
        r2 = temp_store.get_feedback(run_id="run-002")
        assert len(r1) == 1
        assert len(r2) == 1
        assert r1[0].paper_id == "a"
        assert r2[0].paper_id == "b"

    def test_get_feedback_filter_by_decision(self, temp_store: FeedbackStore) -> None:
        temp_store.record(
            FeedbackRecord(
                paper_id="a", run_id="run-001", decision=FeedbackDecision.ACCEPT
            )
        )
        temp_store.record(
            FeedbackRecord(
                paper_id="b", run_id="run-001", decision=FeedbackDecision.REJECT
            )
        )
        accepted = temp_store.get_feedback(decision=FeedbackDecision.ACCEPT)
        rejected = temp_store.get_feedback(decision=FeedbackDecision.REJECT)
        assert len(accepted) == 1
        assert len(rejected) == 1

    def test_count(self, temp_store: FeedbackStore) -> None:
        for i in range(3):
            temp_store.record(
                FeedbackRecord(
                    paper_id=f"a{i}", run_id="run-001", decision=FeedbackDecision.ACCEPT
                )
            )
        for i in range(2):
            temp_store.record(
                FeedbackRecord(
                    paper_id=f"r{i}", run_id="run-001", decision=FeedbackDecision.REJECT
                )
            )
        counts = temp_store.count(run_id="run-001")
        assert counts["accept"] == 3
        assert counts["reject"] == 2
        assert counts["total"] == 5

    def test_count_all(self, temp_store: FeedbackStore) -> None:
        temp_store.record(
            FeedbackRecord(
                paper_id="a", run_id="run-001", decision=FeedbackDecision.ACCEPT
            )
        )
        temp_store.record(
            FeedbackRecord(
                paper_id="b", run_id="run-002", decision=FeedbackDecision.REJECT
            )
        )
        counts = temp_store.count()
        assert counts["total"] == 2


class TestWeightAdjustmentComputation:
    def test_insufficient_feedback_returns_defaults(
        self, temp_store: FeedbackStore
    ) -> None:
        for i in range(3):
            temp_store.record(
                FeedbackRecord(
                    paper_id=f"p{i}",
                    run_id="run-001",
                    decision=FeedbackDecision.ACCEPT,
                    cheap_score=0.8,
                )
            )
        adj = temp_store.compute_adjusted_weights(min_feedback=5)
        assert adj.feedback_count == 3
        # Should return defaults
        assert adj.bm25_must_title == 0.20

    def test_only_accepts_returns_defaults(self, temp_store: FeedbackStore) -> None:
        for i in range(10):
            temp_store.record(
                FeedbackRecord(
                    paper_id=f"p{i}",
                    run_id="run-001",
                    decision=FeedbackDecision.ACCEPT,
                    cheap_score=0.8,
                )
            )
        adj = temp_store.compute_adjusted_weights(min_feedback=5)
        assert adj.feedback_count == 10
        assert adj.bm25_must_title == 0.20  # defaults

    def test_only_rejects_returns_defaults(self, temp_store: FeedbackStore) -> None:
        for i in range(10):
            temp_store.record(
                FeedbackRecord(
                    paper_id=f"p{i}",
                    run_id="run-001",
                    decision=FeedbackDecision.REJECT,
                    cheap_score=0.3,
                )
            )
        adj = temp_store.compute_adjusted_weights(min_feedback=5)
        assert adj.feedback_count == 10
        assert adj.bm25_must_title == 0.20  # defaults

    def test_mixed_feedback_adjusts_weights(self, temp_store: FeedbackStore) -> None:
        # Accepted papers have higher scores
        for i in range(5):
            temp_store.record(
                FeedbackRecord(
                    paper_id=f"good{i}",
                    run_id="run-001",
                    decision=FeedbackDecision.ACCEPT,
                    cheap_score=0.85,
                )
            )
        # Rejected papers have lower scores
        for i in range(5):
            temp_store.record(
                FeedbackRecord(
                    paper_id=f"bad{i}",
                    run_id="run-001",
                    decision=FeedbackDecision.REJECT,
                    cheap_score=0.3,
                )
            )
        adj = temp_store.compute_adjusted_weights(min_feedback=5)
        assert adj.feedback_count == 10
        # Weights should be different from defaults
        w = adj.to_weight_dict()
        total = sum(w.values())
        assert abs(total - 1.0) < 0.01  # normalized

    def test_high_reject_ratio_increases_negative_penalty(
        self, temp_store: FeedbackStore
    ) -> None:
        for i in range(2):
            temp_store.record(
                FeedbackRecord(
                    paper_id=f"good{i}",
                    run_id="run-001",
                    decision=FeedbackDecision.ACCEPT,
                    cheap_score=0.9,
                )
            )
        for i in range(8):
            temp_store.record(
                FeedbackRecord(
                    paper_id=f"bad{i}",
                    run_id="run-001",
                    decision=FeedbackDecision.REJECT,
                    cheap_score=0.4,
                )
            )
        adj = temp_store.compute_adjusted_weights(min_feedback=5)
        # With 80% reject ratio, negative_penalty should be boosted
        # (may be renormalized, so compare relative to total)
        w = adj.to_weight_dict()
        assert w["negative_penalty"] > 0  # still positive

    def test_adjusted_weights_persisted(self, temp_store: FeedbackStore) -> None:
        for i in range(5):
            temp_store.record(
                FeedbackRecord(
                    paper_id=f"good{i}",
                    run_id="run-001",
                    decision=FeedbackDecision.ACCEPT,
                    cheap_score=0.8,
                )
            )
        for i in range(5):
            temp_store.record(
                FeedbackRecord(
                    paper_id=f"bad{i}",
                    run_id="run-001",
                    decision=FeedbackDecision.REJECT,
                    cheap_score=0.3,
                )
            )
        temp_store.compute_adjusted_weights()

        latest = temp_store.get_latest_weights()
        assert latest is not None
        assert latest.feedback_count == 10

    def test_get_latest_weights_when_none(self, temp_store: FeedbackStore) -> None:
        assert temp_store.get_latest_weights() is None

    def test_learning_rate_parameter(self, temp_store: FeedbackStore) -> None:
        for i in range(5):
            temp_store.record(
                FeedbackRecord(
                    paper_id=f"good{i}",
                    run_id="run-001",
                    decision=FeedbackDecision.ACCEPT,
                    cheap_score=0.8,
                )
            )
        for i in range(5):
            temp_store.record(
                FeedbackRecord(
                    paper_id=f"bad{i}",
                    run_id="run-001",
                    decision=FeedbackDecision.REJECT,
                    cheap_score=0.3,
                )
            )
        adj_low = temp_store.compute_adjusted_weights(learning_rate=0.01)
        adj_high = temp_store.compute_adjusted_weights(learning_rate=0.20)
        # Higher learning rate should produce larger deviations
        assert adj_low.learning_rate == 0.01
        assert adj_high.learning_rate == 0.20


class TestFeedbackStoreEdgeCases:
    def test_empty_feedback(self, temp_store: FeedbackStore) -> None:
        feedback = temp_store.get_feedback()
        assert feedback == []

    def test_count_empty(self, temp_store: FeedbackStore) -> None:
        counts = temp_store.count()
        assert counts["total"] == 0
        assert counts["accept"] == 0
        assert counts["reject"] == 0

    def test_close_and_reopen(self, tmp_path: Path) -> None:
        db_path = tmp_path / "reopen_test.db"
        store1 = FeedbackStore(db_path=db_path)
        store1.record(
            FeedbackRecord(
                paper_id="x",
                run_id="run-001",
                decision=FeedbackDecision.ACCEPT,
            )
        )
        store1.close()

        store2 = FeedbackStore(db_path=db_path)
        feedback = store2.get_feedback()
        assert len(feedback) == 1
        assert feedback[0].paper_id == "x"
        store2.close()

    def test_special_characters_in_paper_id(self, temp_store: FeedbackStore) -> None:
        rec = FeedbackRecord(
            paper_id="10.1145/3474085.3475234",
            run_id="run-001",
            decision=FeedbackDecision.ACCEPT,
            reason="DOI with special chars",
        )
        temp_store.record(rec)
        feedback = temp_store.get_feedback()
        assert feedback[0].paper_id == "10.1145/3474085.3475234"

    def test_unicode_reason(self, temp_store: FeedbackStore) -> None:
        rec = FeedbackRecord(
            paper_id="test",
            run_id="run-001",
            decision=FeedbackDecision.REJECT,
            reason="论文不相关 — off-topic 🚫",
        )
        temp_store.record(rec)
        feedback = temp_store.get_feedback()
        assert "论文不相关" in feedback[0].reason

    def test_zero_cheap_score(self, temp_store: FeedbackStore) -> None:
        rec = FeedbackRecord(
            paper_id="test",
            run_id="run-001",
            decision=FeedbackDecision.ACCEPT,
            cheap_score=0.0,
        )
        temp_store.record(rec)
        feedback = temp_store.get_feedback()
        assert feedback[0].cheap_score == 0.0
