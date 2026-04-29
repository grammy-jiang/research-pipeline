"""D01 — feedback domain model + validation tests."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from research_pipeline.briefing.feedback import (
    ALLOWED_TARGET_TYPES,
    NEGATIVE_SIGNALS,
    NEUTRAL_SIGNALS,
    POSITIVE_SIGNALS,
    classify_signal,
    feedback_target_key,
    is_conflicting,
    validate_feedback_input,
)
from research_pipeline.briefing.models import FeedbackEvent, FeedbackSignal

# ---------------------------------------------------------------------------
# FeedbackSignal enum surface
# ---------------------------------------------------------------------------


def test_feedback_signal_values_are_explicit() -> None:
    expected = {
        "keep",
        "hide",
        "more_like_this",
        "less_like_this",
        "too_noisy",
        "already_known",
        "not_actionable",
        "useful",
        "neutral",
        "not_useful",
        "wrong_cadence",
    }
    assert {member.value for member in FeedbackSignal} == expected


def test_feedback_signal_partitions_are_disjoint_and_complete() -> None:
    assert POSITIVE_SIGNALS.isdisjoint(NEGATIVE_SIGNALS)
    assert POSITIVE_SIGNALS.isdisjoint(NEUTRAL_SIGNALS)
    assert NEGATIVE_SIGNALS.isdisjoint(NEUTRAL_SIGNALS)
    assert set(FeedbackSignal) == (
        POSITIVE_SIGNALS | NEGATIVE_SIGNALS | NEUTRAL_SIGNALS
    )


def test_classify_signal_round_trip() -> None:
    for signal in POSITIVE_SIGNALS:
        assert classify_signal(signal) == "positive"
    for signal in NEGATIVE_SIGNALS:
        assert classify_signal(signal) == "negative"
    for signal in NEUTRAL_SIGNALS:
        assert classify_signal(signal) == "neutral"


# ---------------------------------------------------------------------------
# FeedbackEvent (target_type Literal, strength bounds, frozen)
# ---------------------------------------------------------------------------


def test_feedback_event_accepts_valid_target_types() -> None:
    for target_type in ("event", "cluster", "topic", "source", "dossier"):
        event = FeedbackEvent(
            feedback_id=f"feedback_{target_type}",
            timestamp="2026-04-27T00:00:00Z",
            target_type=target_type,  # type: ignore[arg-type]
            target_id="abc",
            signal_type=FeedbackSignal.KEEP,
        )
        assert event.target_type == target_type


def test_feedback_event_rejects_unknown_target_type() -> None:
    with pytest.raises(ValidationError):
        FeedbackEvent(
            feedback_id="feedback_x",
            timestamp="2026-04-27T00:00:00Z",
            target_type="behavior",  # type: ignore[arg-type]
            target_id="abc",
            signal_type=FeedbackSignal.KEEP,
        )


def test_feedback_event_rejects_out_of_range_strength() -> None:
    with pytest.raises(ValidationError):
        FeedbackEvent(
            feedback_id="feedback_neg",
            timestamp="2026-04-27T00:00:00Z",
            target_type="cluster",
            target_id="abc",
            signal_type=FeedbackSignal.KEEP,
            strength=-0.1,
        )
    with pytest.raises(ValidationError):
        FeedbackEvent(
            feedback_id="feedback_high",
            timestamp="2026-04-27T00:00:00Z",
            target_type="cluster",
            target_id="abc",
            signal_type=FeedbackSignal.KEEP,
            strength=5.5,
        )


def test_feedback_event_is_frozen() -> None:
    event = FeedbackEvent(
        feedback_id="feedback_frozen",
        timestamp="2026-04-27T00:00:00Z",
        target_type="cluster",
        target_id="abc",
        signal_type=FeedbackSignal.KEEP,
    )
    with pytest.raises(ValidationError):
        event.target_id = "changed"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# validate_feedback_input
# ---------------------------------------------------------------------------


def test_validate_feedback_input_normalises_string_signal() -> None:
    target_type, signal, strength = validate_feedback_input(
        target_type="cluster",
        target_id="cluster_abc",
        signal="keep",
        strength=2.0,
    )
    assert target_type == "cluster"
    assert signal is FeedbackSignal.KEEP
    assert strength == 2.0


def test_validate_feedback_input_rejects_unsupported_signal() -> None:
    with pytest.raises(ValueError, match="unsupported feedback signal"):
        validate_feedback_input(
            target_type="cluster",
            target_id="cluster_abc",
            signal="click",
        )


def test_validate_feedback_input_rejects_malformed_target_id() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        validate_feedback_input(
            target_type="cluster",
            target_id="   ",
            signal=FeedbackSignal.KEEP,
        )
    with pytest.raises(ValueError, match="malformed"):
        validate_feedback_input(
            target_type="cluster",
            target_id="cluster id with spaces",
            signal=FeedbackSignal.KEEP,
        )
    with pytest.raises(ValueError, match="malformed"):
        validate_feedback_input(
            target_type="cluster",
            target_id="cluster\nnewline",
            signal=FeedbackSignal.KEEP,
        )


def test_validate_feedback_input_rejects_unknown_target_type() -> None:
    with pytest.raises(ValueError, match="target_type"):
        validate_feedback_input(
            target_type="behavior",
            target_id="abc",
            signal=FeedbackSignal.KEEP,
        )


def test_validate_feedback_input_rejects_strength_out_of_range() -> None:
    with pytest.raises(ValueError, match="strength"):
        validate_feedback_input(
            target_type="cluster",
            target_id="abc",
            signal=FeedbackSignal.KEEP,
            strength=10.0,
        )


def test_allowed_target_types_match_model_literal() -> None:
    assert (
        frozenset({"event", "cluster", "topic", "source", "dossier"})
        == ALLOWED_TARGET_TYPES
    )


# ---------------------------------------------------------------------------
# is_conflicting + feedback_target_key helpers
# ---------------------------------------------------------------------------


def _event(signal: FeedbackSignal, target_id: str = "abc") -> FeedbackEvent:
    return FeedbackEvent(
        feedback_id=f"feedback_{signal.value}_{target_id}",
        timestamp="2026-04-27T00:00:00Z",
        target_type="cluster",
        target_id=target_id,
        signal_type=signal,
    )


def test_is_conflicting_detects_pos_and_neg() -> None:
    assert is_conflicting([_event(FeedbackSignal.KEEP), _event(FeedbackSignal.HIDE)])


def test_is_conflicting_returns_false_when_only_positive() -> None:
    assert not is_conflicting(
        [_event(FeedbackSignal.KEEP), _event(FeedbackSignal.MORE_LIKE_THIS)]
    )


def test_is_conflicting_ignores_neutral_only() -> None:
    assert not is_conflicting([_event(FeedbackSignal.NEUTRAL)])


def test_feedback_target_key_format() -> None:
    assert feedback_target_key("cluster", "cluster_abc") == "cluster:cluster_abc"
