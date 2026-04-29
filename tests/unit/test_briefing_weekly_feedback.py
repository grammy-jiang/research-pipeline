"""D08 — weekly feedback section tests."""

from __future__ import annotations

from pathlib import Path

from research_pipeline.briefing.feedback_store import BriefingFeedbackStore
from research_pipeline.briefing.models import FeedbackSignal
from research_pipeline.briefing.weekly import render_feedback_section


def test_empty_store_renders_placeholder(tmp_path: Path) -> None:
    store = BriefingFeedbackStore(tmp_path / "f.db")
    try:
        section = render_feedback_section(store)
    finally:
        store.close()
    assert section.startswith("## Feedback & Source Quality")
    assert "No explicit feedback" in section


def test_section_lists_counts_and_top_targets(tmp_path: Path) -> None:
    store = BriefingFeedbackStore(tmp_path / "f.db")
    try:
        for i in range(3):
            store.record(
                target_type="topic",
                target_id="alpha",
                signal=FeedbackSignal.MORE_LIKE_THIS,
                reason=f"r{i}",
            )
        store.record(
            target_type="topic",
            target_id="beta",
            signal=FeedbackSignal.LESS_LIKE_THIS,
            reason="noise",
        )
        section = render_feedback_section(store)
    finally:
        store.close()
    assert "Total explicit feedback events: 4" in section
    assert "more_like_this: 3" in section
    assert "less_like_this: 1" in section
    assert "topic:alpha" in section
    assert "topic:beta" in section
    assert "Top boosted targets" in section
    assert "Top penalised targets" in section


def test_week_id_filters_events(tmp_path: Path) -> None:
    """When week_id is given, the renderer only includes matching timestamps."""
    store = BriefingFeedbackStore(tmp_path / "f.db")
    try:
        store.record(
            target_type="topic",
            target_id="alpha",
            signal=FeedbackSignal.MORE_LIKE_THIS,
        )
        # week_id far in the future filters out current event.
        section = render_feedback_section(store, week_id="2099-01")
    finally:
        store.close()
    assert "2099-01" in section
    assert "No explicit feedback" in section


def test_section_does_not_mention_behavioural_signals(tmp_path: Path) -> None:
    store = BriefingFeedbackStore(tmp_path / "f.db")
    try:
        store.record(
            target_type="topic",
            target_id="alpha",
            signal=FeedbackSignal.MORE_LIKE_THIS,
        )
        section = render_feedback_section(store)
    finally:
        store.close()
    forbidden = ("dwell", "click", "behavioural", "behavioral", "tracking")
    lowered = section.lower()
    for term in forbidden:
        assert term not in lowered
