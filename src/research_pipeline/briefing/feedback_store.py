"""Canonical Phase D02 import surface for the explicit feedback event store.

This module mirrors the Phase B02 pattern in
:mod:`research_pipeline.briefing.topic_memory_store`: it provides a
single, ticket-owned import path while the implementation continues to
live in :mod:`research_pipeline.briefing.feedback` so existing call sites
(``briefing.workflow``, Phase A/B/C tests) keep working unchanged.

The store appends explicit-feedback events only; behavioural signals
(read-time, click-tracking, dwell-time) are *not* accepted.  All writes
flow through :func:`research_pipeline.briefing.feedback.validate_feedback_input`,
so malformed target IDs and unsupported signals are rejected at the
boundary.
"""

from __future__ import annotations

from research_pipeline.briefing.feedback import (
    NEGATIVE_SIGNALS,
    POSITIVE_SIGNALS,
    BriefingFeedbackStore,
    feedback_target_key,
    is_conflicting,
)

#: Public alias so ``from briefing.feedback_store import FeedbackStore`` works.
FeedbackStore = BriefingFeedbackStore

__all__ = [
    "NEGATIVE_SIGNALS",
    "POSITIVE_SIGNALS",
    "BriefingFeedbackStore",
    "FeedbackStore",
    "feedback_target_key",
    "is_conflicting",
]
