"""D08 — weekly feedback / source-quality section renderer.

Produces a Markdown section summarising explicit feedback received in
the briefing system: counts by signal, top targets by net weight,
conflict counts.  Used by the weekly digest to give the user
visibility into how their explicit feedback is shaping ranking.

Phase D scope: explicit feedback ONLY.  Behavioural signals are never
surfaced.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable

from research_pipeline.briefing.feedback import (
    NEGATIVE_SIGNALS,
    POSITIVE_SIGNALS,
    BriefingFeedbackStore,
)
from research_pipeline.briefing.models import FeedbackEvent, FeedbackSignal


def _week_filter(
    events: Iterable[FeedbackEvent], week_id: str | None
) -> list[FeedbackEvent]:
    """Filter events whose ``timestamp`` starts with ``week_id`` prefix.

    ``week_id`` is treated as a string prefix on the ISO timestamp
    (e.g. ``"2026-04"``).  When ``None``, returns all events.
    """
    if week_id is None:
        return list(events)
    return [event for event in events if event.timestamp.startswith(week_id)]


def render_feedback_section(
    store: BriefingFeedbackStore,
    *,
    week_id: str | None = None,
) -> str:
    """Render the weekly explicit-feedback section as Markdown."""
    events = _week_filter(store.list_feedback(), week_id)
    if not events:
        header = (
            f"## Feedback & Source Quality — {week_id}\n\n"
            if week_id
            else "## Feedback & Source Quality\n\n"
        )
        return header + "_No explicit feedback recorded._\n"

    counts: Counter[FeedbackSignal] = Counter(event.signal_type for event in events)
    weights = store.weights_by_target()
    conflicts = store.conflict_summary()

    lines: list[str] = []
    title = (
        f"## Feedback & Source Quality — {week_id}"
        if week_id
        else "## Feedback & Source Quality"
    )
    lines.append(title)
    lines.append("")
    lines.append(f"- Total explicit feedback events: {len(events)}")
    lines.append("- Counts by signal:")
    for signal in sorted(FeedbackSignal, key=lambda value: value.value):
        if counts.get(signal, 0):
            lines.append(f"  - {signal.value}: {counts[signal]}")
    lines.append("")

    pos = sorted(
        ((k, v) for k, v in weights.items() if v > 0),
        key=lambda kv: (-kv[1], kv[0]),
    )[:5]
    neg = sorted(
        ((k, v) for k, v in weights.items() if v < 0),
        key=lambda kv: (kv[1], kv[0]),
    )[:5]

    lines.append("### Top boosted targets")
    if pos:
        for key, weight in pos:
            lines.append(f"- `{key}`: {weight:+.3f}")
    else:
        lines.append("- _none_")
    lines.append("")

    lines.append("### Top penalised targets")
    if neg:
        for key, weight in neg:
            lines.append(f"- `{key}`: {weight:+.3f}")
    else:
        lines.append("- _none_")
    lines.append("")

    pos_total = sum(1 for s in counts if s in POSITIVE_SIGNALS)
    neg_total = sum(1 for s in counts if s in NEGATIVE_SIGNALS)
    lines.append("### Source quality summary")
    lines.append(f"- Positive signals across {pos_total} signal type(s).")
    lines.append(f"- Negative signals across {neg_total} signal type(s).")
    lines.append(f"- Targets with conflicting feedback: {len(conflicts)}")
    lines.append("")

    return "\n".join(lines) + "\n"


__all__ = ["render_feedback_section"]
