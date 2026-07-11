"""Explicit feedback domain logic.

This module owns the Phase D explicit-feedback domain surface:

* ``FeedbackEvent`` and ``FeedbackSignal`` re-exports from :mod:`models`.
* Signal classification (positive/negative/neutral).
* Pure validation helpers used by the CLI, store, manual-review importer,
  and audit module.

The SQLite-backed event store lives in
:mod:`research_pipeline.briefing.feedback_store` (Phase D02).  This module
keeps the store class as ``BriefingFeedbackStore`` for backward
compatibility with Phase A/B/C call sites that imported it from here.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from pathlib import Path
from typing import Literal, get_args

from research_pipeline.briefing.models import FeedbackEvent, FeedbackSignal
from research_pipeline.briefing.normalize import stable_hash, utc_now_iso

#: Allowed feedback target types (Literal in :class:`FeedbackEvent`).
ALLOWED_TARGET_TYPES: frozenset[str] = frozenset(
    get_args(FeedbackEvent.model_fields["target_type"].annotation)
)

SignalClass = Literal["positive", "negative", "neutral"]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS feedback_events (
    feedback_id TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL,
    target_type TEXT NOT NULL CHECK (target_type <> ''),
    target_id TEXT NOT NULL CHECK (target_id <> ''),
    signal_type TEXT NOT NULL,
    strength REAL NOT NULL CHECK (strength >= 0 AND strength <= 5),
    reason TEXT DEFAULT '',
    context_json TEXT DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_brief_feedback_target
ON feedback_events(target_type, target_id);

CREATE TABLE IF NOT EXISTS preference_adjustments (
    adjustment_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    target_type TEXT NOT NULL,
    target_id TEXT NOT NULL,
    before_weight REAL NOT NULL,
    after_weight REAL NOT NULL,
    trigger TEXT NOT NULL,
    rollback_json TEXT NOT NULL
);
"""

# Bump when _SCHEMA changes; drives PRAGMA user_version migration.
_SCHEMA_VERSION = 1

POSITIVE_SIGNALS = {
    FeedbackSignal.KEEP,
    FeedbackSignal.MORE_LIKE_THIS,
    FeedbackSignal.USEFUL,
}
NEGATIVE_SIGNALS = {
    FeedbackSignal.HIDE,
    FeedbackSignal.LESS_LIKE_THIS,
    FeedbackSignal.TOO_NOISY,
    FeedbackSignal.ALREADY_KNOWN,
    FeedbackSignal.NOT_ACTIONABLE,
    FeedbackSignal.NOT_USEFUL,
    FeedbackSignal.WRONG_CADENCE,
}

NEUTRAL_SIGNALS = {FeedbackSignal.NEUTRAL}

# Saturating bound on the per-target explicit-feedback weight, so repeated
# feedback on one target cannot compound into an ever-growing rank boost (#123).
_WEIGHT_CAP = 1.0


def classify_signal(signal: FeedbackSignal) -> SignalClass:
    """Classify a feedback signal as positive, negative, or neutral.

    A signal that is not registered in any of the three explicit sets is
    rejected as malformed; this guards against silently absorbing
    behavioral signals.
    """
    if signal in POSITIVE_SIGNALS:
        return "positive"
    if signal in NEGATIVE_SIGNALS:
        return "negative"
    if signal in NEUTRAL_SIGNALS:
        return "neutral"
    raise ValueError(f"unsupported feedback signal: {signal!r}")


def validate_feedback_input(
    *,
    target_type: str,
    target_id: str,
    signal: FeedbackSignal | str,
    strength: float = 1.0,
) -> tuple[str, FeedbackSignal, float]:
    """Validate explicit feedback inputs before they reach the store.

    Returns a ``(target_type, signal, strength)`` triple normalised into
    canonical types.  Raises :class:`ValueError` for malformed input so
    callers can surface a deterministic non-zero exit status.
    """
    if target_type not in ALLOWED_TARGET_TYPES:
        raise ValueError(
            f"unsupported feedback target_type: {target_type!r}; "
            f"allowed: {sorted(ALLOWED_TARGET_TYPES)}"
        )
    if not isinstance(target_id, str) or not target_id.strip():
        raise ValueError("feedback target_id must be a non-empty string")
    if "\n" in target_id or "\t" in target_id or " " in target_id.strip(" "):
        # Accept bare alphanumerics, dots, dashes, underscores, colons.
        # Reject embedded whitespace/newlines that would corrupt audit rows.
        raise ValueError(f"feedback target_id is malformed: {target_id!r}")
    if isinstance(signal, str):
        try:
            signal_obj = FeedbackSignal(signal)
        except ValueError as exc:
            raise ValueError(f"unsupported feedback signal: {signal!r}") from exc
    else:
        signal_obj = signal
    # Trigger classification so behavioral / unknown signals are rejected.
    classify_signal(signal_obj)
    if not (0.0 <= strength <= 5.0):
        raise ValueError(f"feedback strength out of range [0,5]: {strength!r}")
    return target_type, signal_obj, float(strength)


def is_conflicting(events: list[FeedbackEvent]) -> bool:
    """Return True if a target has both positive and negative feedback."""
    has_pos = any(event.signal_type in POSITIVE_SIGNALS for event in events)
    has_neg = any(event.signal_type in NEGATIVE_SIGNALS for event in events)
    return has_pos and has_neg


def feedback_target_key(target_type: str, target_id: str) -> str:
    """Return the canonical ``"<type>:<id>"`` key used by the store."""
    return f"{target_type}:{target_id}"


class BriefingFeedbackStore:
    """SQLite-backed store for briefing feedback and preference adjustments."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path))
        self._conn.row_factory = sqlite3.Row
        # Durability + referential integrity at the DB, not only in Python (#119):
        # WAL for concurrent-reader safety, and enforce any FK constraints.
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(_SCHEMA)
        # Track the schema version so future changes have a migration path (#119).
        current = self._conn.execute("PRAGMA user_version").fetchone()[0]
        if current == 0:
            self._conn.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}")
        self._conn.commit()

    def close(self) -> None:
        """Close the backing database."""
        self._conn.close()

    def record(
        self,
        *,
        target_type: str,
        target_id: str,
        signal: FeedbackSignal,
        strength: float = 1.0,
        reason: str = "",
        context: dict[str, str] | None = None,
    ) -> FeedbackEvent:
        """Record a feedback event."""
        target_type, signal, strength = validate_feedback_input(
            target_type=target_type,
            target_id=target_id,
            signal=signal,
            strength=strength,
        )
        timestamp = utc_now_iso()
        # Include a per-call uuid so repeated identical feedback in the same
        # second still produces a unique audit row; the hash remains
        # deterministic for any single record.
        nonce = uuid.uuid4().hex
        feedback = FeedbackEvent(
            feedback_id=stable_hash(
                timestamp,
                target_type,
                target_id,
                signal.value,
                reason,
                nonce,
                prefix="feedback_",
            ),
            timestamp=timestamp,
            target_type=target_type,  # type: ignore[arg-type]
            target_id=target_id,
            signal_type=signal,
            strength=strength,
            reason=reason,
            context=context or {},
        )
        self._conn.execute(
            """
            INSERT INTO feedback_events
                (feedback_id, timestamp, target_type, target_id, signal_type,
                 strength, reason, context_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                feedback.feedback_id,
                feedback.timestamp,
                feedback.target_type,
                feedback.target_id,
                feedback.signal_type.value,
                feedback.strength,
                feedback.reason,
                json.dumps(feedback.context, sort_keys=True),
            ),
        )
        self._conn.commit()
        return feedback

    def list_feedback(self, target_type: str | None = None) -> list[FeedbackEvent]:
        """List feedback events."""
        query = "SELECT * FROM feedback_events"
        params: list[str] = []
        if target_type is not None:
            query += " WHERE target_type = ?"
            params.append(target_type)
        query += " ORDER BY timestamp DESC"
        rows = self._conn.execute(query, params).fetchall()
        return [
            FeedbackEvent(
                feedback_id=row["feedback_id"],
                timestamp=row["timestamp"],
                target_type=row["target_type"],
                target_id=row["target_id"],
                signal_type=FeedbackSignal(row["signal_type"]),
                strength=float(row["strength"]),
                reason=row["reason"],
                context=json.loads(row["context_json"]),
            )
            for row in rows
        ]

    def conflict_summary(self) -> dict[str, dict[str, int]]:
        """Return targets with both positive and negative feedback."""
        summary: dict[str, dict[str, int]] = {}
        for feedback in self.list_feedback():
            key = f"{feedback.target_type}:{feedback.target_id}"
            counts = summary.setdefault(key, {"positive": 0, "negative": 0})
            if feedback.signal_type in POSITIVE_SIGNALS:
                counts["positive"] += 1
            elif feedback.signal_type in NEGATIVE_SIGNALS:
                counts["negative"] += 1
        return {
            key: value
            for key, value in summary.items()
            if value["positive"] > 0 and value["negative"] > 0
        }

    def weights_by_target(self) -> dict[str, float]:
        """Compute reversible explicit-feedback weights by target key.

        The per-target weight is clamped to ``[-_WEIGHT_CAP, _WEIGHT_CAP]`` so an
        early-liked target cannot accumulate an ever-growing boost as it keeps
        resurfacing and drawing more feedback — the echo-chamber loop the
        unbounded sum created (#123). Repeated feedback now *saturates* rather
        than compounds.
        """
        weights: dict[str, float] = {}
        for feedback in self.list_feedback():
            key = f"{feedback.target_type}:{feedback.target_id}"
            delta = 0.0
            if feedback.signal_type in POSITIVE_SIGNALS:
                delta = 0.25 * feedback.strength
            elif feedback.signal_type in NEGATIVE_SIGNALS:
                delta = -0.35 * feedback.strength
            weights[key] = weights.get(key, 0.0) + delta
        return {
            key: max(-_WEIGHT_CAP, min(_WEIGHT_CAP, value))
            for key, value in weights.items()
        }

    def create_adjustments(self, min_feedback: int = 3) -> list[dict[str, object]]:
        """Create auditable preference adjustments from explicit feedback."""
        feedback = self.list_feedback()
        if len(feedback) < min_feedback:
            return []
        weights = self.weights_by_target()
        conflicts = self.conflict_summary()
        adjustments: list[dict[str, object]] = []
        for key, after_weight in sorted(weights.items()):
            target_type, target_id = key.split(":", 1)
            # Unique per event: an audit row is append-only, so the id must not
            # collide when the same (target, weight) is recomputed later — a
            # deterministic hash + INSERT OR REPLACE silently destroyed the prior
            # row's created_at (#119). Mirror feedback.record()'s timestamp+nonce.
            created_at = utc_now_iso()
            nonce = uuid.uuid4().hex
            adjustment_id = stable_hash(
                key, after_weight, created_at, nonce, prefix="adjustment_"
            )
            row = {
                "adjustment_id": adjustment_id,
                "created_at": created_at,
                "target_type": target_type,
                "target_id": target_id,
                "before_weight": 0.0,
                "after_weight": after_weight,
                "trigger": f"{len(feedback)} explicit feedback events",
                "rollback": {"target": key, "restore_weight": 0.0},
                "conflict": conflicts.get(key, {}),
            }
            self._conn.execute(
                """
                INSERT INTO preference_adjustments
                    (adjustment_id, created_at, target_type, target_id,
                     before_weight, after_weight, trigger, rollback_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    adjustment_id,
                    row["created_at"],
                    target_type,
                    target_id,
                    0.0,
                    after_weight,
                    row["trigger"],
                    json.dumps(row["rollback"], sort_keys=True),
                ),
            )
            adjustments.append(row)
        self._conn.commit()
        return adjustments
