"""Explicit feedback store and reversible preference adjustments."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from research_pipeline.briefing.models import FeedbackEvent, FeedbackSignal
from research_pipeline.briefing.normalize import stable_hash, utc_now_iso

_SCHEMA = """
CREATE TABLE IF NOT EXISTS feedback_events (
    feedback_id TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL,
    target_type TEXT NOT NULL,
    target_id TEXT NOT NULL,
    signal_type TEXT NOT NULL,
    strength REAL NOT NULL,
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


class BriefingFeedbackStore:
    """SQLite-backed store for briefing feedback and preference adjustments."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
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
        timestamp = utc_now_iso()
        feedback = FeedbackEvent(
            feedback_id=stable_hash(
                timestamp,
                target_type,
                target_id,
                signal.value,
                reason,
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
        """Compute reversible explicit-feedback weights by target key."""
        weights: dict[str, float] = {}
        for feedback in self.list_feedback():
            key = f"{feedback.target_type}:{feedback.target_id}"
            delta = 0.0
            if feedback.signal_type in POSITIVE_SIGNALS:
                delta = 0.25 * feedback.strength
            elif feedback.signal_type in NEGATIVE_SIGNALS:
                delta = -0.35 * feedback.strength
            weights[key] = weights.get(key, 0.0) + delta
        return weights

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
            adjustment_id = stable_hash(key, after_weight, prefix="adjustment_")
            row = {
                "adjustment_id": adjustment_id,
                "created_at": utc_now_iso(),
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
                INSERT OR REPLACE INTO preference_adjustments
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
