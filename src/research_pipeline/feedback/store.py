"""SQLite-backed feedback store and ELO-style weight adjuster.

Stores user accept/reject decisions on screened papers and uses
accumulated feedback to adjust BM25 scoring weights via an
ELO-inspired gradient approach.

Reference: Deep Research Report §E6 (User Feedback Loop).
"""

import json
import logging
import math
import sqlite3
from pathlib import Path

from research_pipeline.feedback.models import (
    FeedbackDecision,
    FeedbackRecord,
    WeightAdjustment,
)

logger = logging.getLogger(__name__)

DEFAULT_FEEDBACK_DIR = Path.home() / ".cache" / "research-pipeline"
DEFAULT_FEEDBACK_PATH = DEFAULT_FEEDBACK_DIR / "feedback.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS feedback (
    paper_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    decision TEXT NOT NULL,
    reason TEXT DEFAULT '',
    recorded_at TEXT NOT NULL,
    cheap_score REAL DEFAULT 0.0,
    PRIMARY KEY (paper_id, run_id)
);
CREATE INDEX IF NOT EXISTS idx_feedback_run ON feedback(run_id);
CREATE INDEX IF NOT EXISTS idx_feedback_decision ON feedback(decision);

CREATE TABLE IF NOT EXISTS adjusted_weights (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    weights_json TEXT NOT NULL,
    feedback_count INTEGER DEFAULT 0,
    created_at TEXT NOT NULL
);
"""

# Default BM25 weights (non-semantic mode)
_DEFAULT_WEIGHTS: dict[str, float] = {
    "bm25_must_title": 0.20,
    "bm25_nice_title": 0.10,
    "bm25_must_abstract": 0.25,
    "bm25_nice_abstract": 0.10,
    "cat_match": 0.15,
    "negative_penalty": 0.10,
    "recency_bonus": 0.10,
}


class FeedbackStore:
    """SQLite-backed store for user screening feedback.

    Stores accept/reject decisions and computes adjusted BM25 weights
    using an ELO-inspired approach: weights that correlate with
    accepted papers are boosted; those correlating with rejected
    papers are dampened.
    """

    def __init__(self, db_path: Path | None = None) -> None:
        self._db_path = db_path or DEFAULT_FEEDBACK_PATH
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()

    def record(self, feedback: FeedbackRecord) -> None:
        """Record a single feedback entry (upsert)."""
        self._conn.execute(
            """
            INSERT INTO feedback
                (paper_id, run_id, decision, reason,
                 recorded_at, cheap_score)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(paper_id, run_id) DO UPDATE SET
                decision = excluded.decision,
                reason = excluded.reason,
                recorded_at = excluded.recorded_at,
                cheap_score = excluded.cheap_score
            """,
            (
                feedback.paper_id,
                feedback.run_id,
                feedback.decision.value,
                feedback.reason,
                feedback.recorded_at,
                feedback.cheap_score,
            ),
        )
        self._conn.commit()
        logger.debug(
            "Recorded feedback: %s -> %s", feedback.paper_id, feedback.decision.value
        )

    def record_batch(self, records: list[FeedbackRecord]) -> int:
        """Record multiple feedback entries. Returns count recorded."""
        for rec in records:
            self.record(rec)
        return len(records)

    def get_feedback(
        self,
        run_id: str | None = None,
        decision: FeedbackDecision | None = None,
    ) -> list[FeedbackRecord]:
        """Retrieve feedback records, optionally filtered."""
        query = "SELECT * FROM feedback WHERE 1=1"
        params: list[str] = []
        if run_id is not None:
            query += " AND run_id = ?"
            params.append(run_id)
        if decision is not None:
            query += " AND decision = ?"
            params.append(decision.value)
        query += " ORDER BY recorded_at DESC"

        rows = self._conn.execute(query, params).fetchall()
        return [
            FeedbackRecord(
                paper_id=row["paper_id"],
                run_id=row["run_id"],
                decision=FeedbackDecision(row["decision"]),
                reason=row["reason"],
                recorded_at=row["recorded_at"],
                cheap_score=row["cheap_score"],
            )
            for row in rows
        ]

    def count(self, run_id: str | None = None) -> dict[str, int]:
        """Count feedback by decision type."""
        query = "SELECT decision, COUNT(*) as cnt FROM feedback"
        params: list[str] = []
        if run_id is not None:
            query += " WHERE run_id = ?"
            params.append(run_id)
        query += " GROUP BY decision"

        rows = self._conn.execute(query, params).fetchall()
        result = {"accept": 0, "reject": 0, "total": 0}
        for row in rows:
            result[row["decision"]] = row["cnt"]
            result["total"] += row["cnt"]
        return result

    def compute_adjusted_weights(
        self,
        learning_rate: float = 0.05,
        min_feedback: int = 5,
    ) -> WeightAdjustment:
        """Compute ELO-style adjusted BM25 weights from accumulated feedback.

        The algorithm compares mean cheap_score between accepted and
        rejected papers. For each weight component, the gradient is
        proportional to the score difference: weights are nudged up if
        they correlate with acceptance and down if rejection.

        Args:
            learning_rate: K-factor for weight adjustment (0.01-0.20).
            min_feedback: Minimum total feedback before adjusting.

        Returns:
            WeightAdjustment with adjusted weights.
        """
        all_feedback = self.get_feedback()
        if len(all_feedback) < min_feedback:
            logger.info(
                "Insufficient feedback (%d < %d), returning defaults",
                len(all_feedback),
                min_feedback,
            )
            return WeightAdjustment(
                feedback_count=len(all_feedback),
                learning_rate=learning_rate,
            )

        accepted = [f for f in all_feedback if f.decision == FeedbackDecision.ACCEPT]
        rejected = [f for f in all_feedback if f.decision == FeedbackDecision.REJECT]

        if not accepted or not rejected:
            logger.info(
                "Need both accept and reject feedback (got %d/%d)",
                len(accepted),
                len(rejected),
            )
            return WeightAdjustment(
                feedback_count=len(all_feedback),
                learning_rate=learning_rate,
            )

        # Compute mean cheap_score for each group
        mean_accepted = sum(f.cheap_score for f in accepted) / len(accepted)
        mean_rejected = sum(f.cheap_score for f in rejected) / len(rejected)

        # Score gap: positive means accepted papers score higher (good)
        score_gap = mean_accepted - mean_rejected

        # ELO-style: expected outcome vs actual
        # If gap is small, weights need more adjustment
        expected = 1.0 / (1.0 + math.exp(-10.0 * score_gap))
        actual = len(accepted) / len(all_feedback)
        adjustment = learning_rate * (actual - expected)

        # Apply adjustment proportionally to each weight
        weights = dict(_DEFAULT_WEIGHTS)
        # Boost precision weights (must_terms) when users reject many papers
        precision_boost = 1.0 + adjustment
        recall_boost = 1.0 - adjustment * 0.5

        weights["bm25_must_title"] *= precision_boost
        weights["bm25_must_abstract"] *= precision_boost
        weights["bm25_nice_title"] *= recall_boost
        weights["bm25_nice_abstract"] *= recall_boost

        # Increase negative penalty if many rejections
        reject_ratio = len(rejected) / len(all_feedback)
        if reject_ratio > 0.5:
            weights["negative_penalty"] *= 1.0 + learning_rate * (reject_ratio - 0.5)

        # Normalize so weights sum to ~1.0
        total = sum(weights.values())
        if total > 0:
            for key in weights:
                weights[key] = round(weights[key] / total, 4)

        adjusted = WeightAdjustment(
            bm25_must_title=weights["bm25_must_title"],
            bm25_nice_title=weights["bm25_nice_title"],
            bm25_must_abstract=weights["bm25_must_abstract"],
            bm25_nice_abstract=weights["bm25_nice_abstract"],
            cat_match=weights["cat_match"],
            negative_penalty=weights["negative_penalty"],
            recency_bonus=weights["recency_bonus"],
            feedback_count=len(all_feedback),
            learning_rate=learning_rate,
        )

        # Persist the adjusted weights
        self._conn.execute(
            "INSERT INTO adjusted_weights "
            "(weights_json, feedback_count, created_at) "
            "VALUES (?, ?, ?)",
            (
                json.dumps(adjusted.to_weight_dict()),
                len(all_feedback),
                all_feedback[0].recorded_at,
            ),
        )
        self._conn.commit()
        logger.info(
            "Computed adjusted weights from %d feedback records (accept=%d, reject=%d)",
            len(all_feedback),
            len(accepted),
            len(rejected),
        )
        return adjusted

    def get_latest_weights(self) -> WeightAdjustment | None:
        """Retrieve the most recently computed adjusted weights."""
        row = self._conn.execute(
            "SELECT weights_json, feedback_count "
            "FROM adjusted_weights ORDER BY id DESC LIMIT 1"
        ).fetchone()
        if row is None:
            return None
        weights = json.loads(row["weights_json"])
        return WeightAdjustment(
            feedback_count=row["feedback_count"],
            **weights,
        )
