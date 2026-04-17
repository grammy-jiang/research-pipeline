"""Episodic memory: SQLite-backed run history.

Stores summaries of past runs — what topic was researched, how many papers
were found, key decisions, outcomes, and lessons learned.  Used to inform
future runs (e.g., avoid re-searching the same topic, reuse good queries).

Supports **segment-level entries**: large synthesis summaries or key
decisions are automatically split into ≤450-token segments for retrieval
precision (per Memory Survey, PVLDB 2026 recommendation).
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

from research_pipeline.memory.segmentation import (
    DEFAULT_MAX_TOKENS,
    estimate_tokens,
    segment_text,
)

logger = logging.getLogger(__name__)

DEFAULT_EPISODIC_DIR = Path.home() / ".cache" / "research-pipeline"
DEFAULT_EPISODIC_PATH = DEFAULT_EPISODIC_DIR / "episodic_memory.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS episodes (
    run_id TEXT PRIMARY KEY,
    topic TEXT NOT NULL,
    profile TEXT DEFAULT 'standard',
    started_at TEXT NOT NULL,
    completed_at TEXT,
    stages_completed TEXT DEFAULT '[]',
    paper_count INTEGER DEFAULT 0,
    shortlist_count INTEGER DEFAULT 0,
    synthesis_summary TEXT DEFAULT '',
    gaps_found TEXT DEFAULT '[]',
    key_decisions TEXT DEFAULT '[]',
    outcome TEXT DEFAULT '',
    metadata TEXT DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_topic ON episodes(topic);
CREATE INDEX IF NOT EXISTS idx_started_at ON episodes(started_at);
"""


@dataclass
class Episode:
    """A record of one pipeline run."""

    run_id: str
    topic: str
    profile: str = "standard"
    started_at: str = ""
    completed_at: str = ""
    stages_completed: list[str] = field(default_factory=list)
    paper_count: int = 0
    shortlist_count: int = 0
    synthesis_summary: str = ""
    gaps_found: list[str] = field(default_factory=list)
    key_decisions: list[str] = field(default_factory=list)
    outcome: str = ""
    metadata: dict[str, str] = field(default_factory=dict)


class EpisodicMemory:
    """SQLite-backed episodic memory for run history."""

    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = db_path or DEFAULT_EPISODIC_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None
        self._ensure_schema()

    def _get_conn(self) -> sqlite3.Connection:
        """Get or create a database connection."""
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path))
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def _ensure_schema(self) -> None:
        """Create tables if they don't exist."""
        conn = self._get_conn()
        conn.executescript(_SCHEMA)
        conn.commit()

    def record_episode(self, episode: Episode) -> None:
        """Store or update an episode."""
        conn = self._get_conn()
        conn.execute(
            """INSERT OR REPLACE INTO episodes
               (run_id, topic, profile, started_at, completed_at,
                stages_completed, paper_count, shortlist_count,
                synthesis_summary, gaps_found, key_decisions, outcome, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                episode.run_id,
                episode.topic,
                episode.profile,
                episode.started_at,
                episode.completed_at,
                json.dumps(episode.stages_completed),
                episode.paper_count,
                episode.shortlist_count,
                episode.synthesis_summary,
                json.dumps(episode.gaps_found),
                json.dumps(episode.key_decisions),
                episode.outcome,
                json.dumps(episode.metadata),
            ),
        )
        conn.commit()

    def get_episode(self, run_id: str) -> Episode | None:
        """Get a specific episode by run_id."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM episodes WHERE run_id = ?", (run_id,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_episode(row)

    def search_by_topic(self, topic: str, limit: int = 10) -> list[Episode]:
        """Find past runs with similar topics (substring match)."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM episodes WHERE topic LIKE ?"
            " ORDER BY started_at DESC LIMIT ?",
            (f"%{topic}%", limit),
        ).fetchall()
        return [self._row_to_episode(row) for row in rows]

    def recent_episodes(self, limit: int = 10) -> list[Episode]:
        """Get most recent episodes."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM episodes ORDER BY started_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [self._row_to_episode(row) for row in rows]

    def count(self) -> int:
        """Total number of episodes."""
        conn = self._get_conn()
        row = conn.execute("SELECT COUNT(*) FROM episodes").fetchone()
        return row[0] if row else 0

    def _row_to_episode(self, row: sqlite3.Row) -> Episode:
        """Convert a database row to an Episode dataclass."""
        return Episode(
            run_id=row["run_id"],
            topic=row["topic"],
            profile=row["profile"] or "standard",
            started_at=row["started_at"],
            completed_at=row["completed_at"] or "",
            stages_completed=json.loads(row["stages_completed"] or "[]"),
            paper_count=row["paper_count"] or 0,
            shortlist_count=row["shortlist_count"] or 0,
            synthesis_summary=row["synthesis_summary"] or "",
            gaps_found=json.loads(row["gaps_found"] or "[]"),
            key_decisions=json.loads(row["key_decisions"] or "[]"),
            outcome=row["outcome"] or "",
            metadata=json.loads(row["metadata"] or "{}"),
        )

    def close(self) -> None:
        """Close the database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    # ------------------------------------------------------------------
    # Segment-level helpers
    # ------------------------------------------------------------------

    def get_segmented_summary(
        self,
        run_id: str,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> list[str]:
        """Return the synthesis summary for *run_id*, split into segments.

        Each segment is ≤ *max_tokens* estimated tokens.  If the summary
        already fits, a single-element list is returned.
        """
        episode = self.get_episode(run_id)
        if episode is None or not episode.synthesis_summary:
            return []
        return segment_text(episode.synthesis_summary, max_tokens=max_tokens)

    def get_segmented_decisions(
        self,
        run_id: str,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> list[str]:
        """Return key decisions for *run_id*, each capped to *max_tokens*.

        Decisions that exceed the limit are split; short ones pass through
        unchanged.
        """
        episode = self.get_episode(run_id)
        if episode is None:
            return []
        result: list[str] = []
        for decision in episode.key_decisions:
            if estimate_tokens(decision) > max_tokens:
                result.extend(segment_text(decision, max_tokens=max_tokens))
            else:
                result.append(decision)
        return result
