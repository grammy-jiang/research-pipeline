"""SQLite-backed topic memory for briefing fatigue and resurfacing."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable
from datetime import date, datetime
from pathlib import Path
from typing import Literal, cast

from research_pipeline.briefing.models import (
    BriefingCluster,
    TopicAliasSuggestion,
    TopicMemory,
)
from research_pipeline.briefing.normalize import (
    normalize_title,
    stable_hash,
    utc_now_iso,
)


def _encode_list(values: Iterable[str]) -> str:
    """Serialize a multivalued field as a JSON array (comma-safe, #119)."""
    return json.dumps(list(values))


def _decode_list(raw: str) -> tuple[str, ...]:
    """Parse a multivalued field, tolerating legacy comma-joined values (#119)."""
    if not raw:
        return ()
    try:
        data = json.loads(raw)
    except (ValueError, TypeError):
        data = None
    if isinstance(data, list):
        return tuple(str(v) for v in data)
    # Backward-compat: rows written before JSON encoding used comma-join.
    return tuple(filter(None, raw.split(",")))


_SCHEMA = """
CREATE TABLE IF NOT EXISTS topic_memory (
    topic_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    aliases TEXT DEFAULT '',
    first_seen_at TEXT NOT NULL,
    last_seen_at TEXT NOT NULL,
    status TEXT NOT NULL,
    summary TEXT DEFAULT '',
    key_entities TEXT DEFAULT '',
    canonical_clusters TEXT DEFAULT '',
    obsidian_note TEXT DEFAULT '',
    interest_score REAL DEFAULT 0.0,
    fatigue_score REAL DEFAULT 0.0,
    last_reported_at TEXT,
    report_count_7d INTEGER DEFAULT 0,
    report_count_30d INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS topic_memory_audit (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    topic_id TEXT NOT NULL,
    trigger TEXT NOT NULL,
    effect TEXT NOT NULL,
    rollback TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS topic_alias_suggestions (
    suggestion_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    topic_id TEXT NOT NULL,
    suggested_alias TEXT NOT NULL,
    reason TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    review_record TEXT DEFAULT ''
);
"""


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    return datetime.fromisoformat(value[:10]).date()


class TopicMemoryStore:
    """Durable topic memory store."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path))
        self._conn.row_factory = sqlite3.Row
        # Durability + referential integrity at the DB, not only in Python (#119).
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        """Close the backing database."""
        self._conn.close()

    def get(self, topic_id: str) -> TopicMemory | None:
        """Get a topic memory record."""
        row = self._conn.execute(
            "SELECT * FROM topic_memory WHERE topic_id = ?", (topic_id,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_memory(row)

    def list_memories(self) -> list[TopicMemory]:
        """List all topic memories in deterministic order."""
        rows = self._conn.execute(
            "SELECT * FROM topic_memory ORDER BY topic_id ASC"
        ).fetchall()
        return [self._row_to_memory(row) for row in rows]

    def upsert_from_clusters(
        self, clusters: list[BriefingCluster], run_date: str
    ) -> list[TopicMemory]:
        """Update topic memory from reported clusters."""
        updated: list[TopicMemory] = []
        for cluster in clusters:
            for topic_id in cluster.topic_ids:
                prior = self.get(topic_id)
                if prior is not None:
                    self.suggest_alias(topic_id, cluster.title)
                memory = self._next_memory(topic_id, cluster, run_date, prior)
                self._upsert(memory)
                self._audit(
                    run_date,
                    topic_id,
                    "cluster_reported",
                    f"status={memory.status}; fatigue={memory.fatigue_score:.2f}",
                    "restore previous topic_memory row from backup",
                )
                updated.append(memory)
        self._conn.commit()
        return updated

    def suggest_alias(
        self,
        topic_id: str,
        alias: str,
        reason: str = "cluster title differed from canonical topic name",
    ) -> TopicAliasSuggestion | None:
        """Persist a pending alias suggestion without applying it."""
        memory = self.get(topic_id)
        normalized = normalize_title(alias)
        if memory is None or normalized == normalize_title(memory.name):
            return None
        if normalized in {normalize_title(item) for item in memory.aliases}:
            return None
        suggestion = TopicAliasSuggestion(
            suggestion_id=stable_hash(topic_id, normalized, prefix="alias_suggestion_"),
            created_at=utc_now_iso(),
            topic_id=topic_id,
            suggested_alias=normalized,
            reason=reason,
        )
        self._conn.execute(
            """
            INSERT OR IGNORE INTO topic_alias_suggestions
                (suggestion_id, created_at, topic_id, suggested_alias,
                 reason, status, review_record)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                suggestion.suggestion_id,
                suggestion.created_at,
                suggestion.topic_id,
                suggestion.suggested_alias,
                suggestion.reason,
                suggestion.status,
                suggestion.review_record,
            ),
        )
        self._conn.commit()
        return suggestion

    def list_alias_suggestions(
        self, status: str | None = "pending"
    ) -> list[TopicAliasSuggestion]:
        """List reviewable alias suggestions."""
        query = "SELECT * FROM topic_alias_suggestions"
        params: list[str] = []
        if status is not None:
            query += " WHERE status = ?"
            params.append(status)
        query += " ORDER BY created_at DESC"
        rows = self._conn.execute(query, params).fetchall()
        return [
            TopicAliasSuggestion(
                suggestion_id=row["suggestion_id"],
                created_at=row["created_at"],
                topic_id=row["topic_id"],
                suggested_alias=row["suggested_alias"],
                reason=row["reason"],
                status=cast(
                    Literal["pending", "approved", "rejected"],
                    row["status"],
                ),
                review_record=row["review_record"],
            )
            for row in rows
        ]

    def review_alias_suggestion(
        self, suggestion_id: str, *, approve: bool, review_record: str
    ) -> TopicAliasSuggestion:
        """Approve or reject a pending alias suggestion."""
        if not review_record.strip():
            raise ValueError(
                "review_record is required once a suggestion is approved or rejected"
            )
        row = self._conn.execute(
            "SELECT * FROM topic_alias_suggestions WHERE suggestion_id = ?",
            (suggestion_id,),
        ).fetchone()
        if row is None:
            raise ValueError(f"alias suggestion not found: {suggestion_id}")
        status: Literal["approved", "rejected"] = "approved" if approve else "rejected"
        if approve:
            memory = self.get(str(row["topic_id"]))
            if memory is None:
                raise ValueError(f"topic not found: {row['topic_id']}")
            aliases = tuple(
                dict.fromkeys((*memory.aliases, str(row["suggested_alias"])))
            )
            self._upsert(memory.model_copy(update={"aliases": aliases}))
            self._audit(
                utc_now_iso(),
                memory.topic_id,
                "alias_approved",
                f"alias={row['suggested_alias']}",
                f"remove alias {row['suggested_alias']}",
            )
        self._conn.execute(
            """
            UPDATE topic_alias_suggestions
            SET status = ?, review_record = ?
            WHERE suggestion_id = ?
            """,
            (status, review_record, suggestion_id),
        )
        self._conn.commit()
        return TopicAliasSuggestion(
            suggestion_id=row["suggestion_id"],
            created_at=row["created_at"],
            topic_id=row["topic_id"],
            suggested_alias=row["suggested_alias"],
            reason=row["reason"],
            status=status,
            review_record=review_record,
        )

    def _next_memory(
        self,
        topic_id: str,
        cluster: BriefingCluster,
        run_date: str,
        prior: TopicMemory | None,
    ) -> TopicMemory:
        if prior is None:
            return TopicMemory(
                topic_id=topic_id,
                name=cluster.title,
                first_seen_at=run_date,
                last_seen_at=run_date,
                status="new",
                canonical_clusters=(cluster.cluster_id,),
                last_reported_at=run_date,
                report_count_7d=1,
                report_count_30d=1,
            )
        last = _parse_date(prior.last_reported_at)
        current = _parse_date(run_date)
        days = (current - last).days if last is not None and current is not None else 0
        status: Literal["active", "resurfaced"] = "active"
        resurfaced = days >= 14
        if resurfaced:
            status = "resurfaced"
        fatigue = min(3.0, prior.fatigue_score + (0.35 if days <= 2 else -0.2))
        fatigue = max(0.0, fatigue)
        clusters = tuple(dict.fromkeys((*prior.canonical_clusters, cluster.cluster_id)))
        return TopicMemory(
            topic_id=topic_id,
            name=prior.name,
            aliases=prior.aliases,
            first_seen_at=prior.first_seen_at,
            last_seen_at=run_date,
            status=status,
            summary=prior.summary,
            key_entities=prior.key_entities,
            canonical_clusters=clusters,
            obsidian_note=prior.obsidian_note,
            interest_score=prior.interest_score,
            fatigue_score=fatigue,
            last_reported_at=run_date,
            report_count_7d=prior.report_count_7d + (1 if days <= 7 else 0),
            report_count_30d=prior.report_count_30d + (1 if days <= 30 else 0),
        )

    def _upsert(self, memory: TopicMemory) -> None:
        self._conn.execute(
            """
            INSERT OR REPLACE INTO topic_memory
                (topic_id, name, aliases, first_seen_at, last_seen_at, status,
                 summary, key_entities, canonical_clusters, obsidian_note,
                 interest_score, fatigue_score, last_reported_at,
                 report_count_7d, report_count_30d)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                memory.topic_id,
                memory.name,
                _encode_list(memory.aliases),
                memory.first_seen_at,
                memory.last_seen_at,
                memory.status,
                memory.summary,
                _encode_list(memory.key_entities),
                _encode_list(memory.canonical_clusters),
                memory.obsidian_note,
                memory.interest_score,
                memory.fatigue_score,
                memory.last_reported_at,
                memory.report_count_7d,
                memory.report_count_30d,
            ),
        )

    def _audit(
        self, timestamp: str, topic_id: str, trigger: str, effect: str, rollback: str
    ) -> None:
        self._conn.execute(
            """
            INSERT INTO topic_memory_audit
                (timestamp, topic_id, trigger, effect, rollback)
            VALUES (?, ?, ?, ?, ?)
            """,
            (timestamp, topic_id, trigger, effect, rollback),
        )

    def _row_to_memory(self, row: sqlite3.Row) -> TopicMemory:
        return TopicMemory(
            topic_id=row["topic_id"],
            name=row["name"],
            aliases=_decode_list(row["aliases"]),
            first_seen_at=row["first_seen_at"],
            last_seen_at=row["last_seen_at"],
            status=row["status"],
            summary=row["summary"],
            key_entities=_decode_list(row["key_entities"]),
            canonical_clusters=_decode_list(row["canonical_clusters"]),
            obsidian_note=row["obsidian_note"],
            interest_score=float(row["interest_score"]),
            fatigue_score=float(row["fatigue_score"]),
            last_reported_at=row["last_reported_at"],
            report_count_7d=int(row["report_count_7d"]),
            report_count_30d=int(row["report_count_30d"]),
        )
