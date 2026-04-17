"""Knowledge graph with typed triples stored in SQLite.

Stores entities (papers, concepts, methods, experiments, claims, authors,
venues) and typed relations with provenance tracking.  Follows the same
SQLite pattern as :mod:`global_index`.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

DEFAULT_KG_DIR = Path.home() / ".cache" / "research-pipeline"
DEFAULT_KG_PATH = DEFAULT_KG_DIR / "knowledge_graph.db"


class EntityType(StrEnum):
    """Types of entities in the knowledge graph."""

    PAPER = "paper"
    CONCEPT = "concept"
    METHOD = "method"
    EXPERIMENT = "experiment"
    CLAIM = "claim"
    AUTHOR = "author"
    VENUE = "venue"


class RelationType(StrEnum):
    """Types of relations between entities."""

    CITES = "cites"
    USES_METHOD = "uses_method"
    PROPOSES_METHOD = "proposes_method"
    ADDRESSES_CONCEPT = "addresses_concept"
    EVALUATES_ON = "evaluates_on"
    RELATED_TO = "related_to"
    SUPPORTS_CLAIM = "supports_claim"
    CONTRADICTS_CLAIM = "contradicts_claim"
    AUTHORED_BY = "authored_by"
    PUBLISHED_IN = "published_in"


class Entity(BaseModel):
    """A node in the knowledge graph."""

    entity_id: str = Field(description="Unique entity identifier.")
    entity_type: EntityType = Field(description="Type of entity.")
    name: str = Field(description="Display name.")
    properties: dict[str, str] = Field(
        default_factory=dict, description="Additional properties."
    )


class Triple(BaseModel):
    """A typed relation (edge) between two entities."""

    subject_id: str = Field(description="Source entity ID.")
    relation: RelationType = Field(description="Relation type.")
    object_id: str = Field(description="Target entity ID.")
    provenance_paper: str = Field(
        default="", description="Paper that provided this relation."
    )
    provenance_run: str = Field(
        default="", description="Run ID where this was extracted."
    )
    confidence: float = Field(
        default=1.0, description="Confidence in this relation (0-1)."
    )
    created_at: str = Field(default="")


_SCHEMA = """
CREATE TABLE IF NOT EXISTS entities (
    entity_id TEXT PRIMARY KEY,
    entity_type TEXT NOT NULL,
    name TEXT NOT NULL,
    properties TEXT DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_entity_type ON entities(entity_type);
CREATE INDEX IF NOT EXISTS idx_entity_name ON entities(name);

CREATE TABLE IF NOT EXISTS triples (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    subject_id TEXT NOT NULL,
    relation TEXT NOT NULL,
    object_id TEXT NOT NULL,
    provenance_paper TEXT DEFAULT '',
    provenance_run TEXT DEFAULT '',
    confidence REAL DEFAULT 1.0,
    created_at TEXT NOT NULL,
    FOREIGN KEY (subject_id) REFERENCES entities(entity_id),
    FOREIGN KEY (object_id) REFERENCES entities(entity_id),
    UNIQUE(subject_id, relation, object_id, provenance_paper)
);
CREATE INDEX IF NOT EXISTS idx_triple_subject ON triples(subject_id);
CREATE INDEX IF NOT EXISTS idx_triple_object ON triples(object_id);
CREATE INDEX IF NOT EXISTS idx_triple_relation ON triples(relation);
CREATE INDEX IF NOT EXISTS idx_triple_provenance ON triples(provenance_paper);
"""


class KnowledgeGraph:
    """SQLite-backed knowledge graph with typed triples.

    Args:
        db_path: Path to SQLite database file.
                 None uses the default cache location.
    """

    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = db_path or DEFAULT_KG_PATH
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

    def add_entity(
        self,
        entity_id: str,
        entity_type: EntityType,
        name: str,
        properties: dict[str, str] | None = None,
    ) -> None:
        """Add or update an entity.

        Args:
            entity_id: Unique identifier for the entity.
            entity_type: Type classification.
            name: Human-readable display name.
            properties: Optional key-value metadata.
        """
        now = datetime.now(tz=UTC).isoformat()
        props_json = json.dumps(properties or {})
        conn = self._get_conn()
        conn.execute(
            """INSERT INTO entities
                   (entity_id, entity_type, name, properties, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?)
               ON CONFLICT(entity_id) DO UPDATE SET
                   name = excluded.name,
                   properties = excluded.properties,
                   updated_at = excluded.updated_at""",
            (entity_id, entity_type.value, name, props_json, now, now),
        )
        conn.commit()

    def add_triple(
        self,
        subject_id: str,
        relation: RelationType,
        object_id: str,
        provenance_paper: str = "",
        provenance_run: str = "",
        confidence: float = 1.0,
    ) -> None:
        """Add a typed relation between two entities.

        Args:
            subject_id: Source entity ID.
            relation: Relation type.
            object_id: Target entity ID.
            provenance_paper: Paper that provided this relation.
            provenance_run: Pipeline run ID.
            confidence: Confidence score (0-1).
        """
        now = datetime.now(tz=UTC).isoformat()
        conn = self._get_conn()
        try:
            conn.execute(
                """INSERT INTO triples
                       (subject_id, relation, object_id,
                        provenance_paper, provenance_run, confidence, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    subject_id,
                    relation.value,
                    object_id,
                    provenance_paper,
                    provenance_run,
                    confidence,
                    now,
                ),
            )
            conn.commit()
        except sqlite3.IntegrityError:
            logger.debug(
                "Triple already exists: %s -[%s]-> %s",
                subject_id,
                relation.value,
                object_id,
            )

    def get_entity(self, entity_id: str) -> Entity | None:
        """Get an entity by ID.

        Args:
            entity_id: Entity identifier.

        Returns:
            Entity if found, else None.
        """
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM entities WHERE entity_id = ?", (entity_id,)
        ).fetchone()
        if row is None:
            return None
        return Entity(
            entity_id=row["entity_id"],
            entity_type=EntityType(row["entity_type"]),
            name=row["name"],
            properties=json.loads(row["properties"]),
        )

    def get_triples(
        self,
        subject_id: str | None = None,
        relation: RelationType | None = None,
        object_id: str | None = None,
        provenance_paper: str | None = None,
    ) -> list[Triple]:
        """Query triples with optional filters.

        Args:
            subject_id: Filter by source entity.
            relation: Filter by relation type.
            object_id: Filter by target entity.
            provenance_paper: Filter by provenance paper.

        Returns:
            List of matching triples.
        """
        conditions: list[str] = []
        params: list[str | float] = []
        if subject_id:
            conditions.append("subject_id = ?")
            params.append(subject_id)
        if relation:
            conditions.append("relation = ?")
            params.append(relation.value)
        if object_id:
            conditions.append("object_id = ?")
            params.append(object_id)
        if provenance_paper:
            conditions.append("provenance_paper = ?")
            params.append(provenance_paper)

        where = " AND ".join(conditions) if conditions else "1=1"
        conn = self._get_conn()
        rows = conn.execute(
            f"SELECT * FROM triples WHERE {where}",  # nosec B608 - parameterized
            params,
        ).fetchall()
        return [self._row_to_triple(r) for r in rows]

    def get_neighbors(self, entity_id: str, direction: str = "both") -> list[Triple]:
        """Get all triples connected to an entity.

        Args:
            entity_id: The entity to query.
            direction: ``"outgoing"``, ``"incoming"``, or ``"both"``.

        Returns:
            List of connected triples.
        """
        conn = self._get_conn()
        results: list[sqlite3.Row] = []
        if direction in ("outgoing", "both"):
            rows = conn.execute(
                "SELECT * FROM triples WHERE subject_id = ?", (entity_id,)
            ).fetchall()
            results.extend(rows)
        if direction in ("incoming", "both"):
            rows = conn.execute(
                "SELECT * FROM triples WHERE object_id = ?", (entity_id,)
            ).fetchall()
            results.extend(rows)
        return [self._row_to_triple(r) for r in results]

    def get_entities_by_type(self, entity_type: EntityType) -> list[Entity]:
        """Get all entities of a given type.

        Args:
            entity_type: The entity type to filter by.

        Returns:
            List of matching entities.
        """
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM entities WHERE entity_type = ?",
            (entity_type.value,),
        ).fetchall()
        return [
            Entity(
                entity_id=r["entity_id"],
                entity_type=EntityType(r["entity_type"]),
                name=r["name"],
                properties=json.loads(r["properties"]),
            )
            for r in rows
        ]

    def count_entities(self) -> dict[str, int]:
        """Count entities grouped by type."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT entity_type, COUNT(*) as cnt FROM entities GROUP BY entity_type"
        ).fetchall()
        return {r["entity_type"]: r["cnt"] for r in rows}

    def count_triples(self) -> dict[str, int]:
        """Count triples grouped by relation type."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT relation, COUNT(*) as cnt FROM triples GROUP BY relation"
        ).fetchall()
        return {r["relation"]: r["cnt"] for r in rows}

    def stats(self) -> dict[str, object]:
        """Get summary statistics.

        Returns:
            Dict with entity/triple counts and totals.
        """
        entity_counts = self.count_entities()
        triple_counts = self.count_triples()
        return {
            "entities": entity_counts,
            "triples": triple_counts,
            "total_entities": sum(entity_counts.values()),
            "total_triples": sum(triple_counts.values()),
        }

    def ingest_from_candidates(self, candidates: list[object], run_id: str = "") -> int:
        """Ingest paper entities from CandidateRecord list.

        Args:
            candidates: List of CandidateRecord objects.
            run_id: Pipeline run ID.

        Returns:
            Number of paper entities added.
        """
        count = 0
        for c in candidates:
            paper_id = getattr(c, "arxiv_id", "") or getattr(c, "paper_id", "")
            if not paper_id:
                continue
            title = getattr(c, "title", paper_id)
            props: dict[str, str] = {}
            doi = getattr(c, "doi", None)
            if doi:
                props["doi"] = str(doi)
            published = getattr(c, "published", None)
            if published:
                props["published"] = str(published)

            self.add_entity(paper_id, EntityType.PAPER, title, props)

            # Add authors if available
            authors: list[str] = getattr(c, "authors", []) or []
            for author in authors:
                author_id = f"author:{author.lower().replace(' ', '_')}"
                self.add_entity(author_id, EntityType.AUTHOR, author)
                self.add_triple(
                    paper_id,
                    RelationType.AUTHORED_BY,
                    author_id,
                    provenance_run=run_id,
                )

            count += 1
        return count

    def ingest_from_claims(self, decomposition: object, run_id: str = "") -> int:
        """Ingest claim entities and relations from a ClaimDecomposition.

        Args:
            decomposition: ClaimDecomposition object from analysis/decomposer.
            run_id: Pipeline run ID.

        Returns:
            Number of triples added.
        """
        triple_count = 0
        paper_id: str = getattr(decomposition, "paper_id", "")
        title: str = getattr(decomposition, "title", paper_id)

        self.add_entity(paper_id, EntityType.PAPER, title)

        claims: list[object] = getattr(decomposition, "claims", [])
        for claim in claims:
            claim_id: str = getattr(claim, "claim_id", "")
            claim_entity_id = f"claim:{paper_id}:{claim_id}"
            statement: str = getattr(claim, "statement", "")
            source_type: str = getattr(claim, "source_type", "")
            evidence_class = getattr(claim, "evidence_class", None)
            evidence_class_value: str = evidence_class.value if evidence_class else ""
            confidence_score: float = getattr(claim, "confidence_score", 0.0)

            self.add_entity(
                claim_entity_id,
                EntityType.CLAIM,
                statement[:200],
                {
                    "source_type": source_type,
                    "evidence_class": evidence_class_value,
                },
            )

            rel = RelationType.SUPPORTS_CLAIM
            if evidence_class_value == "conflicting":
                rel = RelationType.CONTRADICTS_CLAIM

            self.add_triple(
                paper_id,
                rel,
                claim_entity_id,
                provenance_paper=paper_id,
                provenance_run=run_id,
                confidence=confidence_score,
            )
            triple_count += 1

        return triple_count

    def clear(self) -> None:
        """Clear all data from the knowledge graph."""
        conn = self._get_conn()
        conn.execute("DELETE FROM triples")
        conn.execute("DELETE FROM entities")
        conn.commit()

    def close(self) -> None:
        """Close the database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    @staticmethod
    def _row_to_triple(r: sqlite3.Row) -> Triple:
        """Convert a database row to a Triple model."""
        return Triple(
            subject_id=r["subject_id"],
            relation=RelationType(r["relation"]),
            object_id=r["object_id"],
            provenance_paper=r["provenance_paper"],
            provenance_run=r["provenance_run"],
            confidence=r["confidence"],
            created_at=r["created_at"],
        )
