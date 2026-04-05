"""Global paper index for incremental runs.

Stores metadata about papers processed across runs in a SQLite
database, enabling dedup and artifact reuse.
"""

import logging
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_INDEX_DIR = Path.home() / ".cache" / "research-pipeline"
DEFAULT_INDEX_PATH = DEFAULT_INDEX_DIR / "paper_index.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS papers (
    arxiv_id TEXT,
    doi TEXT,
    s2_id TEXT,
    title TEXT,
    run_id TEXT NOT NULL,
    stage TEXT NOT NULL,
    pdf_path TEXT,
    markdown_path TEXT,
    summary_path TEXT,
    pdf_sha256 TEXT,
    indexed_at TEXT NOT NULL,
    PRIMARY KEY (arxiv_id, run_id)
);
CREATE INDEX IF NOT EXISTS idx_doi ON papers(doi);
CREATE INDEX IF NOT EXISTS idx_s2_id ON papers(s2_id);
CREATE INDEX IF NOT EXISTS idx_title ON papers(title);
"""


class GlobalPaperIndex:
    """SQLite-backed global index of processed papers."""

    def __init__(self, db_path: Path | None = None) -> None:
        """Initialize the global paper index.

        Args:
            db_path: Path to SQLite database file.
                     None uses the default cache location.
        """
        self.db_path = db_path or DEFAULT_INDEX_PATH
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

    def register_paper(
        self,
        arxiv_id: str,
        run_id: str,
        stage: str,
        doi: str | None = None,
        s2_id: str | None = None,
        title: str | None = None,
        pdf_path: str | None = None,
        markdown_path: str | None = None,
        summary_path: str | None = None,
        pdf_sha256: str | None = None,
    ) -> None:
        """Register or update a paper in the index.

        Args:
            arxiv_id: arXiv identifier.
            run_id: Pipeline run ID.
            stage: Pipeline stage that produced this entry.
            doi: Digital Object Identifier.
            s2_id: Semantic Scholar paper ID.
            title: Paper title.
            pdf_path: Path to downloaded PDF.
            markdown_path: Path to converted markdown.
            summary_path: Path to summary file.
            pdf_sha256: SHA-256 hash of the PDF.
        """
        conn = self._get_conn()
        conn.execute(
            """
            INSERT OR REPLACE INTO papers
                (arxiv_id, doi, s2_id, title, run_id, stage,
                 pdf_path, markdown_path, summary_path, pdf_sha256, indexed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                arxiv_id,
                doi,
                s2_id,
                title,
                run_id,
                stage,
                pdf_path,
                markdown_path,
                summary_path,
                pdf_sha256,
                datetime.now(UTC).isoformat(),
            ),
        )
        conn.commit()

    def is_known(self, arxiv_id: str) -> bool:
        """Check if a paper has been processed in any previous run.

        Args:
            arxiv_id: arXiv identifier to check.

        Returns:
            True if the paper exists in the index.
        """
        conn = self._get_conn()
        row = conn.execute(
            "SELECT 1 FROM papers WHERE arxiv_id = ? LIMIT 1",
            (arxiv_id,),
        ).fetchone()
        return row is not None

    def find_known_ids(self, arxiv_ids: list[str]) -> set[str]:
        """Find which of the given IDs are already in the index.

        Args:
            arxiv_ids: List of arXiv IDs to check.

        Returns:
            Set of IDs that exist in the index.
        """
        if not arxiv_ids:
            return set()

        conn = self._get_conn()
        placeholders = ",".join("?" for _ in arxiv_ids)
        rows = conn.execute(
            f"SELECT DISTINCT arxiv_id FROM papers WHERE arxiv_id IN ({placeholders})",  # nosec B608 - parameterized query
            arxiv_ids,
        ).fetchall()
        return {row["arxiv_id"] for row in rows}

    def find_artifact(
        self,
        arxiv_id: str,
        stage: str,
    ) -> dict[str, str | None] | None:
        """Find an existing artifact for a paper.

        Args:
            arxiv_id: arXiv identifier.
            stage: Pipeline stage to look up.

        Returns:
            Dict with pdf_path, markdown_path, summary_path, pdf_sha256;
            or None if not found.
        """
        conn = self._get_conn()
        row = conn.execute(
            """
            SELECT pdf_path, markdown_path, summary_path, pdf_sha256, run_id
            FROM papers WHERE arxiv_id = ? AND stage = ?
            ORDER BY indexed_at DESC LIMIT 1
            """,
            (arxiv_id, stage),
        ).fetchone()

        if row is None:
            return None

        return {
            "pdf_path": row["pdf_path"],
            "markdown_path": row["markdown_path"],
            "summary_path": row["summary_path"],
            "pdf_sha256": row["pdf_sha256"],
            "run_id": row["run_id"],
        }

    def list_papers(self, limit: int = 100) -> list[dict[str, str | None]]:
        """List papers in the index.

        Args:
            limit: Maximum number of results.

        Returns:
            List of paper dicts.
        """
        conn = self._get_conn()
        rows = conn.execute(
            """
            SELECT arxiv_id, doi, title, run_id, stage, indexed_at
            FROM papers ORDER BY indexed_at DESC LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [dict(row) for row in rows]

    def garbage_collect(self) -> int:
        """Remove entries whose referenced files no longer exist.

        Returns:
            Number of entries removed.
        """
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT arxiv_id, run_id, pdf_path, markdown_path, summary_path FROM papers"
        ).fetchall()

        stale = []
        for row in rows:
            paths = [row["pdf_path"], row["markdown_path"], row["summary_path"]]
            existing = [p for p in paths if p and Path(p).exists()]
            if not existing and any(p for p in paths if p):
                stale.append((row["arxiv_id"], row["run_id"]))

        for arxiv_id, run_id in stale:
            conn.execute(
                "DELETE FROM papers WHERE arxiv_id = ? AND run_id = ?",
                (arxiv_id, run_id),
            )
        conn.commit()

        if stale:
            logger.info("Garbage collected %d stale index entries", len(stale))
        return len(stale)

    def close(self) -> None:
        """Close the database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None
