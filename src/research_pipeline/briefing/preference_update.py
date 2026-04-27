"""Reviewable preference update and rollback helpers."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path


def rollback_preference_adjustment(
    db_path: Path, adjustment_id: str
) -> dict[str, object]:
    """Rollback a stored briefing preference adjustment by deleting it."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute(
            "SELECT * FROM preference_adjustments WHERE adjustment_id = ?",
            (adjustment_id,),
        ).fetchone()
        if row is None:
            raise ValueError(f"preference adjustment not found: {adjustment_id}")
        rollback = json.loads(row["rollback_json"])
        conn.execute(
            "DELETE FROM preference_adjustments WHERE adjustment_id = ?",
            (adjustment_id,),
        )
        conn.commit()
        return {
            "adjustment_id": adjustment_id,
            "rolled_back": True,
            "rollback": rollback,
        }
    finally:
        conn.close()
