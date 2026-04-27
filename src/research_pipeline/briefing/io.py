"""Filesystem helpers for briefing artifacts."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any, overload

from pydantic import BaseModel


def ensure_parent(path: Path) -> None:
    """Create the parent directory for an artifact path."""
    path.parent.mkdir(parents=True, exist_ok=True)


def model_to_jsonable(value: BaseModel) -> dict[str, Any]:
    """Serialize a Pydantic model for JSON artifacts."""
    return value.model_dump(mode="json")


def write_json(path: Path, data: Any) -> None:
    """Write a JSON artifact with stable formatting."""
    ensure_parent(path)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    """Read a JSON artifact."""
    return json.loads(path.read_text(encoding="utf-8"))


def write_jsonl(path: Path, rows: Iterable[BaseModel | dict[str, Any]]) -> int:
    """Write JSONL rows and return the row count."""
    ensure_parent(path)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            data = model_to_jsonable(row) if isinstance(row, BaseModel) else row
            handle.write(json.dumps(data, sort_keys=True) + "\n")
            count += 1
    return count


@overload
def read_jsonl(path: Path) -> list[dict[str, Any]]: ...


@overload
def read_jsonl[T: BaseModel](path: Path, model: type[T]) -> list[T]: ...


def read_jsonl[T: BaseModel](
    path: Path, model: type[T] | None = None
) -> list[dict[str, Any]] | list[T]:
    """Read JSONL rows, optionally validating with a Pydantic model."""
    if not path.exists():
        return []
    rows: list[Any] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                data = json.loads(line)
                rows.append(model.model_validate(data) if model is not None else data)
    return rows
