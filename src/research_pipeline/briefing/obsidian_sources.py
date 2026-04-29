"""Source Obsidian note export (Phase C ticket C04).

Source notes live at ``<vault>/<subdir>/Sources/<source_id>.md`` and capture
the registry metadata (name, class, weight) so the generator's archive is
self-describing without external state. Wiki-links in the body are
preserved; link construction happens in ticket C05.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from research_pipeline.briefing.obsidian import (
    ObsidianConfig,
    is_owned_generated_note,
    validate_vault_path,
)
from research_pipeline.briefing.obsidian_notes import (
    ObsidianNote,
    compose_note,
)

_SLUG_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def source_note_path(config: ObsidianConfig, source_id: str) -> Path:
    """Return the canonical path for a source note in the configured vault."""
    if not _SLUG_RE.match(source_id):
        raise ValueError(f"invalid source_id (must be slug-like): {source_id!r}")
    return config.namespace_root / "Sources" / f"{source_id}.md"


def build_source_note(
    *,
    source_id: str,
    name: str,
    body: str,
    source_class: str | None = None,
    weight: float | None = None,
    extra: dict[str, Any] | None = None,
) -> ObsidianNote:
    """Build a source :class:`ObsidianNote` from registry metadata."""
    if not name.strip():
        raise ValueError("source name is required")
    fm: dict[str, Any] = {"source_id": source_id, "name": name}
    if source_class:
        fm["source_class"] = source_class
    if weight is not None:
        fm["weight"] = float(weight)
    if extra:
        for key, value in extra.items():
            if key in fm:
                continue
            fm[key] = value
    return ObsidianNote(
        note_type="briefing-source",
        generated_id=f"source-{source_id}",
        extra=fm,
        body=body,
    )


def export_source_note(note: ObsidianNote, config: ObsidianConfig) -> Path | None:
    """Write ``note`` to the sources directory, honouring ownership/dry-run."""
    if note.note_type != "briefing-source":
        raise ValueError(
            f"export_source_note expects briefing-source, got {note.note_type!r}"
        )
    source_id = str(note.extra.get("source_id") or "")
    if not source_id:
        raise ValueError("source note frontmatter is missing source_id")
    target = source_note_path(config, source_id)
    resolved = validate_vault_path(target, config)
    if resolved.exists() and not is_owned_generated_note(resolved, note.generated_id):
        raise ValueError(
            "refusing to overwrite Obsidian source note without matching "
            f"generated_id at {resolved}"
        )
    text = compose_note(note)
    if config.dry_run:
        return None
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(text, encoding="utf-8")
    return resolved
