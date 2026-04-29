"""Daily Obsidian note export (Phase C ticket C03).

This module composes the human-readable daily briefing into an
:class:`ObsidianNote` and writes it to ``<vault>/<subdir>/Daily/<run_date>.md``
using the Phase C C01 path/ownership safety helpers.

Ownership rule: an existing note at the target path is overwritten only if
its frontmatter contains a ``generated_id`` matching the new note (i.e. it
was produced by a previous run of this generator). Human-authored notes
are refused.
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

_LEGACY_FRONTMATTER_RE = re.compile(r"\A---\n.*?\n---\n?", re.DOTALL)


def daily_note_path(config: ObsidianConfig, run_date: str) -> Path:
    """Return the canonical daily-note path inside the configured vault."""
    return config.namespace_root / "Daily" / f"{run_date}.md"


def build_daily_note(
    *,
    run_date: str,
    body: str,
    item_count: int,
    link_count: int,
    source_mix: dict[str, int] | None = None,
    extra: dict[str, Any] | None = None,
) -> ObsidianNote:
    """Build an :class:`ObsidianNote` for the given daily briefing body.

    A leading legacy ``---`` frontmatter block (e.g. one produced by
    :func:`research_pipeline.briefing.report.render_daily_brief`) is stripped
    so the renderer can re-emit Obsidian-shaped frontmatter via
    :class:`ObsidianNote`. Wiki-links inside the body are preserved verbatim.
    """
    cleaned = _LEGACY_FRONTMATTER_RE.sub("", body, count=1).lstrip("\n")
    frontmatter: dict[str, Any] = {
        "date": run_date,
        "item_count": int(item_count),
        "link_count": int(link_count),
    }
    if source_mix:
        frontmatter["source_mix"] = [
            f"{key}={source_mix[key]}" for key in sorted(source_mix)
        ]
    if extra:
        for key, value in extra.items():
            if key in frontmatter:
                continue
            frontmatter[key] = value
    return ObsidianNote(
        note_type="briefing-daily",
        generated_id=f"brief-{run_date}",
        extra=frontmatter,
        body=cleaned,
    )


def export_daily_note(note: ObsidianNote, config: ObsidianConfig) -> Path | None:
    """Write ``note`` to the daily-note path, honouring ownership and dry-run.

    Returns the resolved path on a real write, or ``None`` when
    ``config.dry_run`` is set.
    """
    if note.note_type != "briefing-daily":
        raise ValueError(
            f"export_daily_note expects briefing-daily, got {note.note_type!r}"
        )
    run_date = str(note.extra.get("date") or note.generated_id.removeprefix("brief-"))
    target = daily_note_path(config, run_date)
    resolved = validate_vault_path(target, config)
    if resolved.exists() and not is_owned_generated_note(resolved, note.generated_id):
        raise ValueError(
            "refusing to overwrite Obsidian note without matching "
            f"generated_id at {resolved}"
        )
    text = compose_note(note)
    if config.dry_run:
        return None
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(text, encoding="utf-8")
    return resolved
