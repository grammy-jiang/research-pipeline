"""Obsidian note models and YAML frontmatter helpers (Phase C ticket C02).

The generator emits three note types under the configured Obsidian vault
namespace:

* ``briefing-daily`` — one per run date
* ``briefing-topic`` — one per topic ID
* ``briefing-source`` — one per registered source ID

Every generated note carries a ``generated_id`` frontmatter key that downstream
exporters and validators use to enforce the Phase C ownership rule (see
``obsidian.is_owned_generated_note``). This module owns:

* the ``ObsidianNote`` Pydantic model
* a deterministic, intentionally restricted YAML frontmatter renderer
* a permissive, frontmatter-only parser used to read the leading YAML block
  back from existing notes for ownership checks

It does NOT touch the filesystem; later tickets (C03/C04) compose notes here
and write them via the safe path validators in ``obsidian.py``.
"""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

GENERATED_ID_KEY = "generated_id"
TYPE_KEY = "type"
RESERVED_FRONTMATTER_KEYS: frozenset[str] = frozenset({TYPE_KEY, GENERATED_ID_KEY})

NoteType = Literal["briefing-daily", "briefing-topic", "briefing-source"]
NOTE_TYPES: tuple[NoteType, ...] = (
    "briefing-daily",
    "briefing-topic",
    "briefing-source",
)

_GENERATED_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_FRONTMATTER_RE = re.compile(r"\A---\n(.*?)\n---\n?", re.DOTALL)


def render_frontmatter(fields: dict[str, Any]) -> str:
    """Render a dict as a deterministic YAML frontmatter block.

    Supported value types: ``str``, ``int``, ``float``, ``bool``, and
    ``list``/``tuple`` of those scalars. Keys are emitted in sorted order so
    the same input always produces the same output (idempotent writes).
    """
    if not fields:
        raise ValueError("cannot render empty frontmatter")
    lines: list[str] = ["---"]
    for key in sorted(fields):
        value = fields[key]
        lines.append(f"{key}: {_format_yaml_value(key, value)}")
    lines.append("---")
    return "\n".join(lines) + "\n"


def parse_frontmatter(content: str) -> tuple[dict[str, str], str]:
    """Extract the leading YAML frontmatter block and return ``(fields, body)``.

    Values are returned as strings; this parser is intentionally minimal — it
    only needs to support the deterministic shape produced by
    :func:`render_frontmatter` plus already-existing generator output. Lists
    are returned as their raw bracketed text so downstream code can decide how
    to compare them (ownership checks only need exact equality).
    """
    match = _FRONTMATTER_RE.match(content)
    if not match:
        raise ValueError("missing or malformed YAML frontmatter")
    block = match.group(1)
    fields: dict[str, str] = {}
    for line in block.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if ":" not in line:
            raise ValueError(f"invalid frontmatter line: {line!r}")
        key, _, raw = line.partition(":")
        fields[key.strip()] = raw.strip()
    body = content[match.end() :]
    return fields, body


class ObsidianNote(BaseModel):
    """One generated Obsidian note ready to be written to the vault."""

    model_config = ConfigDict(frozen=True)

    note_type: NoteType
    generated_id: str = Field(min_length=1)
    extra: dict[str, Any] = Field(default_factory=dict)
    body: str

    @model_validator(mode="after")
    def _check(self) -> ObsidianNote:
        if not _GENERATED_ID_RE.match(self.generated_id):
            raise ValueError(
                f"invalid generated_id (must be slug-like): {self.generated_id!r}"
            )
        for reserved in RESERVED_FRONTMATTER_KEYS:
            if reserved in self.extra:
                raise ValueError(
                    f"frontmatter key {reserved!r} is reserved and managed by "
                    "ObsidianNote"
                )
        # Daily briefs use the new icon-prefixed sections; topic/source notes
        # still emit the legacy "Agent Read Map" map. Either marker satisfies
        # the structural requirement.
        if "## Agent Read Map" not in self.body and "## ⭐ Top Items" not in self.body:
            raise ValueError(
                "note body must contain '## Agent Read Map' or '## ⭐ Top Items' "
                "section"
            )
        if "\n---\n" in self.body:
            raise ValueError("note body must not contain a YAML separator ('---') line")
        return self

    def frontmatter(self) -> dict[str, Any]:
        """Return the full frontmatter dict including reserved keys."""
        merged: dict[str, Any] = dict(self.extra)
        merged[TYPE_KEY] = self.note_type
        merged[GENERATED_ID_KEY] = self.generated_id
        return merged


def compose_note(note: ObsidianNote) -> str:
    """Return the full Markdown text (frontmatter + body) for ``note``."""
    fm = render_frontmatter(note.frontmatter())
    body = note.body
    if not body.startswith("\n"):
        fm = fm + "\n"
    return fm + body


def _format_yaml_value(key: str, value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        items: list[str] = []
        for entry in value:
            if not isinstance(entry, (str, int, float, bool)):
                raise ValueError(
                    f"unsupported list entry for {key!r}: {type(entry).__name__}"
                )
            if isinstance(entry, bool):
                items.append("true" if entry else "false")
            else:
                items.append(str(entry))
        return "[" + ", ".join(items) + "]"
    raise ValueError(
        f"unsupported frontmatter value type for {key!r}: {type(value).__name__}"
    )
