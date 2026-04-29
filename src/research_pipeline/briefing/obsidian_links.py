"""Wiki-link, backlink, and idempotent-update helpers (Phase C ticket C05).

Pure-string utilities used by daily/topic/source exporters. **Never**
rewrite an existing ``[[wiki-link]]`` to a Markdown link — Obsidian wiki
syntax is preserved verbatim.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

_SLUG_REPLACE_RE = re.compile(r"[^A-Za-z0-9._-]+")
_SLUG_TRIM_RE = re.compile(r"(^[._-]+|[._-]+$)")
_WIKILINK_RE = re.compile(r"\[\[([^\[\]\|]+)(?:\|([^\[\]]+))?\]\]")
_BACKLINKS_HEADING = "## Backlinks"


def slugify(name: str) -> str:
    """Return a slug-safe identifier for ``name``.

    Whitespace and unsupported characters collapse to ``-``. Leading and
    trailing punctuation is stripped. ``ValueError`` is raised when the
    result would be empty.
    """
    if not isinstance(name, str):
        raise TypeError(f"slugify expects str, got {type(name).__name__}")
    collapsed = _SLUG_REPLACE_RE.sub("-", name.strip())
    trimmed = _SLUG_TRIM_RE.sub("", collapsed)
    if not trimmed:
        raise ValueError(f"slugify produced empty slug for {name!r}")
    return trimmed


def make_wikilink(target: str, alias: str | None = None) -> str:
    """Return an Obsidian ``[[wiki-link]]`` (optionally aliased)."""
    target = target.strip()
    if not target:
        raise ValueError("wiki-link target cannot be empty")
    if "[" in target or "]" in target or "|" in target:
        raise ValueError(f"invalid wiki-link target: {target!r}")
    if alias is not None:
        alias = alias.strip()
        if not alias:
            raise ValueError("wiki-link alias cannot be blank when supplied")
        if "[" in alias or "]" in alias or "|" in alias:
            raise ValueError(f"invalid wiki-link alias: {alias!r}")
        return f"[[{target}|{alias}]]"
    return f"[[{target}]]"


def find_wikilinks(text: str) -> list[tuple[str, str | None]]:
    """Return every ``[[target]]`` or ``[[target|alias]]`` found in ``text``."""
    out: list[tuple[str, str | None]] = []
    for match in _WIKILINK_RE.finditer(text):
        target = match.group(1).strip()
        alias = match.group(2)
        if alias is not None:
            alias = alias.strip() or None
        out.append((target, alias))
    return out


def inject_backlinks(body: str, links: Iterable[str]) -> str:
    """Append (or replace) a deterministic ``## Backlinks`` section.

    Existing wiki-links inside ``body`` are preserved exactly. ``links`` is
    deduplicated and sorted to keep re-runs idempotent. Each entry is
    rendered with :func:`make_wikilink` if it does not already contain
    ``[[ ]]`` syntax.
    """
    cleaned: list[str] = []
    for link in links:
        token = link.strip()
        if not token:
            continue
        if not (token.startswith("[[") and token.endswith("]]")):
            token = make_wikilink(token)
        cleaned.append(token)
    if not cleaned:
        return _strip_existing_backlinks(body)
    deduped = sorted(set(cleaned))
    block = (
        _BACKLINKS_HEADING
        + "\n\n"
        + "\n".join(f"- {entry}" for entry in deduped)
        + "\n"
    )
    stripped = _strip_existing_backlinks(body)
    if stripped.endswith("\n\n"):
        return stripped + block
    if stripped.endswith("\n"):
        return stripped + "\n" + block
    return stripped + "\n\n" + block


def _strip_existing_backlinks(body: str) -> str:
    idx = body.find(_BACKLINKS_HEADING)
    if idx < 0:
        return body
    return body[:idx].rstrip() + "\n"


def is_idempotent_update(old_text: str, new_text: str) -> bool:
    """Return ``True`` when ``new_text`` is byte-equivalent to ``old_text``.

    Used by exporters to skip writes when nothing has changed and to
    detect regressions where wiki-link syntax has been mutated.
    """
    return old_text == new_text
