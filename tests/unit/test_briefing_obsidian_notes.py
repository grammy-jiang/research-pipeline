"""Unit tests for Phase C C02 Obsidian note models and frontmatter."""

from __future__ import annotations

import pytest

from research_pipeline.briefing.obsidian_notes import (
    ObsidianNote,
    compose_note,
    parse_frontmatter,
    render_frontmatter,
)


def test_render_frontmatter_emits_sorted_yaml() -> None:
    text = render_frontmatter({"type": "briefing-daily", "generated_id": "abc"})
    assert text.startswith("---\n")
    assert text.endswith("---\n")
    body = text.strip().splitlines()
    assert body[0] == "---"
    assert body[-1] == "---"
    keys = [line.split(":")[0] for line in body[1:-1]]
    assert keys == sorted(keys)


def test_render_frontmatter_supports_scalar_and_list_values() -> None:
    text = render_frontmatter(
        {
            "type": "briefing-topic",
            "generated_id": "topic_x",
            "aliases": ["a", "b"],
            "count": 3,
            "active": True,
        }
    )
    assert "active: true" in text
    assert "aliases: [a, b]" in text
    assert "count: 3" in text


def test_render_frontmatter_rejects_unsupported_value_type() -> None:
    with pytest.raises(ValueError, match="unsupported"):
        render_frontmatter({"obj": {"nested": 1}})


def test_render_frontmatter_rejects_empty() -> None:
    with pytest.raises(ValueError, match="empty"):
        render_frontmatter({})


def test_parse_frontmatter_round_trip() -> None:
    text = render_frontmatter(
        {"type": "briefing-source", "generated_id": "source_x", "active": True}
    )
    full = text + "\nbody here\n"
    fm, body = parse_frontmatter(full)
    assert fm["type"] == "briefing-source"
    assert fm["generated_id"] == "source_x"
    assert fm["active"] == "true"
    assert body.lstrip("\n") == "body here\n"


def test_parse_frontmatter_rejects_missing_block() -> None:
    with pytest.raises(ValueError, match="frontmatter"):
        parse_frontmatter("# heading\nbody")


def test_obsidian_note_requires_agent_read_map() -> None:
    with pytest.raises(ValueError, match="Agent Read Map"):
        ObsidianNote(
            note_type="briefing-daily",
            generated_id="brief-2026-04-29",
            extra={},
            body="# Daily\n\nno agent read map here\n",
        )


def test_obsidian_note_rejects_extra_with_reserved_key() -> None:
    with pytest.raises(ValueError, match="reserved"):
        ObsidianNote(
            note_type="briefing-daily",
            generated_id="brief-1",
            extra={"type": "x"},
            body="# h\n\n## Agent Read Map\nx\n",
        )


def test_obsidian_note_rejects_invalid_generated_id() -> None:
    with pytest.raises(ValueError, match="generated_id"):
        ObsidianNote(
            note_type="briefing-daily",
            generated_id="bad id with spaces",
            extra={},
            body="## Agent Read Map\nx\n",
        )


def test_obsidian_note_rejects_invalid_type() -> None:
    with pytest.raises(ValueError):
        ObsidianNote(
            note_type="briefing-evil",  # type: ignore[arg-type]
            generated_id="x",
            extra={},
            body="## Agent Read Map\nx\n",
        )


def test_compose_note_emits_owned_frontmatter() -> None:
    note = ObsidianNote(
        note_type="briefing-daily",
        generated_id="brief-2026-04-29",
        extra={"date": "2026-04-29", "item_count": 5},
        body="# Daily\n\n## Agent Read Map\n\n- a\n",
    )
    text = compose_note(note)
    assert text.startswith("---\n")
    assert "type: briefing-daily" in text
    assert "generated_id: brief-2026-04-29" in text
    assert "date: 2026-04-29" in text
    assert "item_count: 5" in text
    assert "## Agent Read Map" in text


def test_compose_note_round_trips_through_parse() -> None:
    note = ObsidianNote(
        note_type="briefing-topic",
        generated_id="topic_alpha",
        extra={"topic_id": "topic_alpha"},
        body="# t\n\n## Agent Read Map\n\nx\n",
    )
    text = compose_note(note)
    fm, body = parse_frontmatter(text)
    assert fm["generated_id"] == "topic_alpha"
    assert fm["type"] == "briefing-topic"
    assert "## Agent Read Map" in body


def test_compose_note_is_deterministic() -> None:
    note = ObsidianNote(
        note_type="briefing-source",
        generated_id="source_x",
        extra={"source_id": "source_x", "tag": "ai"},
        body="## Agent Read Map\n\n- info\n",
    )
    a = compose_note(note)
    b = compose_note(note)
    assert a == b
