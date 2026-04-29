"""Unit tests for Phase C C04 topic note export."""

from __future__ import annotations

from pathlib import Path

import pytest

from research_pipeline.briefing.obsidian import ObsidianConfig
from research_pipeline.briefing.obsidian_topics import (
    build_topic_note,
    export_topic_note,
    topic_note_path,
)


def _make_vault(tmp_path: Path) -> Path:
    vault = tmp_path / "vault"
    (vault / "AI-Intelligence" / "Topics").mkdir(parents=True)
    return vault


_BODY = (
    "# Topic Alpha\n\n## Agent Read Map\n\n| use | watchlist |\n\nlinks: [[Source X]]\n"
)


def test_topic_note_path_uses_namespace(tmp_path: Path) -> None:
    cfg = ObsidianConfig(vault_root=_make_vault(tmp_path))
    p = topic_note_path(cfg, "topic_alpha")
    assert p.name == "topic_alpha.md"
    assert p.parent.name == "Topics"


def test_topic_note_path_rejects_unsafe_topic_id(tmp_path: Path) -> None:
    cfg = ObsidianConfig(vault_root=_make_vault(tmp_path))
    with pytest.raises(ValueError, match="topic_id"):
        topic_note_path(cfg, "../escape")


def test_build_topic_note_emits_owned_frontmatter() -> None:
    note = build_topic_note(
        topic_id="topic_alpha",
        name="Topic Alpha",
        body=_BODY,
        status="active",
        fatigue_score=0.25,
        last_reported_at="2026-04-29",
        aliases=("alpha", "a-prime"),
    )
    assert note.note_type == "briefing-topic"
    assert note.generated_id == "topic-topic_alpha"
    assert note.extra["topic_id"] == "topic_alpha"
    assert note.extra["status"] == "active"
    assert note.extra["fatigue_score"] == 0.25
    assert "[[Source X]]" in note.body


def test_build_topic_note_rejects_blank_name() -> None:
    with pytest.raises(ValueError, match="name"):
        build_topic_note(topic_id="topic_x", name="   ", body=_BODY)


def test_export_topic_note_writes_file(tmp_path: Path) -> None:
    cfg = ObsidianConfig(vault_root=_make_vault(tmp_path))
    note = build_topic_note(
        topic_id="topic_alpha", name="Topic Alpha", body=_BODY, status="active"
    )
    p = export_topic_note(note, cfg)
    assert p is not None
    text = p.read_text(encoding="utf-8")
    assert "type: briefing-topic" in text
    assert "topic_id: topic_alpha" in text
    assert "[[Source X]]" in text


def test_export_topic_note_dry_run_writes_nothing(tmp_path: Path) -> None:
    cfg = ObsidianConfig(vault_root=_make_vault(tmp_path), dry_run=True)
    note = build_topic_note(topic_id="topic_alpha", name="Topic Alpha", body=_BODY)
    assert export_topic_note(note, cfg) is None
    assert not topic_note_path(cfg, "topic_alpha").exists()


def test_export_topic_note_refuses_to_overwrite_human_note(tmp_path: Path) -> None:
    cfg = ObsidianConfig(vault_root=_make_vault(tmp_path))
    target = topic_note_path(cfg, "topic_alpha")
    target.write_text("# my own topic notes\n", encoding="utf-8")
    note = build_topic_note(topic_id="topic_alpha", name="Topic Alpha", body=_BODY)
    with pytest.raises(ValueError, match="generated_id"):
        export_topic_note(note, cfg)
    assert "my own topic notes" in target.read_text(encoding="utf-8")


def test_export_topic_note_is_idempotent(tmp_path: Path) -> None:
    cfg = ObsidianConfig(vault_root=_make_vault(tmp_path))
    note = build_topic_note(topic_id="topic_alpha", name="Topic Alpha", body=_BODY)
    p1 = export_topic_note(note, cfg)
    p2 = export_topic_note(note, cfg)
    assert p1 == p2
    assert p1 is not None and p1.read_text(encoding="utf-8") == p2.read_text(  # type: ignore[union-attr]
        encoding="utf-8"
    )
