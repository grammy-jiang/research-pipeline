"""Unit tests for Phase C C03 daily note export."""

from __future__ import annotations

from pathlib import Path

import pytest

from research_pipeline.briefing.obsidian import ObsidianConfig
from research_pipeline.briefing.obsidian_daily import (
    build_daily_note,
    daily_note_path,
    export_daily_note,
)


def _make_vault(tmp_path: Path) -> Path:
    vault = tmp_path / "vault"
    (vault / "AI-Intelligence" / "Daily").mkdir(parents=True)
    (vault / "AI-Intelligence" / "Topics").mkdir(parents=True)
    (vault / "AI-Intelligence" / "Sources").mkdir(parents=True)
    return vault


_BRIEF_BODY = (
    "# Daily AI Intelligence Brief - 2026-04-29\n\n"
    "## Agent Read Map\n\n"
    "| Field | Value |\n|---|---|\n| Use | demo |\n\n"
    "## Top Items\n\n- alpha [[Topic Alpha]]\n"
)


def test_daily_note_path_uses_namespace(tmp_path: Path) -> None:
    cfg = ObsidianConfig(vault_root=_make_vault(tmp_path))
    p = daily_note_path(cfg, "2026-04-29")
    assert p.name == "2026-04-29.md"
    assert p.parent.name == "Daily"
    assert p.parent.parent.name == "AI-Intelligence"


def test_build_daily_note_emits_owned_frontmatter() -> None:
    note = build_daily_note(
        run_date="2026-04-29", body=_BRIEF_BODY, item_count=3, link_count=2
    )
    assert note.note_type == "briefing-daily"
    assert note.generated_id == "brief-2026-04-29"
    assert note.extra["date"] == "2026-04-29"
    assert note.extra["item_count"] == 3
    assert "## Agent Read Map" in note.body
    # Wiki-link must be preserved verbatim.
    assert "[[Topic Alpha]]" in note.body


def test_build_daily_note_strips_legacy_frontmatter() -> None:
    raw = "---\ntype: daily-brief\ndate: 2026-04-29\n---\n\n" + _BRIEF_BODY
    note = build_daily_note(run_date="2026-04-29", body=raw, item_count=1, link_count=1)
    assert "type: daily-brief" not in note.body
    assert note.body.lstrip().startswith("# Daily AI Intelligence Brief")


def test_build_daily_note_rejects_missing_agent_read_map() -> None:
    with pytest.raises(ValueError, match="Agent Read Map"):
        build_daily_note(
            run_date="2026-04-29",
            body="# nope\n\nbody only\n",
            item_count=0,
            link_count=0,
        )


def test_export_daily_note_writes_file(tmp_path: Path) -> None:
    cfg = ObsidianConfig(vault_root=_make_vault(tmp_path))
    note = build_daily_note(
        run_date="2026-04-29", body=_BRIEF_BODY, item_count=3, link_count=2
    )
    path = export_daily_note(note, cfg)
    assert path is not None
    assert path.exists()
    text = path.read_text(encoding="utf-8")
    assert "generated_id: brief-2026-04-29" in text
    assert "type: briefing-daily" in text
    assert "[[Topic Alpha]]" in text


def test_export_daily_note_is_idempotent(tmp_path: Path) -> None:
    cfg = ObsidianConfig(vault_root=_make_vault(tmp_path))
    note = build_daily_note(
        run_date="2026-04-29", body=_BRIEF_BODY, item_count=3, link_count=2
    )
    p1 = export_daily_note(note, cfg)
    p2 = export_daily_note(note, cfg)
    assert p1 == p2
    text_a = p1.read_text(encoding="utf-8")  # type: ignore[union-attr]
    text_b = p2.read_text(encoding="utf-8")  # type: ignore[union-attr]
    assert text_a == text_b


def test_export_daily_note_dry_run_writes_nothing(tmp_path: Path) -> None:
    cfg = ObsidianConfig(vault_root=_make_vault(tmp_path), dry_run=True)
    note = build_daily_note(
        run_date="2026-04-29", body=_BRIEF_BODY, item_count=3, link_count=2
    )
    path = export_daily_note(note, cfg)
    assert path is None
    target = daily_note_path(cfg, "2026-04-29")
    assert not target.exists()


def test_export_daily_note_refuses_to_overwrite_human_note(tmp_path: Path) -> None:
    vault = _make_vault(tmp_path)
    cfg = ObsidianConfig(vault_root=vault)
    target = daily_note_path(cfg, "2026-04-29")
    target.write_text("# my own daily journal\n\nhandwritten notes\n", encoding="utf-8")
    note = build_daily_note(
        run_date="2026-04-29", body=_BRIEF_BODY, item_count=3, link_count=2
    )
    with pytest.raises(ValueError, match="ownership|generated_id"):
        export_daily_note(note, cfg)
    # Human content untouched.
    assert "handwritten notes" in target.read_text(encoding="utf-8")


def test_export_daily_note_overwrites_owned_note(tmp_path: Path) -> None:
    cfg = ObsidianConfig(vault_root=_make_vault(tmp_path))
    n1 = build_daily_note(
        run_date="2026-04-29", body=_BRIEF_BODY, item_count=3, link_count=2
    )
    export_daily_note(n1, cfg)
    new_body = _BRIEF_BODY + "\n## More\n\nfresh content\n"
    n2 = build_daily_note(
        run_date="2026-04-29", body=new_body, item_count=4, link_count=3
    )
    p2 = export_daily_note(n2, cfg)
    assert p2 is not None
    assert "fresh content" in p2.read_text(encoding="utf-8")
