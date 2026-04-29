"""Unit tests for Phase C C04 source note export."""

from __future__ import annotations

from pathlib import Path

import pytest

from research_pipeline.briefing.obsidian import ObsidianConfig
from research_pipeline.briefing.obsidian_sources import (
    build_source_note,
    export_source_note,
    source_note_path,
)


def _make_vault(tmp_path: Path) -> Path:
    vault = tmp_path / "vault"
    (vault / "AI-Intelligence" / "Sources").mkdir(parents=True)
    return vault


_BODY = (
    "# Source X\n\n## Agent Read Map\n\n| Field | Value |\n|---|---|\n| feed | rss |\n"
)


def test_source_note_path_uses_namespace(tmp_path: Path) -> None:
    cfg = ObsidianConfig(vault_root=_make_vault(tmp_path))
    p = source_note_path(cfg, "github_releases")
    assert p.name == "github_releases.md"
    assert p.parent.name == "Sources"


def test_source_note_path_rejects_unsafe_id(tmp_path: Path) -> None:
    cfg = ObsidianConfig(vault_root=_make_vault(tmp_path))
    with pytest.raises(ValueError, match="source_id"):
        source_note_path(cfg, "../leak")


def test_build_source_note_emits_owned_frontmatter() -> None:
    note = build_source_note(
        source_id="github_releases",
        name="GitHub Releases",
        body=_BODY,
        source_class="implementation_source",
        weight=0.8,
    )
    assert note.note_type == "briefing-source"
    assert note.generated_id == "source-github_releases"
    assert note.extra["source_id"] == "github_releases"
    assert note.extra["source_class"] == "implementation_source"
    assert note.extra["weight"] == 0.8


def test_build_source_note_rejects_blank_name() -> None:
    with pytest.raises(ValueError, match="name"):
        build_source_note(source_id="x", name=" ", body=_BODY)


def test_export_source_note_writes_file(tmp_path: Path) -> None:
    cfg = ObsidianConfig(vault_root=_make_vault(tmp_path))
    note = build_source_note(
        source_id="github_releases", name="GitHub Releases", body=_BODY
    )
    p = export_source_note(note, cfg)
    assert p is not None and p.exists()
    assert "source_id: github_releases" in p.read_text(encoding="utf-8")


def test_export_source_note_dry_run_writes_nothing(tmp_path: Path) -> None:
    cfg = ObsidianConfig(vault_root=_make_vault(tmp_path), dry_run=True)
    note = build_source_note(
        source_id="github_releases", name="GitHub Releases", body=_BODY
    )
    assert export_source_note(note, cfg) is None
    assert not source_note_path(cfg, "github_releases").exists()


def test_export_source_note_refuses_human_overwrite(tmp_path: Path) -> None:
    cfg = ObsidianConfig(vault_root=_make_vault(tmp_path))
    target = source_note_path(cfg, "github_releases")
    target.write_text("# my own source page\n", encoding="utf-8")
    note = build_source_note(
        source_id="github_releases", name="GitHub Releases", body=_BODY
    )
    with pytest.raises(ValueError, match="generated_id"):
        export_source_note(note, cfg)
    assert "my own source page" in target.read_text(encoding="utf-8")


def test_export_source_note_is_idempotent(tmp_path: Path) -> None:
    cfg = ObsidianConfig(vault_root=_make_vault(tmp_path))
    note = build_source_note(
        source_id="github_releases", name="GitHub Releases", body=_BODY
    )
    p1 = export_source_note(note, cfg)
    p2 = export_source_note(note, cfg)
    assert p1 == p2
    assert p1 is not None
