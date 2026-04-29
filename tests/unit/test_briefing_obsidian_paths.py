"""Unit tests for Phase C C01 Obsidian config and path allowlist."""

from __future__ import annotations

from pathlib import Path

import pytest

from research_pipeline.briefing.obsidian import (
    GENERATED_ID_KEY,
    ObsidianConfig,
    is_owned_generated_note,
    validate_vault_path,
)


def _make_vault(tmp_path: Path) -> Path:
    vault = tmp_path / "vault"
    (vault / "AI-Intelligence" / "Daily").mkdir(parents=True)
    (vault / "AI-Intelligence" / "Topics").mkdir(parents=True)
    (vault / "AI-Intelligence" / "Sources").mkdir(parents=True)
    return vault


def test_obsidian_config_defaults(tmp_path: Path) -> None:
    vault = _make_vault(tmp_path)
    cfg = ObsidianConfig(vault_root=vault)
    assert cfg.vault_root == vault
    assert cfg.subdir == "AI-Intelligence"
    assert "Daily" in cfg.allowed_subdirs
    assert "Topics" in cfg.allowed_subdirs
    assert "Sources" in cfg.allowed_subdirs
    assert cfg.dry_run is False
    assert cfg.generated_id_key == GENERATED_ID_KEY


def test_obsidian_config_rejects_missing_vault_root(tmp_path: Path) -> None:
    missing = tmp_path / "does-not-exist"
    with pytest.raises(ValueError, match="vault_root"):
        ObsidianConfig(vault_root=missing)


def test_obsidian_config_rejects_non_directory(tmp_path: Path) -> None:
    file_path = tmp_path / "notvault.txt"
    file_path.write_text("hello", encoding="utf-8")
    with pytest.raises(ValueError, match="vault_root"):
        ObsidianConfig(vault_root=file_path)


def test_validate_vault_path_accepts_daily_note(tmp_path: Path) -> None:
    vault = _make_vault(tmp_path)
    cfg = ObsidianConfig(vault_root=vault)
    target = vault / "AI-Intelligence" / "Daily" / "2026-04-29.md"
    resolved = validate_vault_path(target, cfg)
    assert resolved == target.resolve()


def test_validate_vault_path_rejects_traversal(tmp_path: Path) -> None:
    vault = _make_vault(tmp_path)
    cfg = ObsidianConfig(vault_root=vault)
    bad = vault / "AI-Intelligence" / "Daily" / ".." / ".." / ".." / "etc.md"
    with pytest.raises(ValueError, match="escape|outside"):
        validate_vault_path(bad, cfg)


def test_validate_vault_path_rejects_outside_vault(tmp_path: Path) -> None:
    vault = _make_vault(tmp_path)
    cfg = ObsidianConfig(vault_root=vault)
    outside = tmp_path / "elsewhere.md"
    with pytest.raises(ValueError, match="escape|outside"):
        validate_vault_path(outside, cfg)


def test_validate_vault_path_rejects_unknown_subdir(tmp_path: Path) -> None:
    vault = _make_vault(tmp_path)
    (vault / "AI-Intelligence" / "Random").mkdir()
    cfg = ObsidianConfig(vault_root=vault)
    bad = vault / "AI-Intelligence" / "Random" / "foo.md"
    with pytest.raises(ValueError, match="allowed"):
        validate_vault_path(bad, cfg)


def test_validate_vault_path_rejects_outside_namespace(tmp_path: Path) -> None:
    vault = _make_vault(tmp_path)
    (vault / "Other").mkdir()
    cfg = ObsidianConfig(vault_root=vault)
    bad = vault / "Other" / "note.md"
    with pytest.raises(ValueError, match="namespace|subdir"):
        validate_vault_path(bad, cfg)


def test_validate_vault_path_rejects_non_markdown(tmp_path: Path) -> None:
    vault = _make_vault(tmp_path)
    cfg = ObsidianConfig(vault_root=vault)
    bad = vault / "AI-Intelligence" / "Daily" / "2026-04-29.txt"
    with pytest.raises(ValueError, match="markdown|\\.md"):
        validate_vault_path(bad, cfg)


def test_validate_vault_path_rejects_symlink_escape(tmp_path: Path) -> None:
    vault = _make_vault(tmp_path)
    outside = tmp_path / "outside_dir"
    outside.mkdir()
    link = vault / "AI-Intelligence" / "Daily" / "link"
    link.symlink_to(outside, target_is_directory=True)
    cfg = ObsidianConfig(vault_root=vault)
    bad = link / "leak.md"
    with pytest.raises(ValueError, match="escape|outside"):
        validate_vault_path(bad, cfg)


def test_is_owned_generated_note_matches_id(tmp_path: Path) -> None:
    vault = _make_vault(tmp_path)
    note = vault / "AI-Intelligence" / "Daily" / "2026-04-29.md"
    note.write_text(
        "---\n"
        "type: briefing-daily\n"
        f"{GENERATED_ID_KEY}: brief-2026-04-29\n"
        "---\n"
        "\nbody\n",
        encoding="utf-8",
    )
    assert is_owned_generated_note(note, "brief-2026-04-29") is True


def test_is_owned_generated_note_rejects_human_note(tmp_path: Path) -> None:
    vault = _make_vault(tmp_path)
    note = vault / "AI-Intelligence" / "Daily" / "2026-04-29.md"
    note.write_text("# my human note\n\nhand written\n", encoding="utf-8")
    assert is_owned_generated_note(note, "brief-2026-04-29") is False


def test_is_owned_generated_note_rejects_mismatched_id(tmp_path: Path) -> None:
    vault = _make_vault(tmp_path)
    note = vault / "AI-Intelligence" / "Daily" / "2026-04-29.md"
    note.write_text(
        f"---\n{GENERATED_ID_KEY}: brief-2026-04-28\n---\n",
        encoding="utf-8",
    )
    assert is_owned_generated_note(note, "brief-2026-04-29") is False


def test_is_owned_generated_note_missing_file_is_owned(tmp_path: Path) -> None:
    vault = _make_vault(tmp_path)
    note = vault / "AI-Intelligence" / "Daily" / "2026-04-29.md"
    assert is_owned_generated_note(note, "brief-2026-04-29") is True


def test_validate_vault_path_supports_custom_subdirs(tmp_path: Path) -> None:
    vault = _make_vault(tmp_path)
    (vault / "AI-Intelligence" / "Weekly").mkdir()
    cfg = ObsidianConfig(vault_root=vault, allowed_subdirs=("Daily", "Weekly"))
    weekly = vault / "AI-Intelligence" / "Weekly" / "2026-W17.md"
    assert validate_vault_path(weekly, cfg) == weekly.resolve()
    topics = vault / "AI-Intelligence" / "Topics" / "x.md"
    with pytest.raises(ValueError, match="allowed"):
        validate_vault_path(topics, cfg)
