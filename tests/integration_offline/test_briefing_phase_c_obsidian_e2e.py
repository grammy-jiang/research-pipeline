"""Phase C C08 offline end-to-end tests for Obsidian vault export.

Drives `research-pipeline brief export-obsidian` against a fully populated
briefing workspace produced from offline fixtures, then asserts vault
safety, idempotency, dry-run behaviour, and validation of exported notes.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from research_pipeline.briefing.obsidian import GENERATED_ID_KEY
from research_pipeline.briefing.registry import load_source_registry
from research_pipeline.briefing.validate_obsidian import validate_obsidian_export
from research_pipeline.briefing.workflow import run_briefing
from research_pipeline.cli.cmd_brief import brief_app

_FIXTURE_BASE = Path(__file__).parent.parent / "fixtures" / "briefing" / "e2e"
_RUN_DATE = "2026-05-01"


def _seed_workspace(tmp_path: Path, scenario: str) -> Path:
    """Run the briefing pipeline using ``scenario`` fixtures and return workspace."""
    workspace = tmp_path / "workspace"
    registry = load_source_registry(_FIXTURE_BASE / scenario / "registry.toml")
    run_briefing(
        registry,
        workspace=workspace,
        run_date=_RUN_DATE,
        fixture_base_dir=_FIXTURE_BASE / scenario,
    )
    return workspace


def _run_cli(
    runner: CliRunner,
    *,
    vault: Path,
    workspace: Path,
    scenario: str,
    extra: list[str] | None = None,
) -> object:
    args = [
        "export-obsidian",
        "--vault",
        str(vault),
        "--workspace",
        str(workspace),
        "--date",
        _RUN_DATE,
        "--registry",
        str(_FIXTURE_BASE / scenario / "registry.toml"),
    ]
    if extra:
        args.extend(extra)
    return runner.invoke(brief_app, args, catch_exceptions=False)


class TestObsidianExportE2E:
    """Full export under the obsidian_export fixture."""

    def test_full_export_writes_validated_notes(self, tmp_path: Path) -> None:
        workspace = _seed_workspace(tmp_path, "obsidian_export")
        vault = tmp_path / "vault"
        vault.mkdir()
        result = _run_cli(
            CliRunner(), vault=vault, workspace=workspace, scenario="obsidian_export"
        )
        assert result.exit_code == 0, result.stdout

        namespace = vault / "AI-Intelligence"
        daily = namespace / "Daily" / f"{_RUN_DATE}.md"
        assert daily.exists()
        sources = sorted((namespace / "Sources").glob("*.md"))
        assert sources, "expected at least one source note"
        topics = sorted((namespace / "Topics").glob("*.md"))

        validation = validate_obsidian_export(
            daily_path=daily,
            topic_paths=topics,
            source_paths=sources,
        )
        assert validation.passed, validation.errors

    def test_dry_run_does_not_write(self, tmp_path: Path) -> None:
        workspace = _seed_workspace(tmp_path, "obsidian_export")
        vault = tmp_path / "vault"
        vault.mkdir()
        result = _run_cli(
            CliRunner(),
            vault=vault,
            workspace=workspace,
            scenario="obsidian_export",
            extra=["--dry-run"],
        )
        assert result.exit_code == 0, result.stdout
        for md in vault.rglob("*.md"):
            raise AssertionError(f"dry-run produced unexpected file: {md}")

    def test_repeated_export_is_byte_idempotent(self, tmp_path: Path) -> None:
        workspace = _seed_workspace(tmp_path, "obsidian_export")
        vault = tmp_path / "vault"
        vault.mkdir()
        runner = CliRunner()
        first = _run_cli(
            runner, vault=vault, workspace=workspace, scenario="obsidian_export"
        )
        assert first.exit_code == 0, first.stdout
        snapshot = {
            md: md.read_bytes() for md in (vault / "AI-Intelligence").rglob("*.md")
        }
        second = _run_cli(
            runner, vault=vault, workspace=workspace, scenario="obsidian_export"
        )
        assert second.exit_code == 0, second.stdout
        for md, content in snapshot.items():
            assert md.read_bytes() == content, f"file changed on re-export: {md}"

    def test_overwrites_owned_generated_note_in_place(self, tmp_path: Path) -> None:
        workspace = _seed_workspace(tmp_path, "obsidian_export")
        vault = tmp_path / "vault"
        vault.mkdir()
        runner = CliRunner()
        first = _run_cli(
            runner, vault=vault, workspace=workspace, scenario="obsidian_export"
        )
        assert first.exit_code == 0, first.stdout

        daily = vault / "AI-Intelligence" / "Daily" / f"{_RUN_DATE}.md"
        text = daily.read_text(encoding="utf-8")
        assert f"{GENERATED_ID_KEY}: brief-{_RUN_DATE}" in text
        # Tamper with the body but keep the owning frontmatter.
        head, _, _ = text.partition("\n---\n")
        daily.write_text(head + "\n---\nstale body\n", encoding="utf-8")
        second = _run_cli(
            runner, vault=vault, workspace=workspace, scenario="obsidian_export"
        )
        assert second.exit_code == 0, second.stdout
        restored = daily.read_text(encoding="utf-8")
        assert "## Agent Read Map" in restored
        assert "stale body" not in restored


class TestObsidianExportRefusesHumanNotes:
    """Pre-existing human-authored note must be preserved."""

    def test_refuses_to_overwrite_human_note(self, tmp_path: Path) -> None:
        workspace = _seed_workspace(tmp_path, "obsidian_existing_notes")
        vault = tmp_path / "vault"
        # Pre-seed a human note at the daily target path.
        daily_dir = vault / "AI-Intelligence" / "Daily"
        daily_dir.mkdir(parents=True)
        intruder = daily_dir / f"{_RUN_DATE}.md"
        intruder.write_text(
            "# Personal note\n\nDo not overwrite me.\n", encoding="utf-8"
        )
        result = _run_cli(
            CliRunner(),
            vault=vault,
            workspace=workspace,
            scenario="obsidian_existing_notes",
        )
        assert result.exit_code != 0
        assert intruder.read_text(encoding="utf-8") == (
            "# Personal note\n\nDo not overwrite me.\n"
        )

    def test_unsafe_vault_traversal_is_rejected(self, tmp_path: Path) -> None:
        workspace = _seed_workspace(tmp_path, "obsidian_existing_notes")
        # Vault root which does not exist forces ObsidianConfig validation
        # to fail; CLI should not proceed.
        bogus_vault = tmp_path / "does-not-exist"
        with pytest.raises(ValidationError):
            CliRunner().invoke(
                brief_app,
                [
                    "export-obsidian",
                    "--vault",
                    str(bogus_vault),
                    "--workspace",
                    str(workspace),
                    "--date",
                    _RUN_DATE,
                    "--registry",
                    str(_FIXTURE_BASE / "obsidian_existing_notes" / "registry.toml"),
                ],
                catch_exceptions=False,
            )
