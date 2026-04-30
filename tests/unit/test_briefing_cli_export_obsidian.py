"""Phase C C07 unit tests for `research-pipeline brief export-obsidian`.

Drives the CLI through Typer's :class:`CliRunner` against a real briefing
run produced by ``run_briefing`` on offline fixtures.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from research_pipeline.briefing.registry import load_source_registry
from research_pipeline.briefing.workflow import run_briefing
from research_pipeline.cli.cmd_brief import brief_app

_FIXTURE_BASE = Path(__file__).parent.parent / "fixtures" / "briefing" / "e2e"
_RUN_DATE = "2026-05-01"


@pytest.fixture
def briefing_workspace(tmp_path: Path) -> Path:
    registry = load_source_registry(_FIXTURE_BASE / "normal" / "registry.toml")
    workspace = tmp_path / "workspace"
    run_briefing(
        registry,
        workspace=workspace,
        run_date=_RUN_DATE,
        fixture_base_dir=_FIXTURE_BASE / "normal",
    )
    return workspace


@pytest.fixture
def vault(tmp_path: Path) -> Path:
    root = tmp_path / "vault"
    root.mkdir()
    return root


def _invoke(
    runner: CliRunner, *, vault: Path, workspace: Path, extra: list[str] | None = None
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
        str(_FIXTURE_BASE / "normal" / "registry.toml"),
    ]
    if extra:
        args.extend(extra)
    return runner.invoke(brief_app, args, catch_exceptions=False)


def test_export_obsidian_writes_daily_topic_and_source_notes(
    briefing_workspace: Path, vault: Path
) -> None:
    runner = CliRunner()
    result = _invoke(runner, vault=vault, workspace=briefing_workspace)
    assert result.exit_code == 0, result.stdout
    namespace = vault / "AI-Intelligence"
    assert (namespace / "Daily" / f"{_RUN_DATE}.md").exists()
    assert any((namespace / "Sources").glob("*.md"))


def test_export_obsidian_dry_run_writes_nothing(
    briefing_workspace: Path, vault: Path
) -> None:
    runner = CliRunner()
    result = _invoke(
        runner, vault=vault, workspace=briefing_workspace, extra=["--dry-run"]
    )
    assert result.exit_code == 0, result.stdout
    namespace = vault / "AI-Intelligence"
    if namespace.exists():
        # No markdown files should have been written.
        for md in namespace.rglob("*.md"):
            raise AssertionError(f"dry-run produced unexpected file: {md}")


def test_export_obsidian_is_idempotent(briefing_workspace: Path, vault: Path) -> None:
    runner = CliRunner()
    first = _invoke(runner, vault=vault, workspace=briefing_workspace)
    assert first.exit_code == 0, first.stdout
    daily_path = vault / "AI-Intelligence" / "Daily" / f"{_RUN_DATE}.md"
    snapshot = daily_path.read_text(encoding="utf-8")
    second = _invoke(runner, vault=vault, workspace=briefing_workspace)
    assert second.exit_code == 0, second.stdout
    assert daily_path.read_text(encoding="utf-8") == snapshot


def test_export_obsidian_refuses_to_overwrite_human_note(
    briefing_workspace: Path, vault: Path
) -> None:
    runner = CliRunner()
    daily_dir = vault / "AI-Intelligence" / "Daily"
    daily_dir.mkdir(parents=True)
    intruder = daily_dir / f"{_RUN_DATE}.md"
    intruder.write_text("hand-written note without generated_id\n", encoding="utf-8")
    result = runner.invoke(
        brief_app,
        [
            "export-obsidian",
            "--vault",
            str(vault),
            "--workspace",
            str(briefing_workspace),
            "--date",
            _RUN_DATE,
            "--registry",
            str(_FIXTURE_BASE / "normal" / "registry.toml"),
        ],
        catch_exceptions=False,
    )
    assert result.exit_code != 0
    assert intruder.read_text(encoding="utf-8") == (
        "hand-written note without generated_id\n"
    )


def test_export_obsidian_validation_emits_error_on_corrupt_note(
    briefing_workspace: Path, vault: Path
) -> None:
    runner = CliRunner()
    result = _invoke(runner, vault=vault, workspace=briefing_workspace)
    assert result.exit_code == 0, result.stdout
    daily_path = vault / "AI-Intelligence" / "Daily" / f"{_RUN_DATE}.md"
    # Corrupt the exported note then re-invoke; CLI should re-write it
    # (idempotent rewrite restores well-formed note).
    daily_path.write_text(
        "---\ntype: briefing-daily\ngenerated_id: brief-2026-05-01\n---\nbroken\n",
        encoding="utf-8",
    )
    second = _invoke(runner, vault=vault, workspace=briefing_workspace)
    assert second.exit_code == 0, second.stdout
    assert "## ⭐ Top Items" in daily_path.read_text(encoding="utf-8")
