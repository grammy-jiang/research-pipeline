"""D03 — `brief feedback` CLI command tests."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from research_pipeline.briefing.feedback_store import BriefingFeedbackStore
from research_pipeline.cli.cmd_brief import brief_app

_RUN_DATE = "2026-05-01"


def _args(workspace: Path, *extra: str) -> list[str]:
    return [
        "feedback",
        "--workspace",
        str(workspace),
        "--date",
        _RUN_DATE,
        *extra,
    ]


def test_feedback_records_explicit_signal_for_cluster(tmp_path: Path) -> None:
    runner = CliRunner()
    workspace = tmp_path / "workspace"
    result = runner.invoke(
        brief_app,
        _args(
            workspace,
            "--cluster",
            "cluster_alpha",
            "--signal",
            "keep",
            "--reason",
            "actionable",
        ),
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output

    db = workspace / "briefings" / _RUN_DATE / "feedback" / "feedback.db"
    assert db.exists()
    store = BriefingFeedbackStore(db)
    try:
        events = store.list_feedback()
    finally:
        store.close()
    assert len(events) == 1
    assert events[0].target_type == "cluster"
    assert events[0].target_id == "cluster_alpha"
    assert events[0].signal_type.value == "keep"


def test_feedback_requires_exactly_one_target(tmp_path: Path) -> None:
    runner = CliRunner()
    workspace = tmp_path / "workspace"
    no_target = runner.invoke(brief_app, _args(workspace, "--signal", "keep"))
    assert no_target.exit_code != 0

    two_targets = runner.invoke(
        brief_app,
        _args(
            workspace,
            "--cluster",
            "c1",
            "--topic",
            "t1",
            "--signal",
            "keep",
        ),
    )
    assert two_targets.exit_code != 0


def test_feedback_rejects_unsupported_signal(tmp_path: Path) -> None:
    runner = CliRunner()
    workspace = tmp_path / "workspace"
    result = runner.invoke(
        brief_app,
        _args(workspace, "--cluster", "c1", "--signal", "click"),
    )
    assert result.exit_code != 0


def test_feedback_rejects_malformed_target_id(tmp_path: Path) -> None:
    runner = CliRunner()
    workspace = tmp_path / "workspace"
    result = runner.invoke(
        brief_app,
        _args(
            workspace,
            "--cluster",
            "id with spaces",
            "--signal",
            "keep",
        ),
    )
    assert result.exit_code != 0


def test_feedback_show_lists_recorded_events(tmp_path: Path) -> None:
    runner = CliRunner()
    workspace = tmp_path / "workspace"
    record = runner.invoke(
        brief_app,
        _args(workspace, "--cluster", "c1", "--signal", "keep"),
        catch_exceptions=False,
    )
    assert record.exit_code == 0

    show = runner.invoke(
        brief_app,
        _args(workspace, "--show", "--signal", "keep"),
        catch_exceptions=False,
    )
    assert show.exit_code == 0


def test_feedback_conflicts_summarise_pos_and_neg(tmp_path: Path) -> None:
    runner = CliRunner()
    workspace = tmp_path / "workspace"
    for sig in ("keep", "hide"):
        result = runner.invoke(
            brief_app,
            _args(workspace, "--cluster", "c1", "--signal", sig),
            catch_exceptions=False,
        )
        assert result.exit_code == 0, result.output

    conflicts = runner.invoke(
        brief_app,
        _args(workspace, "--conflicts", "--signal", "keep"),
        catch_exceptions=False,
    )
    assert conflicts.exit_code == 0
