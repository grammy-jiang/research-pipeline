"""Regression: the report command accepts --config (manifest parity) (#31)."""

from __future__ import annotations

from typer.testing import CliRunner

from research_pipeline.cli.app import app

runner = CliRunner()


def test_report_help_lists_config() -> None:
    result = runner.invoke(app, ["report", "--help"])
    assert result.exit_code == 0
    assert "--config" in result.output


def test_report_config_option_parses() -> None:
    # The manifest's report task passes --config; parsing must not fail with
    # "No such option" (the command may still exit non-zero on a missing run).
    result = runner.invoke(
        app,
        [
            "report",
            "--run-id",
            "does-not-exist",
            "--template",
            "structured_synthesis",
            "--config",
            "config.example.toml",
        ],
    )
    assert "No such option" not in result.output
