"""Smoke tests for CLI command registration and --help output.

Verifies that:
- The Typer app imports cleanly
- All registered commands respond to ``--help`` with exit code 0
- The version callback prints the current version
"""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from research_pipeline import __version__
from research_pipeline.cli.app import app

runner = CliRunner()


def test_app_help() -> None:
    """Top-level --help exits 0."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert (
        "research-pipeline" in result.output.lower() or "usage" in result.output.lower()
    )


def test_version_callback() -> None:
    """--version prints the current package version."""
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.output


def _all_subcommands() -> list[str]:
    """Discover all registered CLI subcommands dynamically."""
    import typer.main

    click_app = typer.main.get_command(app)
    return sorted(click_app.commands.keys())


@pytest.mark.parametrize("cmd", _all_subcommands())
def test_subcommand_help(cmd: str) -> None:
    """Every registered subcommand responds to --help with exit 0."""
    result = runner.invoke(app, [cmd, "--help"])
    assert result.exit_code == 0, (
        f"{cmd} --help failed (exit {result.exit_code}): {result.output}"
    )
