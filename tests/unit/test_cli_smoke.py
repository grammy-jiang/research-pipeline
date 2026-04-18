"""Smoke tests for CLI command registration and --help output.

Verifies that:
- The Typer app imports cleanly
- All registered commands respond to ``--help`` with exit code 0
- The version callback prints the current version
"""

from __future__ import annotations

from typer.testing import CliRunner

from research_pipeline import __version__
from research_pipeline.cli.app import app

runner = CliRunner()


def test_app_help() -> None:
    """Top-level --help exits 0."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "research-pipeline" in result.output.lower() or "usage" in result.output.lower()


def test_version_callback() -> None:
    """--version prints the current package version."""
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.output


# Parametrize all subcommands that should respond to --help
_SUBCOMMANDS = [
    "plan",
    "search",
    "screen",
    "download",
    "convert",
    "extract",
    "summarize",
    "run",
    "inspect",
    "index",
    "convert-file",
    "expand",
    "quality",
    "convert-rough",
    "convert-fine",
    "export-bibtex",
    "cluster",
    "validate",
    "compare",
    "report",
    "watch",
    "analyze",
]


def test_subcommand_help() -> None:
    """Every registered subcommand responds to --help with exit 0."""
    failures: list[str] = []
    for cmd in _SUBCOMMANDS:
        result = runner.invoke(app, [cmd, "--help"])
        if result.exit_code != 0:
            failures.append(f"{cmd}: exit_code={result.exit_code}")
    assert not failures, f"Subcommands failed --help: {failures}"
