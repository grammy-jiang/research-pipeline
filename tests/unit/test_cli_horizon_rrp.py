"""Integration tests for horizon and rrp CLI commands."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from research_pipeline.cli.app import app

runner = CliRunner()


def test_horizon_cli_writes_json(tmp_path: Path) -> None:
    output = tmp_path / "uhm.json"
    result = runner.invoke(
        app,
        [
            "horizon",
            "--score",
            "0.8",
            "--difficulty",
            "0.5",
            "--achieved",
            "50",
            "--target",
            "50",
            "--output",
            str(output),
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert output.exists()
    payload = json.loads(output.read_text())
    assert "uhm" in payload
    assert 0.0 <= payload["uhm"] <= 1.0


def test_horizon_cli_prints_to_stdout() -> None:
    result = runner.invoke(
        app,
        ["horizon", "--score", "1.0", "--achieved", "10", "--target", "10"],
    )
    assert result.exit_code == 0, result.stdout
    assert "UHM" in result.stdout


def test_rrp_cli_end_to_end(tmp_path: Path) -> None:
    report = tmp_path / "report.md"
    report.write_text(
        "# Executive Summary\n\n"
        "Covers arXiv:1111.0001 and arXiv:2222.0002. [1][2]\n\n"
        "## Themes\nTheme A and Theme B.\n\n"
        "## Contradictions\nPapers contradict each other on X.\n\n"
        "## Gaps\nAn open question remains.\n\n"
        "## Confidence\nHigh confidence in theme A.\n" + ("filler body word " * 100)
    )
    shortlist = tmp_path / "shortlist.json"
    shortlist.write_text(
        json.dumps(
            {
                "papers": [
                    {"paper_id": "arXiv:1111.0001"},
                    {"paper_id": "arXiv:2222.0002"},
                ]
            }
        )
    )
    output = tmp_path / "rrp.json"

    result = runner.invoke(
        app,
        [
            "rrp",
            "--report",
            str(report),
            "--shortlist",
            str(shortlist),
            "--output",
            str(output),
        ],
    )
    assert result.exit_code == 0, result.stdout
    payload = json.loads(output.read_text())
    assert payload["recall"] > 0.3
    assert payload["bottleneck"] in {"recall", "reasoning", "presentation"}


def test_rrp_cli_accepts_plain_list_shortlist(tmp_path: Path) -> None:
    report = tmp_path / "r.md"
    report.write_text("# Executive Summary\n\narXiv:X")
    shortlist = tmp_path / "s.json"
    shortlist.write_text(json.dumps(["arXiv:X"]))
    result = runner.invoke(
        app,
        ["rrp", "--report", str(report), "--shortlist", str(shortlist)],
    )
    assert result.exit_code == 0, result.stdout
    assert "Recall=" in result.stdout


def test_rrp_cli_missing_report(tmp_path: Path) -> None:
    shortlist = tmp_path / "s.json"
    shortlist.write_text("[]")
    result = runner.invoke(
        app,
        [
            "rrp",
            "--report",
            str(tmp_path / "missing.md"),
            "--shortlist",
            str(shortlist),
        ],
    )
    assert result.exit_code != 0
