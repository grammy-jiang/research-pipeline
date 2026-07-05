"""Regression test: skill runner --status exits 0 on a populated state (#22)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

_RUNNER = (
    Path(__file__).resolve().parents[2]
    / "src/research_pipeline/skill_data/research-pipeline/runners/runner.py"
)


def test_status_returns_zero_on_populated_state(tmp_path: Path) -> None:
    state = tmp_path / "workflow_state.json"
    state.write_text(
        json.dumps(
            {
                "workflow_id": "wf",
                "run_id": "r1",
                "status": "running",
                "topic": "t",
                "tasks": {"plan": {"status": "accepted"}},
            }
        )
    )
    result = subprocess.run(
        [sys.executable, str(_RUNNER), "--status", "--state", str(state)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "Status" in result.stdout


def test_status_returns_nonzero_without_state(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(_RUNNER),
            "--status",
            "--state",
            str(tmp_path / "none.json"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1
