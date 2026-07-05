"""Regression tests for the research-pipeline skill runner run-id capture (#17)."""

from __future__ import annotations

import importlib.util
import os
import time
from pathlib import Path
from types import ModuleType

_RUNNER = (
    Path(__file__).resolve().parents[2]
    / "src/research_pipeline/skill_data/research-pipeline/runners/runner.py"
)


def _load_runner() -> ModuleType:
    spec = importlib.util.spec_from_file_location("rp_skill_runner", _RUNNER)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_capture_run_id_adopts_newest_run(tmp_path: Path) -> None:
    runner = _load_runner()
    runs = tmp_path / "runs"
    (runs / "old-run").mkdir(parents=True)
    (runs / "new-run").mkdir(parents=True)
    later = time.time() + 10
    os.utime(runs / "new-run", (later, later))

    state: dict = {"run_id": ""}
    ctx: dict = {"run_id": "", "run_dir": ""}
    result = runner.capture_run_id(state, ctx, str(tmp_path))

    assert result == "new-run"
    assert state["run_id"] == "new-run"
    assert ctx["run_id"] == "new-run"
    assert ctx["run_dir"].endswith("new-run")


def test_capture_run_id_is_noop_when_already_set(tmp_path: Path) -> None:
    runner = _load_runner()
    state: dict = {"run_id": "existing"}
    ctx: dict = {"run_id": "existing"}
    assert runner.capture_run_id(state, ctx, str(tmp_path)) == "existing"


def test_capture_run_id_returns_none_without_runs_dir(tmp_path: Path) -> None:
    runner = _load_runner()
    state: dict = {"run_id": ""}
    ctx: dict = {"run_id": ""}
    assert runner.capture_run_id(state, ctx, str(tmp_path)) is None
