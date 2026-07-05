"""Regression test: check_completion finds validate's output filename (#32)."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

_CC = (
    Path(__file__).resolve().parents[2]
    / "src/research_pipeline/skill_data/research-pipeline/scripts/check_completion.py"
)


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("rp_check_completion", _CC)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_finds_validation_result_json(tmp_path: Path) -> None:
    cc = _load()
    (tmp_path / "validation_result.json").write_text('{"passed": true}')
    found = cc._find_first(tmp_path, cc.VALIDATION_CANDIDATES)
    assert found is not None
    assert found.name == "validation_result.json"


def test_still_finds_legacy_validation_json(tmp_path: Path) -> None:
    cc = _load()
    (tmp_path / "validate").mkdir()
    (tmp_path / "validate" / "validation.json").write_text('{"passed": true}')
    found = cc._find_first(tmp_path, cc.VALIDATION_CANDIDATES)
    assert found is not None
