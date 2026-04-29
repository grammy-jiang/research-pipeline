"""Tests for the briefing stage verifier registry (G02)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from research_pipeline.briefing.workflow_verification import (
    StageVerification,
    get_stage_verifier,
    stage_verifiers,
    verify_completed_stages,
    verify_stage,
)


@pytest.fixture
def run_root(tmp_path: Path) -> Path:
    root = tmp_path / "run"
    root.mkdir()
    return root


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_registry_covers_all_known_stages() -> None:
    keys = set(stage_verifiers().keys())
    assert keys == {"planned", "polled", "ranked", "generated", "validated", "archived"}


def test_get_stage_verifier_unknown_raises() -> None:
    with pytest.raises(KeyError):
        get_stage_verifier("nope")


def test_verify_stage_unknown_raises(run_root: Path) -> None:
    with pytest.raises(KeyError):
        verify_stage("not-a-stage", run_root)


def test_planned_ok(run_root: Path) -> None:
    _write(run_root / "workflow_state.json", json.dumps({"run_date": "2026-04-20"}))
    result = verify_stage("planned", run_root)
    assert isinstance(result, StageVerification)
    assert result.ok
    assert result.issues == ()


def test_planned_missing(run_root: Path) -> None:
    result = verify_stage("planned", run_root)
    assert not result.ok
    assert any("missing" in i for i in result.issues)


def test_planned_invalid_json(run_root: Path) -> None:
    _write(run_root / "workflow_state.json", "{not json}")
    result = verify_stage("planned", run_root)
    assert not result.ok
    assert any("invalid JSON" in i for i in result.issues)


def test_polled_ok(run_root: Path) -> None:
    _write(
        run_root / "polled.jsonl",
        '{"id": "a"}\n{"id": "b"}\n',
    )
    assert verify_stage("polled", run_root).ok


def test_polled_empty_file_flagged(run_root: Path) -> None:
    _write(run_root / "polled.jsonl", "")
    result = verify_stage("polled", run_root)
    assert not result.ok
    assert any("empty" in i for i in result.issues)


def test_polled_invalid_jsonl(run_root: Path) -> None:
    _write(run_root / "polled.jsonl", '{"ok": 1}\nnot-json\n')
    result = verify_stage("polled", run_root)
    assert not result.ok
    assert any("invalid JSON" in i for i in result.issues)


def test_ranked_ok(run_root: Path) -> None:
    _write(run_root / "ranked.jsonl", '{"cluster": 1}\n')
    assert verify_stage("ranked", run_root).ok


def test_ranked_missing(run_root: Path) -> None:
    assert not verify_stage("ranked", run_root).ok


def test_generated_ok(run_root: Path) -> None:
    _write(run_root / "daily.md", "# Daily Brief\n\nContent\n")
    assert verify_stage("generated", run_root).ok


def test_generated_empty(run_root: Path) -> None:
    _write(run_root / "daily.md", "  \n")
    result = verify_stage("generated", run_root)
    assert not result.ok


def test_validated_ok(run_root: Path) -> None:
    _write(run_root / "validation.json", json.dumps({"ok": True, "errors": []}))
    assert verify_stage("validated", run_root).ok


def test_archived_ok(run_root: Path) -> None:
    _write(
        run_root / "archived.json", json.dumps({"archived_at": "2026-04-20T00:00:00Z"})
    )
    assert verify_stage("archived", run_root).ok


def test_archived_missing(run_root: Path) -> None:
    result = verify_stage("archived", run_root)
    assert not result.ok
    assert any("missing marker" in i for i in result.issues)


def test_verify_completed_stages_order_and_results(run_root: Path) -> None:
    _write(run_root / "workflow_state.json", json.dumps({"run_date": "2026-04-20"}))
    _write(run_root / "polled.jsonl", '{"id":"a"}\n')
    # ranked missing on purpose
    results = verify_completed_stages(("planned", "polled", "ranked"), run_root)
    assert len(results) == 3
    assert [r.stage for r in results] == ["planned", "polled", "ranked"]
    assert results[0].ok and results[1].ok
    assert not results[2].ok


def test_stage_verification_is_frozen() -> None:
    sv = StageVerification(stage="polled", ok=True, issues=())
    with pytest.raises(Exception):  # noqa: B017
        sv.ok = False  # type: ignore[misc]


def test_issues_are_sorted(run_root: Path) -> None:
    # Force two issues by writing both empty + invalid via two-line invalid file
    _write(run_root / "polled.jsonl", "not-json\nalso-bad\n")
    result = verify_stage("polled", run_root)
    assert list(result.issues) == sorted(result.issues)
