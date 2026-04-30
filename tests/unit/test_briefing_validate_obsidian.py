"""Unit tests for Phase C C06 Obsidian export validators."""

from __future__ import annotations

from pathlib import Path

import pytest

from research_pipeline.briefing.obsidian_daily import build_daily_note
from research_pipeline.briefing.obsidian_notes import compose_note
from research_pipeline.briefing.obsidian_sources import build_source_note
from research_pipeline.briefing.obsidian_topics import build_topic_note
from research_pipeline.briefing.validate_obsidian import (
    validate_obsidian_export,
    validate_obsidian_note_file,
)


def _write(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _daily_body() -> str:
    return "## Agent Read Map\n\nthis is the agent read map\n\n## ⭐ Top Items\nsig\n"


def _topic_body() -> str:
    return "## Agent Read Map\nentry\n"


def test_validate_obsidian_note_file_happy_daily(tmp_path: Path) -> None:
    note = build_daily_note(
        run_date="2026-04-25",
        body=_daily_body(),
        item_count=3,
        link_count=4,
    )
    path = _write(tmp_path / "Daily.md", compose_note(note))
    result = validate_obsidian_note_file(path, expected_note_type="briefing-daily")
    assert result.passed, result.errors


def test_validate_obsidian_note_file_detects_wrong_type(tmp_path: Path) -> None:
    note = build_topic_note(
        topic_id="topic-alpha",
        name="Topic Alpha",
        body=_topic_body(),
    )
    path = _write(tmp_path / "Topic.md", compose_note(note))
    result = validate_obsidian_note_file(path, expected_note_type="briefing-daily")
    assert not result.passed
    assert any("expected type='briefing-daily'" in e for e in result.errors)


def test_validate_obsidian_note_file_detects_missing_heading(tmp_path: Path) -> None:
    note = build_topic_note(
        topic_id="topic-alpha",
        name="Topic Alpha",
        body=_topic_body(),
    )
    raw = compose_note(note).replace("## Agent Read Map", "## Random Heading")
    path = _write(tmp_path / "Topic.md", raw)
    result = validate_obsidian_note_file(path, expected_note_type="briefing-topic")
    assert not result.passed
    assert any("Agent Read Map" in e for e in result.errors)


def test_validate_obsidian_note_file_detects_wikilink_rewrite(tmp_path: Path) -> None:
    note = build_topic_note(
        topic_id="topic-alpha",
        name="Topic Alpha",
        body="## Agent Read Map\nsee [Topic Alpha](Topic-Alpha.md) page\n",
    )
    path = _write(tmp_path / "Topic.md", compose_note(note))
    result = validate_obsidian_note_file(path, expected_note_type="briefing-topic")
    assert not result.passed
    assert any("converted to markdown" in e for e in result.errors)


def test_validate_obsidian_note_file_accepts_real_wikilinks(tmp_path: Path) -> None:
    note = build_topic_note(
        topic_id="topic-alpha",
        name="Topic Alpha",
        body="## Agent Read Map\nsee [[Topic Alpha]] and [[Source X|sx]]\n",
    )
    path = _write(tmp_path / "Topic.md", compose_note(note))
    result = validate_obsidian_note_file(path, expected_note_type="briefing-topic")
    assert result.passed, result.errors
    assert result.metrics["wikilink_count"] == 2


def test_validate_obsidian_note_file_missing_file(tmp_path: Path) -> None:
    missing = tmp_path / "nope.md"
    result = validate_obsidian_note_file(missing, expected_note_type="briefing-daily")
    assert not result.passed
    assert any("missing exported note" in e for e in result.errors)


def test_validate_obsidian_note_file_unknown_type_raises(tmp_path: Path) -> None:
    path = _write(tmp_path / "any.md", "---\ntype: foo\n---\nbody\n")
    with pytest.raises(ValueError):
        validate_obsidian_note_file(path, expected_note_type="bogus")


def test_validate_obsidian_note_file_expected_generated_id(tmp_path: Path) -> None:
    note = build_topic_note(
        topic_id="topic-alpha",
        name="Topic Alpha",
        body=_topic_body(),
    )
    path = _write(tmp_path / "Topic.md", compose_note(note))
    bad = validate_obsidian_note_file(
        path,
        expected_note_type="briefing-topic",
        expected_generated_id="topic-other",
    )
    assert not bad.passed
    good = validate_obsidian_note_file(
        path,
        expected_note_type="briefing-topic",
        expected_generated_id="topic-topic-alpha",
    )
    assert good.passed, good.errors


def test_validate_obsidian_export_bundles_results(tmp_path: Path) -> None:
    daily = build_daily_note(
        run_date="2026-04-25", body=_daily_body(), item_count=3, link_count=4
    )
    topic = build_topic_note(
        topic_id="topic-alpha", name="Topic Alpha", body=_topic_body()
    )
    source = build_source_note(source_id="src-x", name="Source X", body=_topic_body())
    daily_path = _write(tmp_path / "Daily.md", compose_note(daily))
    topic_path = _write(tmp_path / "Topic.md", compose_note(topic))
    source_path = _write(tmp_path / "Source.md", compose_note(source))

    result = validate_obsidian_export(
        daily_path=daily_path,
        topic_paths=[topic_path],
        source_paths=[source_path],
    )
    assert result.passed, result.errors
    assert result.metrics == {
        "daily_count": 1,
        "topic_count": 1,
        "source_count": 1,
        "wikilink_count": 0,
    }


def test_validate_obsidian_export_aggregates_errors(tmp_path: Path) -> None:
    daily = build_daily_note(
        run_date="2026-04-25", body=_daily_body(), item_count=3, link_count=4
    )
    daily_path = _write(tmp_path / "Daily.md", compose_note(daily))
    result = validate_obsidian_export(
        daily_path=daily_path,
        topic_paths=[tmp_path / "missing.md"],
    )
    assert not result.passed
    assert any("missing exported note" in e for e in result.errors)
