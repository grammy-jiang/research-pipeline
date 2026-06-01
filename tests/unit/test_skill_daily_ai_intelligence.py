"""Tests for the bundled daily-ai-intelligence skill (G06)."""

from __future__ import annotations

import tomllib
from pathlib import Path

SKILL_PKG = "research_pipeline.skill_data.daily-ai-intelligence"


def _skill_root() -> Path:
    # Resolve via filesystem; skill files are bundled with the package.
    import research_pipeline

    return (
        Path(research_pipeline.__file__).parent / "skill_data" / "daily-ai-intelligence"
    )


REQUIRED_REFERENCES = [
    "command-reference.md",
    "source-policy.md",
    "report-templates.md",
    "feedback-loop.md",
    "troubleshooting.md",
]


def test_skill_md_exists_and_has_frontmatter() -> None:
    skill = _skill_root() / "SKILL.md"
    assert skill.exists()
    text = skill.read_text(encoding="utf-8")
    assert text.startswith("---\n"), "SKILL.md must start with YAML frontmatter"
    head, _, _ = text[4:].partition("\n---\n")
    # Required frontmatter keys (standard Copilot CLI fields only)
    for key in ("name:", "description:", "license:"):
        assert key in head, f"SKILL.md frontmatter missing: {key}"


def test_skill_md_declares_correct_name() -> None:
    text = (_skill_root() / "SKILL.md").read_text(encoding="utf-8")
    assert "name: daily-ai-intelligence" in text


def test_skill_md_has_trigger_phrases() -> None:
    text = (_skill_root() / "SKILL.md").read_text(encoding="utf-8")
    # Trigger phrases per phase G spec / agent-evaluation reference.
    expected = [
        "daily AI technical brief",
        "daily brief",
    ]
    for phrase in expected:
        assert phrase in text, f"SKILL.md missing trigger phrase: {phrase}"


def test_skill_md_has_non_trigger_rules() -> None:
    text = (_skill_root() / "SKILL.md").read_text(encoding="utf-8")
    # Must explicitly hand off paper/literature requests to academic skill.
    assert "academic" in text.lower() or "research-pipeline" in text
    assert ("Do not use this skill" in text) or ("not use this skill" in text)
    # Must mention paper-only / literature-review handoff
    assert "paper" in text.lower() or "literature" in text.lower()


def test_skill_md_warns_about_raw_source_dumps() -> None:
    text = (_skill_root() / "SKILL.md").read_text(encoding="utf-8")
    assert "raw" in text.lower() and "cloud" in text.lower()


def test_skill_md_prefers_brief_cli_or_mcp() -> None:
    text = (_skill_root() / "SKILL.md").read_text(encoding="utf-8")
    assert "research-pipeline brief" in text


def test_config_toml_parses() -> None:
    cfg = _skill_root() / "config.toml"
    assert cfg.exists()
    with cfg.open("rb") as fh:
        data = tomllib.load(fh)
    assert isinstance(data, dict)
    assert data, "config.toml must not be empty"


def test_all_reference_files_exist_and_nonempty() -> None:
    refs_dir = _skill_root() / "references"
    assert refs_dir.is_dir()
    for name in REQUIRED_REFERENCES:
        path = refs_dir / name
        assert path.exists(), f"Missing reference: {name}"
        assert path.read_text(encoding="utf-8").strip(), f"Empty reference: {name}"


def test_skill_does_not_mention_unsupported_sources() -> None:
    text = (_skill_root() / "SKILL.md").read_text(encoding="utf-8")
    # Phase G non-goal: "no new source expansion".
    assert "browser scraping" not in text.lower()


def test_skill_references_agent_evaluation() -> None:
    """The agent-evaluation reference must exist for G08 held-out evals."""
    path = _skill_root() / "references" / "agent-evaluation.md"
    assert path.exists()
    text = path.read_text(encoding="utf-8")
    # Must enumerate the held-out tasks
    for task in [
        "run_daily_brief",
        "validate_malformed_report",
        "record_feedback",
        "export_obsidian",
        "refuse_unsupported_source",
        "paper_request_handoff",
    ]:
        assert task in text, f"agent-evaluation.md missing task: {task}"
