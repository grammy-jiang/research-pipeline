"""Tests for the setup --check install-health diagnostic (#19)."""

from __future__ import annotations

from pathlib import Path

from research_pipeline.cli.cmd_setup import check_installation


def test_healthy_install_reports_no_problems(tmp_path: Path) -> None:
    skill = tmp_path / ".claude" / "skills" / "research-pipeline"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text("ok")
    assert check_installation(home=tmp_path) == []


def test_detects_dangling_skill_symlink(tmp_path: Path) -> None:
    skills = tmp_path / ".claude" / "skills"
    skills.mkdir(parents=True)
    (skills / "research-pipeline").symlink_to(tmp_path / "gone")
    problems = check_installation(home=tmp_path)
    assert any("dangling skill symlink" in p for p in problems)


def test_detects_skill_missing_skill_md(tmp_path: Path) -> None:
    (tmp_path / ".claude" / "skills" / "blueprint").mkdir(parents=True)
    problems = check_installation(home=tmp_path)
    assert any("missing SKILL.md" in p for p in problems)


def test_detects_stub_agent(tmp_path: Path) -> None:
    agents = tmp_path / ".claude" / "agents"
    agents.mkdir(parents=True)
    (agents / "paper-analyzer.md").write_text("# Analyzer")  # 10-byte stub
    problems = check_installation(home=tmp_path)
    assert any("stub" in p for p in problems)


def test_detects_dangling_agent_symlink(tmp_path: Path) -> None:
    agents = tmp_path / ".claude" / "agents"
    agents.mkdir(parents=True)
    (agents / "paper-screener.md").symlink_to(tmp_path / "gone.md")
    problems = check_installation(home=tmp_path)
    assert any("dangling agent symlink" in p for p in problems)
