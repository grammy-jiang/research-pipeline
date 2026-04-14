"""Tests for cmd_setup — unified skill + agent installer."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from research_pipeline.cli.cmd_setup import (
    _find_agent_source,
    _find_skill_source,
    _install_agent_files,
    run_setup,
)


class TestFindSources:
    """Test source discovery functions."""

    def test_find_skill_source_returns_path(self) -> None:
        result = _find_skill_source()
        # In dev mode, should find .github/skills/research-pipeline
        if result is not None:
            assert (result / "SKILL.md").is_file()

    def test_find_agent_source_returns_path(self) -> None:
        result = _find_agent_source()
        if result is not None:
            assert result.is_dir()
            md_files = list(result.glob("*.md"))
            assert len(md_files) > 0


class TestInstallAgentFiles:
    """Test individual agent file installation."""

    def test_copies_agent_files(self, tmp_path: Path) -> None:
        # Create a fake agent source dir
        src = tmp_path / "agents_src"
        src.mkdir()
        (src / "paper-analyzer.md").write_text("# Analyzer")
        (src / "paper-screener.md").write_text("# Screener")

        target = tmp_path / "agents_dest"
        count = _install_agent_files(src, target, symlink=False, force=False)
        assert count == 2
        assert (target / "paper-analyzer.md").read_text() == "# Analyzer"
        assert (target / "paper-screener.md").read_text() == "# Screener"

    def test_symlinks_agent_files(self, tmp_path: Path) -> None:
        src = tmp_path / "agents_src"
        src.mkdir()
        (src / "paper-analyzer.md").write_text("# Analyzer")

        target = tmp_path / "agents_dest"
        count = _install_agent_files(src, target, symlink=True, force=False)
        assert count == 1
        dest = target / "paper-analyzer.md"
        assert dest.is_symlink()
        assert dest.read_text() == "# Analyzer"

    def test_skips_existing_without_force(self, tmp_path: Path) -> None:
        src = tmp_path / "agents_src"
        src.mkdir()
        (src / "paper-analyzer.md").write_text("# New")

        target = tmp_path / "agents_dest"
        target.mkdir()
        (target / "paper-analyzer.md").write_text("# Old")

        count = _install_agent_files(src, target, symlink=False, force=False)
        assert count == 0
        assert (target / "paper-analyzer.md").read_text() == "# Old"

    def test_overwrites_existing_with_force(self, tmp_path: Path) -> None:
        src = tmp_path / "agents_src"
        src.mkdir()
        (src / "paper-analyzer.md").write_text("# New")

        target = tmp_path / "agents_dest"
        target.mkdir()
        (target / "paper-analyzer.md").write_text("# Old")

        count = _install_agent_files(src, target, symlink=False, force=True)
        assert count == 1
        assert (target / "paper-analyzer.md").read_text() == "# New"

    def test_ignores_non_md_files(self, tmp_path: Path) -> None:
        src = tmp_path / "agents_src"
        src.mkdir()
        (src / "paper-analyzer.md").write_text("# Analyzer")
        (src / "__init__.py").write_text("")
        (src / "README.txt").write_text("readme")

        target = tmp_path / "agents_dest"
        count = _install_agent_files(src, target, symlink=False, force=False)
        assert count == 1
        assert not (target / "__init__.py").exists()


class TestRunSetup:
    """Test the unified run_setup function."""

    def test_installs_both_skill_and_agents(self, tmp_path: Path) -> None:
        skill_src = tmp_path / "skill_src"
        skill_src.mkdir()
        (skill_src / "SKILL.md").write_text("# Skill")

        agent_src = tmp_path / "agent_src"
        agent_src.mkdir()
        (agent_src / "paper-analyzer.md").write_text("# Analyzer")

        skill_target = tmp_path / "skills" / "research-pipeline"
        agents_target = tmp_path / "agents"

        with (
            patch(
                "research_pipeline.cli.cmd_setup._find_skill_source",
                return_value=skill_src,
            ),
            patch(
                "research_pipeline.cli.cmd_setup._find_agent_source",
                return_value=agent_src,
            ),
        ):
            run_setup(
                skill_target=skill_target,
                agents_target=agents_target,
                force=True,
            )

        assert (skill_target / "SKILL.md").is_file()
        assert (agents_target / "paper-analyzer.md").is_file()

    def test_skip_skill(self, tmp_path: Path) -> None:
        agent_src = tmp_path / "agent_src"
        agent_src.mkdir()
        (agent_src / "paper-analyzer.md").write_text("# Analyzer")

        agents_target = tmp_path / "agents"

        with patch(
            "research_pipeline.cli.cmd_setup._find_agent_source",
            return_value=agent_src,
        ):
            run_setup(
                skill_target=tmp_path / "skill",
                agents_target=agents_target,
                skip_skill=True,
                force=True,
            )

        assert not (tmp_path / "skill").exists()
        assert (agents_target / "paper-analyzer.md").is_file()

    def test_skip_agents(self, tmp_path: Path) -> None:
        skill_src = tmp_path / "skill_src"
        skill_src.mkdir()
        (skill_src / "SKILL.md").write_text("# Skill")

        skill_target = tmp_path / "skills" / "research-pipeline"
        agents_target = tmp_path / "agents"

        with patch(
            "research_pipeline.cli.cmd_setup._find_skill_source",
            return_value=skill_src,
        ):
            run_setup(
                skill_target=skill_target,
                agents_target=agents_target,
                skip_agents=True,
                force=True,
            )

        assert (skill_target / "SKILL.md").is_file()
        assert not agents_target.exists()

    def test_skip_both_does_nothing(self, tmp_path: Path) -> None:
        run_setup(
            skill_target=tmp_path / "skill",
            agents_target=tmp_path / "agents",
            skip_skill=True,
            skip_agents=True,
        )
        assert not (tmp_path / "skill").exists()
        assert not (tmp_path / "agents").exists()

    def test_missing_skill_source_exits(self, tmp_path: Path) -> None:
        with (
            patch(
                "research_pipeline.cli.cmd_setup._find_skill_source",
                return_value=None,
            ),
            pytest.raises(SystemExit),
        ):
            run_setup(skill_target=tmp_path / "skill", agents_target=tmp_path / "a")

    def test_missing_agent_source_exits(self, tmp_path: Path) -> None:
        skill_src = tmp_path / "skill_src"
        skill_src.mkdir()
        (skill_src / "SKILL.md").write_text("# Skill")

        with (
            patch(
                "research_pipeline.cli.cmd_setup._find_skill_source",
                return_value=skill_src,
            ),
            patch(
                "research_pipeline.cli.cmd_setup._find_agent_source",
                return_value=None,
            ),
            pytest.raises(SystemExit),
        ):
            run_setup(
                skill_target=tmp_path / "skill",
                agents_target=tmp_path / "agents",
                force=True,
            )

    def test_backward_compat_alias(self) -> None:
        from research_pipeline.cli.cmd_setup import run_install_skill

        assert run_install_skill is run_setup
