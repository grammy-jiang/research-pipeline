"""Tests for the install-skill CLI command."""

from __future__ import annotations

from pathlib import Path

import pytest


class TestFindSkillSource:
    def test_finds_repo_skill_data(self) -> None:
        from research_pipeline.cli.cmd_install_skill import _find_skill_source

        result = _find_skill_source()
        assert result is not None
        assert (result / "SKILL.md").is_file()
        assert (result / "config.toml").is_file()
        assert (result / "references").is_dir()


class TestRunInstallSkill:
    def test_copy_skill(self, tmp_path: Path) -> None:
        from research_pipeline.cli.cmd_install_skill import run_install_skill

        target = tmp_path / "skills" / "research-pipeline"
        run_install_skill(target=target, symlink=False, force=False)

        assert target.is_dir()
        assert (target / "SKILL.md").is_file()
        assert (target / "config.toml").is_file()
        assert (target / "references" / "sub-agents.md").is_file()

    def test_symlink_skill(self, tmp_path: Path) -> None:
        from research_pipeline.cli.cmd_install_skill import run_install_skill

        target = tmp_path / "skills" / "research-pipeline"
        run_install_skill(target=target, symlink=True, force=False)

        assert target.is_symlink()
        assert (target / "SKILL.md").is_file()

    def test_refuses_overwrite_without_force(self, tmp_path: Path) -> None:
        from research_pipeline.cli.cmd_install_skill import run_install_skill

        target = tmp_path / "skills" / "research-pipeline"
        target.mkdir(parents=True)

        with pytest.raises(SystemExit):
            run_install_skill(target=target, symlink=False, force=False)

    def test_force_overwrite(self, tmp_path: Path) -> None:
        from research_pipeline.cli.cmd_install_skill import run_install_skill

        target = tmp_path / "skills" / "research-pipeline"
        target.mkdir(parents=True)
        (target / "old_file.txt").write_text("old")

        run_install_skill(target=target, symlink=False, force=True)

        assert (target / "SKILL.md").is_file()
        assert not (target / "old_file.txt").exists()

    def test_force_overwrite_symlink(self, tmp_path: Path) -> None:
        from research_pipeline.cli.cmd_install_skill import run_install_skill

        target = tmp_path / "skills" / "research-pipeline"
        target.parent.mkdir(parents=True)
        target.symlink_to(tmp_path)  # Create initial symlink

        run_install_skill(target=target, symlink=False, force=True)

        assert target.is_dir()
        assert not target.is_symlink()
        assert (target / "SKILL.md").is_file()
