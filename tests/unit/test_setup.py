"""Tests for cmd_setup — unified skill + agent installer."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from research_pipeline.cli.cmd_setup import (
    DEFAULT_COPILOT_MCP_CONFIG,
    DEFAULT_COPILOT_SKILL_DIR,
    DEFAULT_MCP_CONFIG_FILE,
    DEFAULT_VSCODE_MCP_CONFIG,
    _find_agent_source,
    _find_skill_source,
    _install_agent_files,
    _install_mcp_config,
    _update_copilot_mcp_config,
    _update_vscode_mcp_config,
    run_setup,
)


class TestFindSources:
    """Test source discovery functions."""

    def test_find_skill_source_returns_path(self) -> None:
        result = _find_skill_source()
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
                mcp_config_target=tmp_path / "mcp.json",
                force=True,
            )

        assert (skill_target / "SKILL.md").is_file()
        assert (agents_target / "paper-analyzer.md").is_file()
        assert (tmp_path / "mcp.json").is_file()

    def test_default_installs_claude_and_codex_skills(self, tmp_path: Path) -> None:
        skill_src = tmp_path / "skill_src"
        skill_src.mkdir()
        (skill_src / "SKILL.md").write_text("# Skill")

        agent_src = tmp_path / "agent_src"
        agent_src.mkdir()
        (agent_src / "paper-analyzer.md").write_text("# Analyzer")

        claude_target = tmp_path / ".claude" / "skills" / "research-pipeline"
        codex_target = tmp_path / ".codex" / "skills" / "research-pipeline"
        agents_target = tmp_path / ".claude" / "agents"

        with (
            patch(
                "research_pipeline.cli.cmd_setup._find_skill_source",
                return_value=skill_src,
            ),
            patch(
                "research_pipeline.cli.cmd_setup._find_agent_source",
                return_value=agent_src,
            ),
            patch(
                "research_pipeline.cli.cmd_setup.DEFAULT_SKILL_TARGETS",
                (claude_target, codex_target),
            ),
        ):
            run_setup(
                agents_target=agents_target,
                mcp_config_target=tmp_path / "mcp.json",
                force=True,
            )

        assert (claude_target / "SKILL.md").is_file()
        assert (codex_target / "SKILL.md").is_file()
        assert (agents_target / "paper-analyzer.md").is_file()
        assert (tmp_path / "mcp.json").is_file()

    def test_default_skips_existing_skill_and_installs_missing(
        self,
        tmp_path: Path,
    ) -> None:
        skill_src = tmp_path / "skill_src"
        skill_src.mkdir()
        (skill_src / "SKILL.md").write_text("# New Skill")

        claude_target = tmp_path / ".claude" / "skills" / "research-pipeline"
        claude_target.mkdir(parents=True)
        (claude_target / "SKILL.md").write_text("# Existing Skill")
        codex_target = tmp_path / ".codex" / "skills" / "research-pipeline"

        with (
            patch(
                "research_pipeline.cli.cmd_setup._find_skill_source",
                return_value=skill_src,
            ),
            patch(
                "research_pipeline.cli.cmd_setup.DEFAULT_SKILL_TARGETS",
                (claude_target, codex_target),
            ),
        ):
            run_setup(
                skip_agents=True,
                skip_mcp=True,
            )

        assert (claude_target / "SKILL.md").read_text() == "# Existing Skill"
        assert (codex_target / "SKILL.md").read_text() == "# New Skill"

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
                skip_mcp=True,
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
                skip_mcp=True,
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
            skip_mcp=True,
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


class TestInstallMcpConfig:
    """Test MCP config snippet installation."""

    def test_installs_mcp_config(self, tmp_path: Path) -> None:
        target = tmp_path / "mcp.json"

        installed = _install_mcp_config(target, force=False)

        assert installed is True
        data = json.loads(target.read_text())
        server = data["mcpServers"]["research-pipeline"]
        assert server["type"] == "stdio"
        assert server["command"] == "research-pipeline"
        assert server["args"] == ["mcp", "serve"]

    def test_skips_existing_without_force(self, tmp_path: Path) -> None:
        target = tmp_path / "mcp.json"
        target.write_text("{}\n")

        installed = _install_mcp_config(target, force=False)

        assert installed is False
        assert target.read_text() == "{}\n"

    def test_default_mcp_config_path_is_packaged_config(self) -> None:
        assert DEFAULT_MCP_CONFIG_FILE.name == "mcp.json"


class TestUpdateCopilotMcpConfig:
    """Test merging the research-pipeline entry into ~/.copilot/mcp-config.json."""

    def test_creates_file_when_missing(self, tmp_path: Path) -> None:
        target = tmp_path / ".copilot" / "mcp-config.json"

        updated = _update_copilot_mcp_config(target, force=False)

        assert updated is True
        data = json.loads(target.read_text())
        server = data["mcpServers"]["research-pipeline"]
        assert server["type"] == "stdio"
        assert server["command"] == "research-pipeline"
        assert server["args"] == ["mcp", "serve"]

    def test_merges_into_existing_config_preserving_other_servers(
        self, tmp_path: Path
    ) -> None:
        target = tmp_path / "mcp-config.json"
        target.write_text(
            json.dumps(
                {
                    "mcpServers": {
                        "other-server": {
                            "type": "stdio",
                            "command": "other-cmd",
                            "args": [],
                        }
                    }
                }
            )
            + "\n"
        )

        updated = _update_copilot_mcp_config(target, force=False)

        assert updated is True
        data = json.loads(target.read_text())
        assert "other-server" in data["mcpServers"], (
            "existing entries must be preserved"
        )
        assert "research-pipeline" in data["mcpServers"]

    def test_skips_when_entry_exists_without_force(self, tmp_path: Path) -> None:
        target = tmp_path / "mcp-config.json"
        original = {
            "mcpServers": {"research-pipeline": {"type": "stdio", "command": "old"}}
        }
        target.write_text(json.dumps(original) + "\n")

        updated = _update_copilot_mcp_config(target, force=False)

        assert updated is False
        data = json.loads(target.read_text())
        assert data["mcpServers"]["research-pipeline"]["command"] == "old"

    def test_overwrites_entry_with_force(self, tmp_path: Path) -> None:
        target = tmp_path / "mcp-config.json"
        original = {
            "mcpServers": {"research-pipeline": {"type": "stdio", "command": "old"}}
        }
        target.write_text(json.dumps(original) + "\n")

        updated = _update_copilot_mcp_config(target, force=True)

        assert updated is True
        data = json.loads(target.read_text())
        assert data["mcpServers"]["research-pipeline"]["command"] == "research-pipeline"

    def test_recovers_from_corrupt_json(self, tmp_path: Path) -> None:
        target = tmp_path / "mcp-config.json"
        target.write_text("not valid json")

        updated = _update_copilot_mcp_config(target, force=False)

        assert updated is True
        data = json.loads(target.read_text())
        assert "research-pipeline" in data["mcpServers"]

    def test_default_copilot_mcp_config_path(self) -> None:
        assert DEFAULT_COPILOT_MCP_CONFIG.name == "mcp-config.json"
        assert ".copilot" in str(DEFAULT_COPILOT_MCP_CONFIG)

    def test_default_copilot_skill_dir_path(self) -> None:
        assert DEFAULT_COPILOT_SKILL_DIR.name == "research-pipeline"
        assert ".copilot" in str(DEFAULT_COPILOT_SKILL_DIR)
        assert "skills" in str(DEFAULT_COPILOT_SKILL_DIR)


class TestUpdateVscodeMcpConfig:
    """Test merging the research-pipeline entry into ~/.config/Code/User/mcp.json."""

    def test_creates_file_when_missing(self, tmp_path: Path) -> None:
        target = tmp_path / "Code" / "User" / "mcp.json"

        updated = _update_vscode_mcp_config(target, force=False)

        assert updated is True
        data = json.loads(target.read_text())
        server = data["servers"]["research-pipeline"]
        assert server["type"] == "stdio"
        assert server["command"] == "research-pipeline"
        assert server["args"] == ["mcp", "serve"]

    def test_merges_into_existing_config_preserving_other_servers(
        self, tmp_path: Path
    ) -> None:
        target = tmp_path / "mcp.json"
        target.write_text(
            json.dumps(
                {
                    "servers": {
                        "other-server": {
                            "type": "stdio",
                            "command": "other-cmd",
                            "args": [],
                        }
                    }
                }
            )
            + "\n"
        )

        updated = _update_vscode_mcp_config(target, force=False)

        assert updated is True
        data = json.loads(target.read_text())
        assert "other-server" in data["servers"], "existing entries must be preserved"
        assert "research-pipeline" in data["servers"]

    def test_skips_when_entry_exists_without_force(self, tmp_path: Path) -> None:
        target = tmp_path / "mcp.json"
        original = {
            "servers": {"research-pipeline": {"type": "stdio", "command": "old"}}
        }
        target.write_text(json.dumps(original) + "\n")

        updated = _update_vscode_mcp_config(target, force=False)

        assert updated is False
        data = json.loads(target.read_text())
        assert data["servers"]["research-pipeline"]["command"] == "old"

    def test_overwrites_entry_with_force(self, tmp_path: Path) -> None:
        target = tmp_path / "mcp.json"
        original = {
            "servers": {"research-pipeline": {"type": "stdio", "command": "old"}}
        }
        target.write_text(json.dumps(original) + "\n")

        updated = _update_vscode_mcp_config(target, force=True)

        assert updated is True
        data = json.loads(target.read_text())
        assert data["servers"]["research-pipeline"]["command"] == "research-pipeline"

    def test_recovers_from_corrupt_json(self, tmp_path: Path) -> None:
        target = tmp_path / "mcp.json"
        target.write_text("not valid json")

        updated = _update_vscode_mcp_config(target, force=False)

        assert updated is True
        data = json.loads(target.read_text())
        assert "research-pipeline" in data["servers"]

    def test_preserves_inputs_key(self, tmp_path: Path) -> None:
        target = tmp_path / "mcp.json"
        original = {
            "inputs": [{"id": "MY_KEY", "type": "promptString"}],
            "servers": {},
        }
        target.write_text(json.dumps(original) + "\n")

        _update_vscode_mcp_config(target, force=False)

        data = json.loads(target.read_text())
        assert data["inputs"] == [{"id": "MY_KEY", "type": "promptString"}], (
            "non-servers keys must be preserved"
        )

    def test_default_vscode_mcp_config_path(self) -> None:
        assert DEFAULT_VSCODE_MCP_CONFIG.name == "mcp.json"
        assert "Code" in str(DEFAULT_VSCODE_MCP_CONFIG)
        assert "User" in str(DEFAULT_VSCODE_MCP_CONFIG)
