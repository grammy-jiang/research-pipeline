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


# ---------------------------------------------------------------------------
# New tests: detection helpers, MCP CLI installs, path constants, suffix param
# ---------------------------------------------------------------------------


class TestDetection:
    """Tests for _detect_claude / _detect_codex / _detect_copilot."""

    def test_detect_claude_found(self) -> None:
        from unittest.mock import patch

        from research_pipeline.cli.cmd_setup import _detect_claude

        with patch("shutil.which", return_value="/usr/bin/claude"):
            assert _detect_claude() is True

    def test_detect_claude_not_found(self) -> None:
        from unittest.mock import patch

        from research_pipeline.cli.cmd_setup import _detect_claude

        with patch("shutil.which", return_value=None):
            assert _detect_claude() is False

    def test_detect_codex_found(self) -> None:
        from unittest.mock import patch

        from research_pipeline.cli.cmd_setup import _detect_codex

        with patch("shutil.which", return_value="/usr/bin/codex"):
            assert _detect_codex() is True

    def test_detect_codex_not_found(self) -> None:
        from unittest.mock import patch

        from research_pipeline.cli.cmd_setup import _detect_codex

        with patch("shutil.which", return_value=None):
            assert _detect_codex() is False

    def test_detect_copilot_found(self) -> None:
        from unittest.mock import patch

        from research_pipeline.cli.cmd_setup import _detect_copilot

        with patch("shutil.which", return_value="/usr/bin/copilot"):
            assert _detect_copilot() is True

    def test_detect_copilot_not_found(self) -> None:
        from unittest.mock import patch

        from research_pipeline.cli.cmd_setup import _detect_copilot

        with patch("shutil.which", return_value=None):
            assert _detect_copilot() is False


class TestFilterTargetsByDetection:
    """Tests for _filter_targets_by_detection."""

    def test_keeps_targets_for_detected_agents(self) -> None:
        from pathlib import Path
        from unittest.mock import patch

        from research_pipeline.cli.cmd_setup import (
            _AGENT_PATH_PREFIXES,
            _filter_targets_by_detection,
        )

        prefixes = list(_AGENT_PATH_PREFIXES.keys())
        assert len(prefixes) >= 3

        targets = [
            Path(prefixes[0]) / "skills" / "research-pipeline",
            Path(prefixes[1]) / "skills" / "research-pipeline",
            Path(prefixes[2]) / "skills" / "research-pipeline",
        ]
        with patch.dict(
            "research_pipeline.cli.cmd_setup._AGENT_PATH_PREFIXES",
            {
                prefixes[0]: lambda: True,
                prefixes[1]: lambda: False,
                prefixes[2]: lambda: True,
            },
        ):
            result = _filter_targets_by_detection(targets)
        assert targets[0] in result
        assert targets[1] not in result
        assert targets[2] in result

    def test_keeps_unknown_prefix_targets(self, tmp_path: Path) -> None:
        from research_pipeline.cli.cmd_setup import _filter_targets_by_detection

        custom = tmp_path / "custom" / "skills" / "research-pipeline"
        result = _filter_targets_by_detection([custom])
        assert custom in result


class TestClaudeMcp:
    """Tests for _install_claude_mcp."""

    def test_skips_when_already_registered(self) -> None:
        from unittest.mock import MagicMock, patch

        from research_pipeline.cli.cmd_setup import _install_claude_mcp

        mock_check = MagicMock()
        mock_check.returncode = 0
        with patch("subprocess.run", return_value=mock_check) as mock_run:
            result = _install_claude_mcp(force=False)
        assert result is False
        # Only one call: the 'get' check
        assert mock_run.call_count == 1

    def test_adds_when_not_registered(self) -> None:
        from unittest.mock import MagicMock, patch

        from research_pipeline.cli.cmd_setup import _install_claude_mcp

        not_found = MagicMock()
        not_found.returncode = 1
        added = MagicMock()
        added.returncode = 0
        added.stderr = ""
        with patch("subprocess.run", side_effect=[not_found, added]) as mock_run:
            result = _install_claude_mcp(force=False)
        assert result is True
        assert mock_run.call_count == 2

    def test_force_removes_then_adds(self) -> None:
        from unittest.mock import MagicMock, patch

        from research_pipeline.cli.cmd_setup import _install_claude_mcp

        removed = MagicMock()
        removed.returncode = 0
        added = MagicMock()
        added.returncode = 0
        added.stderr = ""
        with patch("subprocess.run", side_effect=[removed, added]) as mock_run:
            result = _install_claude_mcp(force=True)
        assert result is True
        first_call_args = mock_run.call_args_list[0][0][0]
        assert "remove" in first_call_args

    def test_returns_false_on_command_failure(self) -> None:
        from unittest.mock import MagicMock, patch

        from research_pipeline.cli.cmd_setup import _install_claude_mcp

        not_found = MagicMock()
        not_found.returncode = 1
        failed = MagicMock()
        failed.returncode = 1
        failed.stderr = "error"
        with patch("subprocess.run", side_effect=[not_found, failed]):
            result = _install_claude_mcp(force=False)
        assert result is False

    def test_returns_false_on_file_not_found(self) -> None:
        from unittest.mock import patch

        from research_pipeline.cli.cmd_setup import _install_claude_mcp

        with patch("subprocess.run", side_effect=FileNotFoundError):
            result = _install_claude_mcp(force=False)
        assert result is False


class TestCodexMcp:
    """Tests for _install_codex_mcp."""

    def test_skips_when_already_registered(self) -> None:
        from unittest.mock import MagicMock, patch

        from research_pipeline.cli.cmd_setup import _install_codex_mcp

        mock_check = MagicMock()
        mock_check.returncode = 0
        with patch("subprocess.run", return_value=mock_check) as mock_run:
            result = _install_codex_mcp(force=False)
        assert result is False
        assert mock_run.call_count == 1

    def test_adds_when_not_registered(self) -> None:
        from unittest.mock import MagicMock, patch

        from research_pipeline.cli.cmd_setup import _install_codex_mcp

        not_found = MagicMock()
        not_found.returncode = 1
        added = MagicMock()
        added.returncode = 0
        added.stderr = ""
        with patch("subprocess.run", side_effect=[not_found, added]) as mock_run:
            result = _install_codex_mcp(force=False)
        assert result is True
        assert mock_run.call_count == 2

    def test_force_removes_then_adds(self) -> None:
        from unittest.mock import MagicMock, patch

        from research_pipeline.cli.cmd_setup import _install_codex_mcp

        removed = MagicMock()
        removed.returncode = 0
        added = MagicMock()
        added.returncode = 0
        added.stderr = ""
        with patch("subprocess.run", side_effect=[removed, added]) as mock_run:
            result = _install_codex_mcp(force=True)
        assert result is True
        first_call_args = mock_run.call_args_list[0][0][0]
        assert "remove" in first_call_args

    def test_returns_false_on_timeout(self) -> None:
        import subprocess
        from unittest.mock import patch

        from research_pipeline.cli.cmd_setup import _install_codex_mcp

        with patch(
            "subprocess.run", side_effect=subprocess.TimeoutExpired("codex", 30)
        ):
            result = _install_codex_mcp(force=False)
        assert result is False


class TestDefaultCodexSkillDir:
    """DEFAULT_CODEX_SKILL_DIR must point to ~/.agents, not ~/.codex."""

    def test_path_uses_dot_agents(self) -> None:
        from research_pipeline.cli.cmd_setup import DEFAULT_CODEX_SKILL_DIR

        assert ".agents" in str(DEFAULT_CODEX_SKILL_DIR), (
            "Codex CLI skill path must be under ~/.agents (official user scope path)"
        )

    def test_path_includes_skills(self) -> None:
        from research_pipeline.cli.cmd_setup import DEFAULT_CODEX_SKILL_DIR

        assert "skills" in str(DEFAULT_CODEX_SKILL_DIR)


class TestCopilotAgentsDir:
    """DEFAULT_COPILOT_AGENTS_DIR must point to ~/.copilot/agents."""

    def test_path_uses_dot_copilot(self) -> None:
        from research_pipeline.cli.cmd_setup import DEFAULT_COPILOT_AGENTS_DIR

        assert ".copilot" in str(DEFAULT_COPILOT_AGENTS_DIR)

    def test_path_includes_agents(self) -> None:
        from research_pipeline.cli.cmd_setup import DEFAULT_COPILOT_AGENTS_DIR

        assert "agents" in str(DEFAULT_COPILOT_AGENTS_DIR)


class TestInstallAgentFilesWithSuffix:
    """_install_agent_files respects target_suffix kwarg."""

    def test_default_suffix_md(self, tmp_path: Path) -> None:
        from research_pipeline.cli.cmd_setup import _install_agent_files

        src = tmp_path / "src"
        src.mkdir()
        (src / "paper-analyzer.md").write_text("# Agent")

        dest = tmp_path / "dest"
        count = _install_agent_files(src, dest, symlink=False, force=False)
        assert count == 1
        assert (dest / "paper-analyzer.md").exists()

    def test_agent_md_suffix_for_copilot(self, tmp_path: Path) -> None:
        from research_pipeline.cli.cmd_setup import _install_agent_files

        src = tmp_path / "src"
        src.mkdir()
        (src / "paper-analyzer.md").write_text("# Agent")

        dest = tmp_path / "dest"
        count = _install_agent_files(
            src, dest, symlink=False, force=False, target_suffix=".agent.md"
        )
        assert count == 1
        assert (dest / "paper-analyzer.agent.md").exists()
        assert not (dest / "paper-analyzer.md").exists()

    def test_force_overwrites_with_new_suffix(self, tmp_path: Path) -> None:
        from research_pipeline.cli.cmd_setup import _install_agent_files

        src = tmp_path / "src"
        src.mkdir()
        (src / "paper-analyzer.md").write_text("# Agent v2")

        dest = tmp_path / "dest"
        dest.mkdir()
        existing = dest / "paper-analyzer.agent.md"
        existing.write_text("old")

        count = _install_agent_files(
            src, dest, symlink=False, force=True, target_suffix=".agent.md"
        )
        assert count == 1
        assert existing.read_text() == "# Agent v2"


class TestAgentDetectionGating:
    """Agents under known home prefixes are skipped when the agent is absent."""

    def _make_sources(self, tmp_path: Path) -> tuple[Path, Path]:
        skill_src = tmp_path / "skill_src"
        skill_src.mkdir()
        (skill_src / "SKILL.md").write_text("# Skill")
        agent_src = tmp_path / "agent_src"
        agent_src.mkdir()
        (agent_src / "paper-analyzer.md").write_text("# Agent")
        return skill_src, agent_src

    def test_skips_claude_agents_when_claude_not_detected(self, tmp_path: Path) -> None:
        """When run in default mode with a real ~/.claude path and no claude
        binary, _install_agent_files should not be called."""
        from pathlib import Path as _Path
        from unittest.mock import MagicMock, patch

        from research_pipeline.cli.cmd_setup import run_setup

        skill_src, agent_src = self._make_sources(tmp_path)
        home = _Path.home()
        agents_target = home / ".claude" / "agents"
        mcp_target = tmp_path / "mcp.json"
        mock_install = MagicMock(return_value=0)
        # Patch the dict directly — it holds captured function references so
        # patching _detect_claude alone would not update the in-dict callable.
        fake_prefixes = {
            str(home / ".claude"): lambda: False,
            str(home / ".agents"): lambda: True,
            str(home / ".copilot"): lambda: True,
        }

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
                (tmp_path / "skill_target",),
            ),
            patch(
                "research_pipeline.cli.cmd_setup._AGENT_PATH_PREFIXES",
                fake_prefixes,
            ),
            patch(
                "research_pipeline.cli.cmd_setup._install_agent_files",
                mock_install,
            ),
            patch(
                "research_pipeline.cli.cmd_setup._update_copilot_mcp_config",
            ),
            patch(
                "research_pipeline.cli.cmd_setup._update_vscode_mcp_config",
            ),
        ):
            run_setup(
                agents_target=agents_target,
                mcp_config_target=mcp_target,
                force=True,
            )

        mock_install.assert_not_called()

    def test_installs_claude_agents_when_claude_detected(self, tmp_path: Path) -> None:
        from pathlib import Path as _Path
        from unittest.mock import MagicMock, patch

        from research_pipeline.cli.cmd_setup import run_setup

        skill_src, agent_src = self._make_sources(tmp_path)
        home = _Path.home()
        agents_target = home / ".claude" / "agents"
        mcp_target = tmp_path / "mcp.json"
        mock_install = MagicMock(return_value=1)
        fake_prefixes = {
            str(home / ".claude"): lambda: True,
            str(home / ".agents"): lambda: True,
            str(home / ".copilot"): lambda: True,
        }

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
                (tmp_path / "skill_target",),
            ),
            patch(
                "research_pipeline.cli.cmd_setup._AGENT_PATH_PREFIXES",
                fake_prefixes,
            ),
            patch(
                "research_pipeline.cli.cmd_setup._install_agent_files",
                mock_install,
            ),
            patch(
                "research_pipeline.cli.cmd_setup._update_copilot_mcp_config",
            ),
            patch(
                "research_pipeline.cli.cmd_setup._update_vscode_mcp_config",
            ),
        ):
            run_setup(
                agents_target=agents_target,
                mcp_config_target=mcp_target,
                force=True,
            )

        mock_install.assert_called_once()

    def test_explicit_path_always_installs_regardless_of_detection(
        self, tmp_path: Path
    ) -> None:
        """An explicit (non-default) agents_target bypasses detection gating."""
        from unittest.mock import patch

        from research_pipeline.cli.cmd_setup import run_setup

        skill_src, agent_src = self._make_sources(tmp_path)
        agents_target = tmp_path / "custom_agents"
        mcp_target = tmp_path / "mcp.json"

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
                "research_pipeline.cli.cmd_setup._update_copilot_mcp_config",
            ),
            patch(
                "research_pipeline.cli.cmd_setup._update_vscode_mcp_config",
            ),
        ):
            # Pass explicit skill_target to avoid default-multi-target mode
            run_setup(
                skill_target=tmp_path / "skill_target",
                agents_target=agents_target,
                mcp_config_target=mcp_target,
                force=True,
            )

        assert (agents_target / "paper-analyzer.md").exists()


class TestMcpDetectionGating:
    """Copilot and VS Code MCP configs respect detection in default mode."""

    def _make_sources(self, tmp_path: Path) -> tuple[Path, Path]:
        skill_src = tmp_path / "skill_src"
        skill_src.mkdir()
        (skill_src / "SKILL.md").write_text("# Skill")
        agent_src = tmp_path / "agent_src"
        agent_src.mkdir()
        (agent_src / "paper-analyzer.md").write_text("# Agent")
        return skill_src, agent_src

    def test_copilot_mcp_skipped_when_not_detected(self, tmp_path: Path) -> None:
        from unittest.mock import MagicMock, patch

        from research_pipeline.cli.cmd_setup import run_setup

        skill_src, agent_src = self._make_sources(tmp_path)
        mock_copilot = MagicMock()

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
                (tmp_path / "skill_target",),
            ),
            patch(
                "research_pipeline.cli.cmd_setup._detect_copilot",
                return_value=False,
            ),
            patch(
                "research_pipeline.cli.cmd_setup._update_copilot_mcp_config",
                mock_copilot,
            ),
            patch(
                "research_pipeline.cli.cmd_setup._update_vscode_mcp_config",
            ),
            patch("shutil.which", return_value=None),
        ):
            run_setup(
                agents_target=tmp_path / "agents",
                mcp_config_target=tmp_path / "mcp.json",
                force=True,
            )

        mock_copilot.assert_not_called()

    def test_copilot_mcp_installed_when_detected(self, tmp_path: Path) -> None:
        from unittest.mock import MagicMock, patch

        from research_pipeline.cli.cmd_setup import run_setup

        skill_src, agent_src = self._make_sources(tmp_path)
        mock_copilot = MagicMock()

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
                (tmp_path / "skill_target",),
            ),
            patch(
                "research_pipeline.cli.cmd_setup._detect_copilot",
                return_value=True,
            ),
            patch(
                "research_pipeline.cli.cmd_setup._update_copilot_mcp_config",
                mock_copilot,
            ),
            patch(
                "research_pipeline.cli.cmd_setup._update_vscode_mcp_config",
            ),
        ):
            run_setup(
                agents_target=tmp_path / "agents",
                mcp_config_target=tmp_path / "mcp.json",
                force=True,
            )

        mock_copilot.assert_called_once()

    def test_copilot_mcp_always_installed_when_explicit_target(
        self, tmp_path: Path
    ) -> None:
        """Explicit skill_target disables default mode.

        Copilot MCP runs unconditionally in explicit mode.
        """
        from unittest.mock import MagicMock, patch

        from research_pipeline.cli.cmd_setup import run_setup

        skill_src, agent_src = self._make_sources(tmp_path)
        mock_copilot = MagicMock()

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
                "research_pipeline.cli.cmd_setup._update_copilot_mcp_config",
                mock_copilot,
            ),
            patch(
                "research_pipeline.cli.cmd_setup._update_vscode_mcp_config",
            ),
        ):
            run_setup(
                skill_target=tmp_path / "skill_target",
                agents_target=tmp_path / "agents",
                mcp_config_target=tmp_path / "mcp.json",
                force=True,
            )

        mock_copilot.assert_called_once()


# ---------------------------------------------------------------------------
# Tests: _find_claude_hooks_source, _install_claude_hooks, skip_hooks
# ---------------------------------------------------------------------------


class TestFindClaudeHooksSource:
    """Tests for _find_claude_hooks_source."""

    def test_returns_none_or_existing_path(self) -> None:
        from research_pipeline.cli.cmd_setup import _find_claude_hooks_source

        result = _find_claude_hooks_source()
        if result is not None:
            assert result.is_file()
            assert result.name == "claude-code-hooks.json"
            assert "hooks" in str(result)

    def test_returns_none_when_skill_source_missing(self) -> None:
        from unittest.mock import patch

        from research_pipeline.cli.cmd_setup import _find_claude_hooks_source

        with patch(
            "research_pipeline.cli.cmd_setup._find_skill_source", return_value=None
        ):
            assert _find_claude_hooks_source() is None

    def test_returns_none_when_hooks_file_absent(self, tmp_path: Path) -> None:
        from unittest.mock import patch

        from research_pipeline.cli.cmd_setup import _find_claude_hooks_source

        fake_skill = tmp_path / "research-pipeline"
        fake_skill.mkdir()
        (fake_skill / "SKILL.md").write_text("# skill")

        with patch(
            "research_pipeline.cli.cmd_setup._find_skill_source",
            return_value=fake_skill,
        ):
            assert _find_claude_hooks_source() is None


class TestInstallClaudeHooks:
    """Tests for _install_claude_hooks."""

    _STOP_CMD = "~/.claude/skills/research-pipeline/hooks/stop-check.sh"
    _RESUME_CMD = "~/.claude/skills/research-pipeline/hooks/resume-inject.sh"

    @property
    def _HOOKS_PAYLOAD(self) -> dict:  # type: ignore[type-arg]
        return {
            "hooks": {
                "Stop": [{"hooks": [{"type": "command", "command": self._STOP_CMD}]}],
                "UserPromptSubmit": [
                    {"hooks": [{"type": "command", "command": self._RESUME_CMD}]}
                ],
            }
        }

    def _make_hooks_source(self, tmp_path: Path) -> Path:
        f = tmp_path / "claude-code-hooks.json"
        f.write_text(json.dumps(self._HOOKS_PAYLOAD))
        return f

    def test_creates_settings_file_with_both_events(self, tmp_path: Path) -> None:
        from research_pipeline.cli.cmd_setup import _install_claude_hooks

        settings = tmp_path / ".claude" / "settings.json"
        hooks_src = self._make_hooks_source(tmp_path)

        result = _install_claude_hooks(settings, hooks_src, force=False)

        assert result is True
        data = json.loads(settings.read_text())
        assert "Stop" in data["hooks"]
        assert "UserPromptSubmit" in data["hooks"]
        assert len(data["hooks"]["Stop"]) == 1
        assert len(data["hooks"]["UserPromptSubmit"]) == 1

    def test_merges_preserving_unrelated_hooks(self, tmp_path: Path) -> None:
        from research_pipeline.cli.cmd_setup import _install_claude_hooks

        settings = tmp_path / "settings.json"
        settings.write_text(
            json.dumps(
                {
                    "hooks": {
                        "PreToolUse": [
                            {"hooks": [{"type": "command", "command": "other-cmd"}]}
                        ]
                    }
                }
            )
        )
        hooks_src = self._make_hooks_source(tmp_path)

        result = _install_claude_hooks(settings, hooks_src, force=False)

        assert result is True
        data = json.loads(settings.read_text())
        assert "PreToolUse" in data["hooks"], "unrelated hooks must be preserved"
        assert "Stop" in data["hooks"]
        assert "UserPromptSubmit" in data["hooks"]

    def test_skips_when_already_registered(self, tmp_path: Path) -> None:
        from research_pipeline.cli.cmd_setup import _install_claude_hooks

        settings = tmp_path / "settings.json"
        settings.write_text(
            json.dumps(
                {
                    "hooks": {
                        "Stop": [
                            {"hooks": [{"type": "command", "command": self._STOP_CMD}]}
                        ],
                        "UserPromptSubmit": [
                            {
                                "hooks": [
                                    {"type": "command", "command": self._RESUME_CMD}
                                ]
                            }
                        ],
                    }
                }
            )
        )
        hooks_src = self._make_hooks_source(tmp_path)
        mtime_before = settings.stat().st_mtime

        result = _install_claude_hooks(settings, hooks_src, force=False)

        assert result is False
        assert settings.stat().st_mtime == mtime_before, "file must not be rewritten"
        data = json.loads(settings.read_text())
        assert len(data["hooks"]["Stop"]) == 1
        assert len(data["hooks"]["UserPromptSubmit"]) == 1

    def test_replaces_with_force(self, tmp_path: Path) -> None:
        from research_pipeline.cli.cmd_setup import _install_claude_hooks

        settings = tmp_path / "settings.json"
        settings.write_text(
            json.dumps(
                {
                    "hooks": {
                        "Stop": [
                            {"hooks": [{"type": "command", "command": self._STOP_CMD}]}
                        ]
                    }
                }
            )
        )
        hooks_src = self._make_hooks_source(tmp_path)

        result = _install_claude_hooks(settings, hooks_src, force=True)

        assert result is True
        data = json.loads(settings.read_text())
        assert len(data["hooks"]["Stop"]) == 1

    def test_returns_false_for_missing_hooks_source(self, tmp_path: Path) -> None:
        from research_pipeline.cli.cmd_setup import _install_claude_hooks

        settings = tmp_path / "settings.json"
        missing = tmp_path / "nonexistent.json"

        result = _install_claude_hooks(settings, missing, force=False)

        assert result is False
        assert not settings.exists()

    def test_recovers_from_corrupt_settings_file(self, tmp_path: Path) -> None:
        from research_pipeline.cli.cmd_setup import _install_claude_hooks

        settings = tmp_path / "settings.json"
        settings.write_text("not valid json {{")
        hooks_src = self._make_hooks_source(tmp_path)

        result = _install_claude_hooks(settings, hooks_src, force=False)

        assert result is True
        data = json.loads(settings.read_text())
        assert "Stop" in data["hooks"]
        assert "UserPromptSubmit" in data["hooks"]


class TestRunSetupHooks:
    """Tests for skip_hooks parameter and default-mode hook installation."""

    def _make_sources(self, tmp_path: Path) -> tuple[Path, Path]:
        skill_src = tmp_path / "skill_src"
        skill_src.mkdir()
        (skill_src / "SKILL.md").write_text("# Skill")
        agent_src = tmp_path / "agent_src"
        agent_src.mkdir()
        (agent_src / "paper-analyzer.md").write_text("# Analyzer")
        return skill_src, agent_src

    def test_skip_hooks_prevents_hook_install_in_default_mode(
        self, tmp_path: Path
    ) -> None:
        from unittest.mock import MagicMock, patch

        from research_pipeline.cli.cmd_setup import run_setup

        skill_src, agent_src = self._make_sources(tmp_path)
        claude_target = tmp_path / "claude" / "skills" / "research-pipeline"
        mock_install_hooks = MagicMock()

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
                (claude_target,),
            ),
            patch(
                "research_pipeline.cli.cmd_setup._install_claude_hooks",
                mock_install_hooks,
            ),
            patch("research_pipeline.cli.cmd_setup._update_copilot_mcp_config"),
            patch("research_pipeline.cli.cmd_setup._update_vscode_mcp_config"),
        ):
            run_setup(
                mcp_config_target=tmp_path / "mcp.json",
                skip_hooks=True,
            )

        mock_install_hooks.assert_not_called()

    def test_default_mode_installs_hooks_when_claude_detected(
        self, tmp_path: Path
    ) -> None:
        from unittest.mock import MagicMock, patch

        from research_pipeline.cli.cmd_setup import run_setup

        skill_src, agent_src = self._make_sources(tmp_path)
        hooks_src = tmp_path / "claude-code-hooks.json"
        hooks_src.write_text(json.dumps({"hooks": {"Stop": []}}))
        claude_target = tmp_path / "claude" / "skills" / "research-pipeline"
        mock_install_hooks = MagicMock(return_value=True)

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
                (claude_target,),
            ),
            patch(
                "research_pipeline.cli.cmd_setup._find_claude_hooks_source",
                return_value=hooks_src,
            ),
            patch("research_pipeline.cli.cmd_setup._detect_claude", return_value=True),
            patch(
                "research_pipeline.cli.cmd_setup._install_claude_hooks",
                mock_install_hooks,
            ),
            patch("research_pipeline.cli.cmd_setup._update_copilot_mcp_config"),
            patch("research_pipeline.cli.cmd_setup._update_vscode_mcp_config"),
        ):
            run_setup(
                mcp_config_target=tmp_path / "mcp.json",
                skip_hooks=False,
            )

        mock_install_hooks.assert_called_once()

    def test_explicit_skill_target_skips_hooks(self, tmp_path: Path) -> None:
        """Hooks are only installed in default mode, not explicit-target mode."""
        from unittest.mock import MagicMock, patch

        from research_pipeline.cli.cmd_setup import run_setup

        skill_src, agent_src = self._make_sources(tmp_path)
        mock_install_hooks = MagicMock()

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
                "research_pipeline.cli.cmd_setup._install_claude_hooks",
                mock_install_hooks,
            ),
        ):
            run_setup(
                skill_target=tmp_path / "skill",
                agents_target=tmp_path / "agents",
                mcp_config_target=tmp_path / "mcp.json",
                skip_hooks=False,
            )

        mock_install_hooks.assert_not_called()

    def test_default_claude_settings_file_path(self) -> None:
        from research_pipeline.cli.cmd_setup import DEFAULT_CLAUDE_SETTINGS_FILE

        assert DEFAULT_CLAUDE_SETTINGS_FILE.name == "settings.json"
        assert ".claude" in str(DEFAULT_CLAUDE_SETTINGS_FILE)
