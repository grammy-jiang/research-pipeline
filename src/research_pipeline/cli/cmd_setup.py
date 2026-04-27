"""Install research-pipeline skill and agents to assistant config directories."""

from __future__ import annotations

import contextlib
import json
import logging
import shutil
from collections.abc import Iterable, Sequence
from pathlib import Path

logger = logging.getLogger(__name__)

# Default install targets
DEFAULT_CLAUDE_SKILL_DIR = Path.home() / ".claude" / "skills" / "research-pipeline"
DEFAULT_CODEX_SKILL_DIR = Path.home() / ".codex" / "skills" / "research-pipeline"
DEFAULT_SKILL_DIR = DEFAULT_CLAUDE_SKILL_DIR
DEFAULT_SKILL_TARGETS = (DEFAULT_CLAUDE_SKILL_DIR, DEFAULT_CODEX_SKILL_DIR)
DEFAULT_AGENTS_DIR = Path.home() / ".claude" / "agents"
DEFAULT_MCP_CONFIG_DIR = Path.home() / ".config" / "research-pipeline"
DEFAULT_MCP_CONFIG_FILE = DEFAULT_MCP_CONFIG_DIR / "mcp.json"


def _find_source(
    package_subdir: str,
    marker_file: str | None = None,
) -> Path | None:
    """Locate a bundled data directory.

    Args:
        package_subdir: Sub-package name under ``research_pipeline`` for
            installed-package fallback (e.g. ``"skill_data"``).
        marker_file: If set, the directory must contain this file to be valid.
    """
    # Use package data in both installed and editable/development modes so setup
    # installs exactly the files that ship in the wheel.
    with contextlib.suppress(Exception):
        import importlib.resources as pkg_resources

        ref = pkg_resources.files("research_pipeline") / package_subdir
        ref_path = Path(str(ref))
        if ref_path.is_dir() and (
            marker_file is None or (ref_path / marker_file).is_file()
        ):
            return ref_path

    return None


def _find_skill_source() -> Path | None:
    """Locate the bundled skill directory."""
    with contextlib.suppress(Exception):
        import importlib.resources as pkg_resources

        ref = pkg_resources.files("research_pipeline") / "skill_data"
        ref_path = Path(str(ref)) / "research-pipeline"
        if ref_path.is_dir() and (ref_path / "SKILL.md").is_file():
            return ref_path

    return None


def _find_skill_sources() -> list[Path]:
    """Locate all bundled skill directories."""
    source = _find_skill_source()
    if source is None:
        return []
    root = source.parent
    return [
        path
        for path in sorted(root.iterdir())
        if path.is_dir() and (path / "SKILL.md").is_file()
    ]


def _find_agent_source() -> Path | None:
    """Locate the bundled agents directory."""
    return _find_source("agent_data")


def _dedupe_paths(paths: Iterable[Path]) -> list[Path]:
    """Return paths in input order, removing duplicates."""
    deduped: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        expanded = path.expanduser()
        key = str(expanded)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(expanded)
    return deduped


def _resolve_skill_targets(
    skill_target: Path | None,
    skill_targets: Sequence[Path] | None,
) -> list[Path]:
    """Resolve explicit or default skill installation targets."""
    if skill_targets is not None:
        return _dedupe_paths(skill_targets)
    if skill_target is not None:
        return _dedupe_paths([skill_target])
    return _dedupe_paths(DEFAULT_SKILL_TARGETS)


def _install_directory(
    source: Path,
    target: Path,
    symlink: bool,
    force: bool,
    label: str,
    skip_existing: bool = False,
) -> bool:
    """Copy or symlink a source directory to a target.

    Args:
        source: Source directory.
        target: Destination directory.
        symlink: Create a symlink instead of copying.
        force: Overwrite existing target.
        label: Human-readable label for log messages.
        skip_existing: If True, leave existing targets untouched instead of
            failing when force is False.

    Returns:
        True if the target was installed or overwritten, False if skipped.
    """
    logger.info("%s source: %s", label, source)
    logger.info("%s target: %s", label, target)

    if target.exists() or target.is_symlink():
        if skip_existing and not force:
            logger.warning(
                "%s target already exists: %s. Use --force to overwrite.",
                label,
                target,
            )
            return False
        if target.is_symlink():
            existing_target = target.resolve()
            if not force:
                logger.error(
                    "Target already exists (symlink → %s). Use --force to overwrite.",
                    existing_target,
                )
                raise SystemExit(1)
            logger.info("Removing existing symlink: %s", target)
            target.unlink()
        elif target.is_dir():
            if not force:
                logger.error(
                    "Target already exists: %s. Use --force to overwrite.", target
                )
                raise SystemExit(1)
            logger.info("Removing existing directory: %s", target)
            shutil.rmtree(target)
        elif target.is_file():
            if not force:
                logger.error(
                    "Target already exists: %s. Use --force to overwrite.", target
                )
                raise SystemExit(1)
            logger.info("Removing existing file: %s", target)
            target.unlink()

    target.parent.mkdir(parents=True, exist_ok=True)

    if symlink:
        target.symlink_to(source)
        logger.info("Created symlink: %s → %s", target, source)
    else:
        shutil.copytree(source, target)
        logger.info("Copied %s to: %s", label.lower(), target)
    return True


def _install_agent_files(
    source_dir: Path,
    target_dir: Path,
    symlink: bool,
    force: bool,
) -> int:
    """Install individual agent .md files into the target directory.

    Unlike skills (which are a directory), agents are individual files
    that live directly in ``~/.claude/agents/``.

    Returns:
        Number of agent files installed.
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for agent_file in sorted(source_dir.glob("*.md")):
        dest = target_dir / agent_file.name
        logger.info("Agent file: %s → %s", agent_file.name, dest)

        if dest.exists() or dest.is_symlink():
            if not force:
                logger.warning(
                    "Agent %s already exists. Use --force to overwrite.",
                    dest,
                )
                continue
            if dest.is_symlink():
                dest.unlink()
            else:
                dest.unlink()

        if symlink:
            dest.symlink_to(agent_file)
            logger.info("Created symlink: %s → %s", dest, agent_file)
        else:
            shutil.copy2(agent_file, dest)
            logger.info("Copied agent: %s", dest)
        count += 1
    return count


def _mcp_server_config() -> dict[str, dict[str, dict[str, object]]]:
    """Return a generic MCP client configuration for research-pipeline."""
    from research_pipeline.cli.cmd_mcp import mcp_server_config

    return mcp_server_config()


def _install_mcp_config(target: Path, force: bool) -> bool:
    """Install a reusable MCP client configuration snippet.

    The file is intentionally written to research-pipeline's own config
    directory rather than mutating client-specific JSON/TOML files with
    unknown schemas. Local agents can import or copy this zero-argument
    server definition, and docs show the same command pair.
    """
    if target.exists() and not force:
        logger.warning(
            "MCP config already exists: %s. Use --force to overwrite.",
            target,
        )
        return False

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(_mcp_server_config(), indent=2) + "\n")
    logger.info("MCP config installed at %s", target)
    return True


def run_setup(
    skill_target: Path | None = None,
    skill_targets: Sequence[Path] | None = None,
    agents_target: Path = DEFAULT_AGENTS_DIR,
    mcp_config_target: Path = DEFAULT_MCP_CONFIG_FILE,
    symlink: bool = False,
    force: bool = False,
    skip_skill: bool = False,
    skip_agents: bool = False,
    skip_mcp: bool = False,
) -> None:
    """Install skills and agents to assistant config directories.

    Args:
        skill_target: Explicit single destination for the skill directory.
            When omitted, installs to Claude/GitHub Copilot and Codex paths.
        skill_targets: Explicit multiple destinations for the skill directory.
            Takes precedence over ``skill_target`` when set.
        agents_target: Destination directory for agent files.
        symlink: If True, create symlinks instead of copying.
        force: If True, overwrite existing files/directories.
        skip_skill: If True, skip skill installation.
        skip_agents: If True, skip agent installation.
        skip_mcp: If True, skip MCP config snippet installation.
    """
    if skip_skill and skip_agents and skip_mcp:
        logger.warning("All setup components skipped; nothing to do.")
        return

    # --- Skill ---
    if not skip_skill:
        targets = _resolve_skill_targets(skill_target, skill_targets)
        default_multi_target = skill_target is None and skill_targets is None
        primary_skill_source = _find_skill_source()
        skill_sources = (
            _find_skill_sources()
            if default_multi_target
            else ([primary_skill_source] if primary_skill_source is not None else [])
        )
        if not skill_sources:
            logger.error(
                "Could not locate skill source files. "
                "Ensure you are running from the repo or the package "
                "includes skill data."
            )
            raise SystemExit(1)

        if not targets:
            logger.warning("No skill targets configured; skipping skill install.")
        fanout_skill_targets = default_multi_target and len(skill_sources) > 1
        for skill_source in skill_sources:
            for target in targets:
                install_target = (
                    target.parent / skill_source.name
                    if fanout_skill_targets
                    else target
                )
                installed = _install_directory(
                    skill_source,
                    install_target,
                    symlink,
                    force,
                    "Skill",
                    skip_existing=default_multi_target,
                )
                if installed:
                    logger.info("Skill installed at %s", install_target)

    # --- Agents ---
    if not skip_agents:
        source = _find_agent_source()
        if source is None:
            logger.error(
                "Could not locate agent source files. "
                "Ensure you are running from the repo or the package "
                "includes agent data."
            )
            raise SystemExit(1)
        count = _install_agent_files(source, agents_target, symlink, force)
        logger.info("Installed %d agent(s) to %s", count, agents_target)

    # --- MCP server config ---
    if not skip_mcp:
        _install_mcp_config(mcp_config_target, force=force)

    logger.info("Done.")


# Backward compatibility
run_install_skill = run_setup
