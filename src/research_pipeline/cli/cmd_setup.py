"""Install research-pipeline skill and agents to the user's Claude config."""

from __future__ import annotations

import contextlib
import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

# Default install targets
DEFAULT_SKILL_DIR = Path.home() / ".claude" / "skills" / "research-pipeline"
DEFAULT_AGENTS_DIR = Path.home() / ".claude" / "agents"

# Source candidates when running from the repo root (development)
_REPO_ROOT = Path(__file__).resolve().parents[3]
_SKILL_SOURCE_CANDIDATES = [
    _REPO_ROOT / ".github" / "skills" / "research-pipeline",
]
_AGENT_SOURCE_CANDIDATES = [
    _REPO_ROOT / ".github" / "agents",
]


def _find_source(
    candidates: list[Path],
    package_subdir: str,
    marker_file: str | None = None,
) -> Path | None:
    """Locate a bundled data directory.

    Args:
        candidates: Filesystem paths to try first (development mode).
        package_subdir: Sub-package name under ``research_pipeline`` for
            installed-package fallback (e.g. ``"skill_data"``).
        marker_file: If set, the directory must contain this file to be valid.
    """
    for candidate in candidates:
        if candidate.is_dir() and (
            marker_file is None or (candidate / marker_file).is_file()
        ):
            return candidate

    # Fallback: use importlib.resources for installed package data
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
    # Repo source candidates point directly to the skill dir
    for candidate in _SKILL_SOURCE_CANDIDATES:
        if candidate.is_dir() and (candidate / "SKILL.md").is_file():
            return candidate

    # Fallback: installed package has skill_data/research-pipeline/
    with contextlib.suppress(Exception):
        import importlib.resources as pkg_resources

        ref = pkg_resources.files("research_pipeline") / "skill_data"
        ref_path = Path(str(ref)) / "research-pipeline"
        if ref_path.is_dir() and (ref_path / "SKILL.md").is_file():
            return ref_path

    return None


def _find_agent_source() -> Path | None:
    """Locate the bundled agents directory."""
    return _find_source(_AGENT_SOURCE_CANDIDATES, "agent_data")


def _install_directory(
    source: Path,
    target: Path,
    symlink: bool,
    force: bool,
    label: str,
) -> None:
    """Copy or symlink a source directory to a target.

    Args:
        source: Source directory.
        target: Destination directory.
        symlink: Create a symlink instead of copying.
        force: Overwrite existing target.
        label: Human-readable label for log messages.
    """
    logger.info("%s source: %s", label, source)
    logger.info("%s target: %s", label, target)

    if target.exists():
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


def run_setup(
    skill_target: Path = DEFAULT_SKILL_DIR,
    agents_target: Path = DEFAULT_AGENTS_DIR,
    symlink: bool = False,
    force: bool = False,
    skip_skill: bool = False,
    skip_agents: bool = False,
) -> None:
    """Install skill and agents to the user's Claude config.

    Args:
        skill_target: Destination for the skill directory.
        agents_target: Destination directory for agent files.
        symlink: If True, create symlinks instead of copying.
        force: If True, overwrite existing files/directories.
        skip_skill: If True, skip skill installation.
        skip_agents: If True, skip agent installation.
    """
    if skip_skill and skip_agents:
        logger.warning("Both --skip-skill and --skip-agents set; nothing to do.")
        return

    # --- Skill ---
    if not skip_skill:
        source = _find_skill_source()
        if source is None:
            logger.error(
                "Could not locate skill source files. "
                "Ensure you are running from the repo or the package "
                "includes skill data."
            )
            raise SystemExit(1)
        _install_directory(source, skill_target, symlink, force, "Skill")
        logger.info("Skill installed at %s", skill_target)

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

    logger.info("Done.")


# Backward compatibility
run_install_skill = run_setup
