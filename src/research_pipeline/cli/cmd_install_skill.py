"""Install the research-pipeline skill to the user's Claude skills directory."""

from __future__ import annotations

import contextlib
import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

# Default skill install target
DEFAULT_SKILL_DIR = Path.home() / ".claude" / "skills" / "research-pipeline"

# Skill source within the repo / installed package data
_SKILL_SOURCE_CANDIDATES = [
    # When running from the repo root (development)
    Path(__file__).resolve().parents[3] / ".github" / "skills" / "research-pipeline",
]


def _find_skill_source() -> Path | None:
    """Locate the bundled skill directory."""
    for candidate in _SKILL_SOURCE_CANDIDATES:
        if candidate.is_dir() and (candidate / "SKILL.md").is_file():
            return candidate

    # Fallback: use importlib.resources for installed package data
    with contextlib.suppress(Exception):
        import importlib.resources as pkg_resources

        ref = pkg_resources.files("research_pipeline") / "skill_data"
        # Convert traversable to concrete path
        ref_path = Path(str(ref))
        if ref_path.is_dir() and (ref_path / "SKILL.md").is_file():
            return ref_path

    return None


def run_install_skill(
    target: Path = DEFAULT_SKILL_DIR,
    symlink: bool = False,
    force: bool = False,
) -> None:
    """Copy or symlink the skill to the target directory.

    Args:
        target: Destination directory (default: ~/.claude/skills/research-pipeline).
        symlink: If True, create a symlink instead of copying.
        force: If True, overwrite existing skill directory.
    """
    source = _find_skill_source()
    if source is None:
        logger.error(
            "Could not locate skill source files. "
            "Ensure you are running from the repo or the package includes skill data."
        )
        raise SystemExit(1)

    logger.info("Skill source: %s", source)
    logger.info("Skill target: %s", target)

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

    # Ensure parent exists
    target.parent.mkdir(parents=True, exist_ok=True)

    if symlink:
        target.symlink_to(source)
        logger.info("Created symlink: %s → %s", target, source)
    else:
        shutil.copytree(source, target)
        logger.info("Copied skill to: %s", target)

    logger.info("Done. Skill installed at %s", target)
