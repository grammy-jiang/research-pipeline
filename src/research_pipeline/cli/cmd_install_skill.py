"""Backward-compatible install-skill helpers.

The public implementation lives in ``cmd_setup`` so skill installation uses
the same packaged source files as the unified setup command.
"""

from __future__ import annotations

from pathlib import Path

from research_pipeline.cli.cmd_setup import (
    DEFAULT_SKILL_DIR,
    _find_skill_source,
    run_setup,
)

__all__ = ["DEFAULT_SKILL_DIR", "_find_skill_source", "run_install_skill"]


def run_install_skill(
    target: Path = DEFAULT_SKILL_DIR,
    symlink: bool = False,
    force: bool = False,
) -> None:
    """Install only the packaged skill files."""
    run_setup(
        skill_target=target,
        symlink=symlink,
        force=force,
        skip_agents=True,
        skip_mcp=True,
    )
