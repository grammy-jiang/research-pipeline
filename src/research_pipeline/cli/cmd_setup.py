"""Install research-pipeline skill and agents to assistant config directories."""

from __future__ import annotations

import contextlib
import json
import logging
import shutil
import subprocess  # nosec B404
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path

logger = logging.getLogger(__name__)

# Default install targets
DEFAULT_CLAUDE_SKILL_DIR = Path.home() / ".claude" / "skills" / "research-pipeline"
# Official Codex CLI skill path: ~/.agents/skills/<name>  (not ~/.codex/skills/)
DEFAULT_CODEX_SKILL_DIR = Path.home() / ".agents" / "skills" / "research-pipeline"
DEFAULT_COPILOT_SKILL_DIR = Path.home() / ".copilot" / "skills" / "research-pipeline"
DEFAULT_SKILL_DIR = DEFAULT_CLAUDE_SKILL_DIR
DEFAULT_SKILL_TARGETS = (
    DEFAULT_CLAUDE_SKILL_DIR,
    DEFAULT_CODEX_SKILL_DIR,
    DEFAULT_COPILOT_SKILL_DIR,
)
DEFAULT_AGENTS_DIR = Path.home() / ".claude" / "agents"
DEFAULT_COPILOT_AGENTS_DIR = Path.home() / ".copilot" / "agents"
DEFAULT_MCP_CONFIG_DIR = Path.home() / ".config" / "research-pipeline"
DEFAULT_MCP_CONFIG_FILE = DEFAULT_MCP_CONFIG_DIR / "mcp.json"
DEFAULT_COPILOT_MCP_CONFIG = Path.home() / ".copilot" / "mcp-config.json"
DEFAULT_VSCODE_MCP_CONFIG = Path.home() / ".config" / "Code" / "User" / "mcp.json"
DEFAULT_CLAUDE_SETTINGS_FILE = Path.home() / ".claude" / "settings.json"


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


def _find_claude_hooks_source() -> Path | None:
    """Locate the bundled Claude Code hooks config for the primary skill."""
    source = _find_skill_source()
    if source is None:
        return None
    candidate = source / "hooks" / "claude-code-hooks.json"
    return candidate if candidate.is_file() else None


def _install_claude_hooks(
    settings_file: Path,
    hooks_source: Path,
    force: bool,
) -> bool:
    """Merge research-pipeline lifecycle hooks into a Claude Code settings.json.

    Reads ``hooks_source`` (a claude-code-hooks.json file) and merges the
    ``hooks`` block into ``settings_file`` without overwriting unrelated keys.
    Deduplicates by command string: if any of our commands are already
    registered for the same event, the function skips that event (unless
    ``force`` is True, in which case it replaces the existing entries).

    Args:
        settings_file: Path to ``~/.claude/settings.json`` or a project-local
            ``.claude/settings.json``.
        hooks_source: Path to the bundled ``claude-code-hooks.json`` config.
        force: If True, replace existing entries that match our commands.

    Returns:
        True if the settings file was written (new or updated), False otherwise.
    """
    if not hooks_source.is_file():
        logger.warning("Claude hooks source not found: %s", hooks_source)
        return False

    try:
        new_hooks: dict[str, list[object]] = json.loads(hooks_source.read_text()).get(
            "hooks", {}
        )
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Could not parse hooks source %s: %s", hooks_source, exc)
        return False

    if not new_hooks:
        return False

    def _extract_commands(matchers: list[object]) -> set[str]:
        cmds: set[str] = set()
        for matcher in matchers:
            if not isinstance(matcher, dict):
                continue
            for hook in matcher.get("hooks", []):
                if isinstance(hook, dict) and "command" in hook:
                    cmds.add(str(hook["command"]))
        return cmds

    existing: dict[str, object] = {}
    if settings_file.is_file():
        try:
            existing = json.loads(settings_file.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning(
                "Could not parse %s: %s — starting fresh.", settings_file, exc
            )

    existing_hooks: dict[str, list[object]] = existing.setdefault(  # type: ignore[assignment]
        "hooks", {}
    )
    changed = False
    for event_name, new_matchers in new_hooks.items():
        our_cmds = _extract_commands(new_matchers)
        existing_matchers: list[object] = existing_hooks.get(event_name, [])
        already_present = bool(our_cmds & _extract_commands(existing_matchers))

        if already_present and not force:
            logger.info(
                "Hook '%s' already registered; skipping (use --force to replace).",
                event_name,
            )
            continue

        if force and already_present:
            existing_hooks[event_name] = [
                m for m in existing_matchers if not (_extract_commands([m]) & our_cmds)
            ] + list(new_matchers)
        else:
            existing_hooks[event_name] = list(existing_matchers) + list(new_matchers)
        changed = True

    if not changed:
        return False

    settings_file.parent.mkdir(parents=True, exist_ok=True)
    settings_file.write_text(json.dumps(existing, indent=2) + "\n")
    logger.info("Claude Code hooks updated in %s", settings_file)
    return True


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


def _detect_claude() -> bool:
    """Return True if the Claude Code CLI (``claude``) is available on PATH."""
    return shutil.which("claude") is not None


def _detect_codex() -> bool:
    """Return True if the Codex CLI (``codex``) is available on PATH."""
    return shutil.which("codex") is not None


def _detect_copilot() -> bool:
    """Return True if the GitHub Copilot CLI (``copilot``) is available on PATH."""
    return shutil.which("copilot") is not None


# Map each agent's home-directory prefix to its detection callable.
# A skill target whose path starts with a given prefix is only installed when
# the associated detector returns True.  Targets matching no prefix are kept
# unconditionally (e.g. explicit, test, or custom paths).
_AGENT_PATH_PREFIXES: dict[str, Callable[[], bool]] = {
    str(Path.home() / ".claude"): _detect_claude,
    str(Path.home() / ".agents"): _detect_codex,
    str(Path.home() / ".copilot"): _detect_copilot,
}


def _filter_targets_by_detection(targets: list[Path]) -> list[Path]:
    """Keep only skill targets whose associated agent is detected on PATH.

    Targets under a known agent home prefix are skipped when that agent is not
    installed.  Targets that match no known prefix are always kept.

    Args:
        targets: Fully-expanded candidate install paths.

    Returns:
        Filtered list preserving the original order.
    """
    kept: list[Path] = []
    for target in targets:
        t_str = str(target)
        matched = False
        for prefix, detector in _AGENT_PATH_PREFIXES.items():
            if t_str.startswith(prefix + "/") or t_str == prefix:
                matched = True
                if detector():
                    kept.append(target)
                    logger.debug(
                        "Agent detected (prefix %s); including target: %s",
                        prefix,
                        target,
                    )
                else:
                    logger.info(
                        "Agent not detected (prefix %s); skipping target: %s",
                        prefix,
                        target,
                    )
                break
        if not matched:
            kept.append(target)
    return kept


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
    *,
    target_suffix: str = ".md",
) -> int:
    """Install individual agent files into the target directory.

    Unlike skills (which are a directory), agents are individual files
    that live directly in the agent's agents directory.

    Args:
        source_dir: Directory containing source ``.md`` agent files.
        target_dir: Destination directory.
        symlink: Create a symlink instead of copying.
        force: Overwrite existing files.
        target_suffix: Extension suffix for the installed file.  Defaults to
            ``".md"`` (e.g. ``paper-analyzer.md``).  Pass ``".agent.md"`` for
            GitHub Copilot CLI targets (e.g. ``paper-analyzer.agent.md``).

    Returns:
        Number of agent files installed.
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for agent_file in sorted(source_dir.glob("*.md")):
        dest = target_dir / (agent_file.stem + target_suffix)
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


def _install_claude_mcp(force: bool) -> bool:
    """Register the research-pipeline MCP server with Claude Code via CLI.

    Runs ``claude mcp add --transport stdio --scope user research-pipeline --
    research-pipeline mcp serve``.  When *force* is True, any existing
    registration is removed first.

    Returns:
        True on success, False when skipped or the command fails.
    """
    try:
        if force:
            subprocess.run(  # nosec B603 B607
                ["claude", "mcp", "remove", "research-pipeline"],
                capture_output=True,
                timeout=30,
            )
        else:
            check = subprocess.run(  # nosec B603 B607
                ["claude", "mcp", "get", "research-pipeline"],
                capture_output=True,
                timeout=30,
            )
            if check.returncode == 0:
                logger.info(
                    "Claude Code MCP server already registered. "
                    "Use --force to overwrite."
                )
                return False
        result = subprocess.run(  # nosec B603 B607
            [
                "claude",
                "mcp",
                "add",
                "--transport",
                "stdio",
                "--scope",
                "user",
                "research-pipeline",
                "--",
                "research-pipeline",
                "mcp",
                "serve",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            logger.info("Claude Code MCP server registered via 'claude mcp add'.")
            return True
        logger.warning(
            "'claude mcp add' failed (rc=%d): %s",
            result.returncode,
            result.stderr.strip(),
        )
        return False
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as exc:
        logger.warning("Could not register Claude Code MCP server: %s", exc)
        return False


def _install_codex_mcp(force: bool) -> bool:
    """Register the research-pipeline MCP server with Codex CLI.

    Runs ``codex mcp add research-pipeline -- research-pipeline mcp serve``.
    When *force* is True, any existing registration is removed first.

    Returns:
        True on success, False when skipped or the command fails.
    """
    try:
        if force:
            subprocess.run(  # nosec B603 B607
                ["codex", "mcp", "remove", "research-pipeline"],
                capture_output=True,
                timeout=30,
            )
        else:
            check = subprocess.run(  # nosec B603 B607
                ["codex", "mcp", "get", "research-pipeline"],
                capture_output=True,
                timeout=30,
            )
            if check.returncode == 0:
                logger.info(
                    "Codex CLI MCP server already registered. Use --force to overwrite."
                )
                return False
        result = subprocess.run(  # nosec B603 B607
            [
                "codex",
                "mcp",
                "add",
                "research-pipeline",
                "--",
                "research-pipeline",
                "mcp",
                "serve",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            logger.info("Codex CLI MCP server registered via 'codex mcp add'.")
            return True
        logger.warning(
            "'codex mcp add' failed (rc=%d): %s",
            result.returncode,
            result.stderr.strip(),
        )
        return False
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as exc:
        logger.warning("Could not register Codex CLI MCP server: %s", exc)
        return False


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


def _update_copilot_mcp_config(target: Path, force: bool) -> bool:
    """Merge the research-pipeline MCP server entry into Copilot CLI mcp-config.json.

    Unlike ``_install_mcp_config`` which writes a standalone snippet file, this
    function reads the existing Copilot config (preserving other server entries)
    and upserts only the ``research-pipeline`` key.
    """
    entry = _mcp_server_config()["mcpServers"]["research-pipeline"]

    existing: dict[str, object] = {}
    if target.exists():
        try:
            existing = json.loads(target.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Could not parse %s: %s — starting fresh.", target, exc)

    servers: dict[str, object] = existing.setdefault("mcpServers", {})  # type: ignore[assignment]
    if "research-pipeline" in servers and not force:
        logger.warning(
            "research-pipeline already present in %s. Use --force to overwrite.",
            target,
        )
        return False

    servers["research-pipeline"] = entry
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(existing, indent=2) + "\n")
    logger.info("Copilot MCP config updated at %s", target)
    return True


def _update_vscode_mcp_config(target: Path, force: bool) -> bool:
    """Merge the research-pipeline MCP server entry into VS Code's global mcp.json.

    VS Code uses a ``"servers"`` key (not ``"mcpServers"``). This function reads
    the existing config (preserving other server entries) and upserts only the
    ``research-pipeline`` key.
    """
    entry = _mcp_server_config()["mcpServers"]["research-pipeline"]

    existing: dict[str, object] = {}
    if target.exists():
        try:
            existing = json.loads(target.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Could not parse %s: %s — starting fresh.", target, exc)

    servers: dict[str, object] = existing.setdefault("servers", {})  # type: ignore[assignment]
    if "research-pipeline" in servers and not force:
        logger.warning(
            "research-pipeline already present in %s. Use --force to overwrite.",
            target,
        )
        return False

    servers["research-pipeline"] = entry
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(existing, indent=2) + "\n")
    logger.info("VS Code MCP config updated at %s", target)
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
    skip_hooks: bool = False,
) -> None:
    """Install skills and agents to assistant config directories.

    Args:
        skill_target: Explicit single destination for the skill directory.
            When omitted, installs to Claude Code, Codex CLI, and GitHub Copilot
            CLI paths for any agents detected on PATH.
        skill_targets: Explicit multiple destinations for the skill directory.
            Takes precedence over ``skill_target`` when set.
        agents_target: Destination directory for agent files (Claude ``.md``
            format).
        mcp_config_target: Destination for the standalone MCP config snippet.
        symlink: If True, create symlinks instead of copying.
        force: If True, overwrite existing files/directories.
        skip_skill: If True, skip skill installation.
        skip_agents: If True, skip agent installation.
        skip_mcp: If True, skip MCP config snippet installation (both the
            standalone snippet and the Copilot CLI ``mcp-config.json`` merge).
        skip_hooks: If True, skip Claude Code lifecycle hook registration.
    """
    if skip_skill and skip_agents and skip_mcp and skip_hooks:
        logger.warning("All setup components skipped; nothing to do.")
        return

    default_multi_target = skill_target is None and skill_targets is None

    # --- Skill ---
    if not skip_skill:
        targets = _resolve_skill_targets(skill_target, skill_targets)
        if default_multi_target:
            targets = _filter_targets_by_detection(targets)
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
        # In default multi-target mode, gate the Claude agents install using
        # the same path-prefix detection logic as skills.  Explicit or custom
        # paths (e.g. tests using tmp_path) do not match any known prefix and
        # are therefore always installed unconditionally.
        install_agents = True
        if default_multi_target:
            agents_str = str(agents_target)
            for prefix, detector in _AGENT_PATH_PREFIXES.items():
                if agents_str.startswith(prefix + "/") or agents_str == prefix:
                    install_agents = detector()
                    if not install_agents:
                        logger.info(
                            "Agent not detected (prefix %s); "
                            "skipping agent install to %s",
                            prefix,
                            agents_target,
                        )
                    break
        if install_agents:
            count = _install_agent_files(source, agents_target, symlink, force)
            logger.info("Installed %d agent(s) to %s", count, agents_target)

    # --- MCP server config ---
    if not skip_mcp:
        _install_mcp_config(mcp_config_target, force=force)
        # In default mode, only write Copilot / VS Code config files when
        # those tools are actually present on PATH.
        if not default_multi_target or _detect_copilot():
            _update_copilot_mcp_config(DEFAULT_COPILOT_MCP_CONFIG, force=force)
        else:
            logger.info(
                "GitHub Copilot CLI not detected; skipping %s",
                DEFAULT_COPILOT_MCP_CONFIG,
            )
        if not default_multi_target or shutil.which("code") is not None:
            _update_vscode_mcp_config(DEFAULT_VSCODE_MCP_CONFIG, force=force)
        else:
            logger.info(
                "VS Code not detected; skipping %s",
                DEFAULT_VSCODE_MCP_CONFIG,
            )

    # --- Claude Code lifecycle hooks ---
    if not skip_hooks and default_multi_target and _detect_claude():
        hooks_source = _find_claude_hooks_source()
        if hooks_source is not None:
            _install_claude_hooks(
                DEFAULT_CLAUDE_SETTINGS_FILE, hooks_source, force=force
            )
        else:
            logger.info("Bundled Claude Code hooks config not found; skipping.")

    logger.info("Done.")


# Backward compatibility
run_install_skill = run_setup
