"""Obsidian archive exporters for briefing artifacts.

Phase C ticket C01: configuration model and vault-path allowlist for
Obsidian exports. Path safety, allowed subdir enforcement, and
generated-note ownership checks live here so later tickets (C02-C08) can
reuse them. Existing exporter helpers (``export_daily_note``,
``export_topic_notes``, ``export_source_notes``) are preserved for
backwards compatibility.
"""

from __future__ import annotations

import re
from pathlib import Path

from pydantic import BaseModel, ConfigDict, model_validator

from research_pipeline.briefing.models import BriefingCluster, BriefingSourceConfig
from research_pipeline.briefing.validate import validate_obsidian_path

GENERATED_ID_KEY = "generated_id"

DEFAULT_VAULT_SUBDIR = "AI-Intelligence"
DEFAULT_ALLOWED_SUBDIRS: tuple[str, ...] = (
    "Daily",
    "Topics",
    "Sources",
    "Weekly",
    "Monthly",
)

_FRONTMATTER_RE = re.compile(r"\A---\n(.*?)\n---\n", re.DOTALL)


class ObsidianConfig(BaseModel):
    """Configuration for Obsidian exports.

    Phase C does not introduce a new on-disk config surface; this model is
    constructed from the briefing config or CLI flags. It captures the vault
    location, the namespace subdir owned by the generator, the per-note-type
    allowlist, the dry-run flag, and the frontmatter key used to claim
    ownership of generated notes.
    """

    model_config = ConfigDict(frozen=True)

    vault_root: Path
    subdir: str = DEFAULT_VAULT_SUBDIR
    allowed_subdirs: tuple[str, ...] = DEFAULT_ALLOWED_SUBDIRS
    dry_run: bool = False
    generated_id_key: str = GENERATED_ID_KEY

    @model_validator(mode="after")
    def _check_vault_root(self) -> ObsidianConfig:
        if not self.vault_root.exists() or not self.vault_root.is_dir():
            raise ValueError(
                f"vault_root must be an existing directory: {self.vault_root}"
            )
        if not self.subdir or "/" in self.subdir or "\\" in self.subdir:
            raise ValueError(f"invalid vault subdir: {self.subdir!r}")
        if not self.allowed_subdirs:
            raise ValueError("allowed_subdirs must be non-empty")
        for name in self.allowed_subdirs:
            if not name or "/" in name or "\\" in name or ".." in name:
                raise ValueError(f"invalid allowed subdir entry: {name!r}")
        return self

    @property
    def namespace_root(self) -> Path:
        """Resolved root of the generator-owned namespace inside the vault."""
        return (self.vault_root / self.subdir).resolve()


def validate_vault_path(target: Path, config: ObsidianConfig) -> Path:
    """Resolve ``target`` and verify it is safe to write under ``config``.

    Enforces:
      * resolved path is under ``vault_root``
      * resolved path is under the configured generator namespace subdir
      * the immediate parent under the namespace is in ``allowed_subdirs``
      * the path has a ``.md`` extension
      * symlink escapes are rejected (because we resolve before checking)

    Returns the resolved path on success and raises ``ValueError`` otherwise.
    """
    if target.suffix != ".md":
        raise ValueError(f"obsidian path must be a .md markdown file: {target}")

    vault_root = config.vault_root.resolve()
    namespace_root = config.namespace_root

    parent_resolved = target.parent.resolve()
    if vault_root != parent_resolved and vault_root not in parent_resolved.parents:
        raise ValueError(f"obsidian path escapes vault root: {target}")

    resolved = parent_resolved / target.name

    if namespace_root != parent_resolved and namespace_root not in (
        parent_resolved.parents
    ):
        raise ValueError(
            "obsidian path is outside the configured namespace/subdir "
            f"{config.subdir!r}: {target}"
        )

    try:
        rel = resolved.relative_to(namespace_root)
    except ValueError as exc:  # pragma: no cover - guarded above
        raise ValueError(f"obsidian path escapes namespace root: {target}") from exc

    if len(rel.parts) < 2:
        raise ValueError(
            f"obsidian path must live in an allowed subdir of {config.subdir!r}: "
            f"{target}"
        )
    top = rel.parts[0]
    if top not in config.allowed_subdirs:
        raise ValueError(
            f"obsidian path subdir {top!r} not in allowed list "
            f"{config.allowed_subdirs}: {target}"
        )

    validate_obsidian_path(resolved, config.vault_root)
    return resolved


def is_owned_generated_note(path: Path, generated_id: str) -> bool:
    """Return True if ``path`` is safe for the generator to overwrite.

    Ownership rule:
      * Missing file → owned (the generator may create it).
      * Existing file with frontmatter containing matching
        ``generated_id`` → owned.
      * Anything else (human note, mismatched id, no frontmatter) →
        NOT owned. Callers must refuse to overwrite.
    """
    if not path.exists():
        return True
    try:
        content = path.read_text(encoding="utf-8")
    except OSError:
        return False
    match = _FRONTMATTER_RE.match(content)
    if not match:
        return False
    block = match.group(1)
    needle = f"{GENERATED_ID_KEY}: {generated_id}"
    return any(line.strip() == needle for line in block.splitlines())


def export_daily_note(markdown: str, *, vault_root: Path, run_date: str) -> Path:
    """Export a daily briefing note into an Obsidian vault."""
    if "/" in run_date or "\\" in run_date or ".." in run_date:
        raise ValueError(f"invalid briefing date for Obsidian export: {run_date}")
    path = vault_root / "AI-Intelligence" / "Daily" / f"{run_date}.md"
    validate_obsidian_path(path, vault_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(markdown, encoding="utf-8")
    return path


def export_topic_notes(
    clusters: list[BriefingCluster], *, vault_root: Path
) -> list[Path]:
    """Export/update topic notes for reported clusters."""
    paths: list[Path] = []
    for cluster in clusters:
        for topic_id in cluster.topic_ids:
            slug = topic_id.removeprefix("topic_")
            path = vault_root / "AI-Intelligence" / "Topics" / f"{slug}.md"
            validate_obsidian_path(path, vault_root)
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists() and f"topic_id: {topic_id}" not in path.read_text(
                encoding="utf-8"
            ):
                raise ValueError(f"refusing to overwrite unrelated topic note: {path}")
            path.write_text(_topic_note(topic_id, cluster), encoding="utf-8")
            paths.append(path)
    return paths


def export_source_notes(
    sources: list[BriefingSourceConfig], *, vault_root: Path
) -> list[Path]:
    """Export source reliability notes."""
    paths: list[Path] = []
    for source in sources:
        path = vault_root / "AI-Intelligence" / "Sources" / f"{source.source_id}.md"
        validate_obsidian_path(path, vault_root)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and f"source_id: {source.source_id}" not in path.read_text(
            encoding="utf-8"
        ):
            raise ValueError(f"refusing to overwrite unrelated source note: {path}")
        path.write_text(_source_note(source), encoding="utf-8")
        paths.append(path)
    return paths


def _topic_note(topic_id: str, cluster: BriefingCluster) -> str:
    slug = topic_id.removeprefix("topic_")
    return "\n".join(
        [
            "---",
            "type: briefing-topic",
            f"topic_id: {topic_id}",
            f"clusters: [{cluster.cluster_id}]",
            "---",
            "",
            f"# {slug}",
            "",
            "## Agent Read Map",
            "",
            "| Field | Value |",
            "|---|---|",
            f"| Topic ID | `{topic_id}` |",
            f"| Latest cluster | `{cluster.cluster_id}` |",
            "",
            "## Latest Evidence",
            "",
            f"- [{cluster.title}]({cluster.canonical_urls[0]})",
            "",
        ]
    )


def _source_note(source: BriefingSourceConfig) -> str:
    return "\n".join(
        [
            "---",
            "type: briefing-source",
            f"source_id: {source.source_id}",
            f"source_class: {source.source_class.value}",
            f"access_method: {source.access_method.value}",
            "---",
            "",
            f"# {source.source_name}",
            "",
            "## Agent Read Map",
            "",
            "| Field | Value |",
            "|---|---|",
            f"| Source ID | `{source.source_id}` |",
            f"| Cadence | {source.cadence} |",
            "| Current weight | "
            f"trust={source.trust_weight}; noise={source.noise_weight} |",
            "",
            "## Policy Notes",
            "",
            f"- Retention: {source.retention_policy}",
            f"- Raw storage allowed: {source.allowed_raw_storage}",
            "",
        ]
    )
