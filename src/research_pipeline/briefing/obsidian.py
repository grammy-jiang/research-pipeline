"""Obsidian archive exporters for briefing artifacts."""

from __future__ import annotations

from pathlib import Path

from research_pipeline.briefing.models import BriefingCluster, BriefingSourceConfig
from research_pipeline.briefing.validate import validate_obsidian_path


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
