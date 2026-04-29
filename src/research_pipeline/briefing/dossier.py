"""Hot-topic dossier rendering and validation."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator

from research_pipeline.briefing.models import BriefingCluster, TopicDossier
from research_pipeline.briefing.normalize import stable_hash


class FactualityLabel(StrEnum):
    """Phase E factuality labels for dossier claims.

    Every important dossier claim must carry one of these labels.
    `supported_fact` claims must additionally cite an evidence URL.
    """

    SUPPORTED_FACT = "supported_fact"
    INFERENCE = "inference"
    SPECULATION_OR_WATCH_ITEM = "speculation_or_watch_item"


class EvidenceTimelineEntry(BaseModel):
    """One row of a dossier evidence timeline."""

    model_config = ConfigDict(frozen=True)

    date: str
    evidence_url: str
    source_class: str
    note: str
    origin: Literal["cluster_event", "topic_memory"] = "cluster_event"

    @model_validator(mode="after")
    def _validate(self) -> EvidenceTimelineEntry:
        if not self.date.strip():
            raise ValueError("timeline entry date is required")
        if not self.evidence_url.strip():
            raise ValueError("timeline entry evidence_url is required")
        if not (
            self.evidence_url.startswith("http://")
            or self.evidence_url.startswith("https://")
            or self.evidence_url.startswith("obsidian://")
        ):
            raise ValueError(
                "timeline entry evidence_url must be http(s) or obsidian:// URL"
            )
        return self


class DossierClaim(BaseModel):
    """A single labeled claim that appears in a dossier."""

    model_config = ConfigDict(frozen=True)

    text: str
    label: FactualityLabel
    evidence_url: str | None = None

    @model_validator(mode="after")
    def _validate(self) -> DossierClaim:
        if not self.text.strip():
            raise ValueError("claim text is required")
        if (
            self.label == FactualityLabel.SUPPORTED_FACT
            and not (self.evidence_url or "").strip()
        ):
            raise ValueError("supported_fact claims require a non-empty evidence_url")
        if (
            self.evidence_url is not None
            and self.evidence_url.strip()
            and not (
                self.evidence_url.startswith("http://")
                or self.evidence_url.startswith("https://")
            )
        ):
            raise ValueError("claim evidence_url must be http(s)")
        return self


def build_dossier(cluster: BriefingCluster, *, run_date: str) -> TopicDossier:
    """Build a single-cluster dossier from primary evidence."""
    from research_pipeline.briefing.dossier_timeline import build_evidence_timeline

    if not cluster.primary_artifact_present:
        raise ValueError(
            f"dossier generation requires a primary artifact "
            f"(cluster_id={cluster.cluster_id})"
        )
    if not cluster.canonical_urls:
        raise ValueError(
            f"dossier generation requires at least one canonical URL "
            f"(cluster_id={cluster.cluster_id})"
        )
    if len(cluster.topic_ids) > 1:
        raise ValueError(
            f"dossier focuses on one topic only; got "
            f"{len(cluster.topic_ids)} (cluster_id={cluster.cluster_id})"
        )
    if not cluster.events:
        raise ValueError(
            f"dossier requires at least one cluster event "
            f"(cluster_id={cluster.cluster_id})"
        )
    topic_id = cluster.topic_ids[0] if cluster.topic_ids else "topic_general"
    entries = build_evidence_timeline(cluster)
    timeline = tuple(
        {
            "date": entry.date,
            "evidence": entry.evidence_url,
            "source_class": entry.source_class,
            "note": entry.note,
            "origin": entry.origin,
        }
        for entry in entries
    )
    return TopicDossier(
        dossier_id=stable_hash(run_date, cluster.cluster_id, prefix="dossier_"),
        topic_id=topic_id,
        cluster_ids=(cluster.cluster_id,),
        title=cluster.title,
        why_it_matters=(
            "This item ranked highly because it has primary or implementation "
            "evidence and a deterministic ranking rationale."
        ),
        what_changed=cluster.events[0].summary_hint or cluster.title,
        prior_context="No prior context was required for this manual dossier.",
        evidence_timeline=timeline,
        linked_artifacts=cluster.canonical_urls,
        open_questions=(
            "What should be tried locally?",
            "Does this change existing workflows?",
        ),
        try_next=("Open the primary artifact.", "Record feedback after review."),
    )


def select_dossier_candidates(
    clusters: list[BriefingCluster], *, max_count: int = 1
) -> list[BriefingCluster]:
    """Select top-ranked primary-artifact clusters for automatic dossiers."""
    candidates = [
        cluster
        for cluster in clusters
        if cluster.primary_artifact_present and cluster.canonical_urls
    ]
    return sorted(
        candidates,
        key=lambda cluster: (
            -cluster.rank_score,
            cluster.title.lower(),
            cluster.cluster_id,
        ),
    )[:max_count]


def render_dossier(dossier: TopicDossier, *, run_date: str) -> str:
    """Render a hot-topic dossier Markdown document."""
    lines = [
        "---",
        "type: topic-dossier",
        f"date: {run_date}",
        f"dossier_id: {dossier.dossier_id}",
        f"topic_id: {dossier.topic_id}",
        "cluster_ids:",
        *[f"  - {cluster_id}" for cluster_id in dossier.cluster_ids],
        "status: draft",
        "---",
        "",
        f"# Hot Topic Dossier - {dossier.title}",
        "",
        "## Agent Read Map",
        "",
        "| Field | Value |",
        "|---|---|",
        "| Use when | User wants depth beyond daily brief |",
        "| Core question | What changed and what should we do with it? |",
        "| Evidence standard | Primary artifact required |",
        "",
        "## One-paragraph Summary",
        "",
        dossier.what_changed,
        "",
        "## What Changed",
        "",
        dossier.what_changed,
        "",
        "## Why It Matters Technically",
        "",
        dossier.why_it_matters,
        "",
        "## Prior Context",
        "",
        dossier.prior_context,
        "",
        "## Evidence Timeline",
        "",
        "| Date | Evidence | Source class | Note |",
        "|---|---|---|---|",
    ]
    for row in dossier.evidence_timeline:
        lines.append(
            f"| {row['date']} | [{row['evidence']}]({row['evidence']}) | "
            f"{row['source_class']} | {row['note']} |"
        )
    lines.extend(
        [
            "",
            "## Artifacts To Open",
            "",
            *[f"- {url}" for url in dossier.linked_artifacts],
            "",
            "## What To Try / Watch / Ignore",
            "",
            *[f"- {item}" for item in dossier.try_next],
            "",
            "## Open Questions",
            "",
            *[f"- {question}" for question in dossier.open_questions],
            "",
            "## Agent Notes",
            "",
            f"- topic_id={dossier.topic_id}",
            f"- dossier_id={dossier.dossier_id}",
            "- factuality_label=supported_fact",
            "",
        ]
    )
    return "\n".join(lines)


def write_dossier(path: Path, markdown: str) -> None:
    """Write dossier Markdown."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(markdown, encoding="utf-8")


def write_dossier_with_archive(
    path: Path,
    markdown: str,
    *,
    archive_path: Path | None = None,
) -> tuple[Path, Path | None]:
    """Write dossier Markdown and optionally mirror to an archive path.

    Returns ``(primary_path, archive_path_written)``. The archive copy is
    only written when ``archive_path`` is supplied. Both writes are
    idempotent: re-running with identical markdown produces identical
    files.
    """
    write_dossier(path, markdown)
    archived: Path | None = None
    if archive_path is not None:
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        archive_path.write_text(markdown, encoding="utf-8")
        archived = archive_path
    return path, archived
