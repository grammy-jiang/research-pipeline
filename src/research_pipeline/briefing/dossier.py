"""Hot-topic dossier rendering and validation."""

from __future__ import annotations

from pathlib import Path

from research_pipeline.briefing.models import BriefingCluster, TopicDossier
from research_pipeline.briefing.normalize import stable_hash


def build_dossier(cluster: BriefingCluster, *, run_date: str) -> TopicDossier:
    """Build a single-cluster dossier from primary evidence."""
    if not cluster.primary_artifact_present:
        raise ValueError("dossier generation requires at least one primary artifact")
    topic_id = cluster.topic_ids[0] if cluster.topic_ids else "topic_general"
    timeline = tuple(
        {
            "date": event.published_at or event.retrieved_at[:10],
            "evidence": str(event.canonical_url),
            "source_class": event.source_type.value,
            "note": event.title,
        }
        for event in cluster.events
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
