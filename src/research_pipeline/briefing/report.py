"""Markdown renderers for daily and weekly briefing reports."""

from __future__ import annotations

from collections import Counter

from research_pipeline.briefing.memory_lookup import lookup_recent_topic_context
from research_pipeline.briefing.models import (
    BriefingCluster,
    IntelligenceEvent,
    SourceClass,
)
from research_pipeline.briefing.topic_memory_store import TopicMemoryStore


def render_daily_brief(
    clusters: list[BriefingCluster],
    *,
    run_date: str,
    status: str = "draft",
    quiet_sources: list[str] | None = None,
    topic_memory: TopicMemoryStore | None = None,
    dossier_links: list[tuple[str, str]] | None = None,
    full_detail_limit: int = 5,
) -> str:
    """Render an extractive daily Markdown brief.

    ``dossier_links`` is an optional ordered list of ``(title, path_or_url)``
    tuples for any manually generated Phase E dossiers. When supplied a
    ``## Linked Dossiers`` section is rendered just below ``## Top Items``.

    ``full_detail_limit`` caps how many top-ranked clusters get verbose
    treatment in ``## Top Items``. Remaining ranked clusters are listed as
    compact bullets under ``### Also tracked`` (no extra evidence link beyond
    those already captured in ``ranked_clusters.jsonl``). This keeps the daily
    brief within the governance link/word budget while still surfacing every
    ranked item to human readers.
    """
    quiet_sources = quiet_sources or []
    item_count = len(clusters)
    link_count = sum(len(cluster.canonical_urls[:1]) for cluster in clusters)
    source_mix = _source_mix(clusters)
    material_updates = [
        cluster for cluster in clusters if cluster.primary_artifact_present
    ]
    lines: list[str] = [
        "---",
        "type: daily-brief",
        f"date: {run_date}",
        f"brief_id: brief_{run_date.replace('-', '_')}",
        f"status: {status}",
        f"item_count: {item_count}",
        f"link_count: {link_count}",
        "source_mix:",
        *[f"  {key}: {source_mix[key]}" for key in sorted(source_mix)],
        "---",
        "",
        f"# Daily AI Intelligence Brief - {run_date}",
        "",
        "## Agent Read Map",
        "",
        "| Field | Value |",
        "|---|---|",
        "| Primary purpose | 20-minute daily technical brief |",
        "| Best sections for action | Executive Signal, Top Items, Follow-up Queue |",
        "| Machine targets | cluster IDs, topic IDs, evidence URLs, feedback IDs |",
        "",
        "## Executive Signal",
        "",
    ]
    if clusters:
        for cluster in clusters[:3]:
            primary = _primary_event(cluster)
            label = _evidence_label(cluster.evidence_type)
            action = cluster.suggested_action
            lines.append(
                f"- {label} {cluster.title} (`{cluster.cluster_id}`): "
                f"{_summary_text(primary, cluster)[:180]} "
                f"Action: **{action}**."
            )
    else:
        lines.append("- No primary artifact passed the daily inclusion threshold.")
    lines.extend(["", "## Top Items", ""])
    if clusters:
        full_detail = clusters[:full_detail_limit]
        compact = clusters[full_detail_limit:]
        for index, cluster in enumerate(full_detail, start=1):
            lines.extend(
                _render_cluster_item(
                    index,
                    cluster,
                    topic_memory=topic_memory,
                )
            )
        if compact:
            lines.extend(
                [
                    "### Also tracked",
                    "",
                    (
                        f"{len(compact)} additional ranked clusters are summarised "
                        "below; full evidence URLs and metadata are in "
                        "`ranked_clusters.jsonl`."
                    ),
                    "",
                ]
            )
            for offset, cluster in enumerate(compact, start=full_detail_limit + 1):
                action = cluster.suggested_action
                url = cluster.canonical_urls[0]
                lines.append(
                    f"- {offset}. [{cluster.title}]({url}) "
                    f"(`{cluster.cluster_id}`) — action: {action}"
                )
            lines.append("")
    else:
        lines.append("No ranked items passed the inclusion threshold today.")
        lines.append("")
    lines.extend(
        _section_for_class(
            "Papers Worth Scanning", clusters, SourceClass.ACADEMIC_SOURCE, 3
        )
    )
    lines.extend(
        _section_for_class(
            "Repos / Releases Worth Opening",
            clusters,
            SourceClass.IMPLEMENTATION_SOURCE,
            3,
        )
    )
    lines.extend(
        _section_for_class(
            "Discussions Worth Watching",
            clusters,
            SourceClass.TECHNICAL_DISCUSSION,
            2,
        )
    )
    lines.extend(["## Follow-up Queue", ""])
    if clusters:
        for cluster in clusters[:3]:
            lines.append(
                f"- Review `{cluster.cluster_id}` ({cluster.title}) "
                "and decide feedback."
            )
    else:
        lines.append("- Re-run after the next scheduled source cadence.")
    if dossier_links:
        lines.extend(["", "## Linked Dossiers", ""])
        for title, link in dossier_links:
            lines.append(f"- [{title}]({link})")
    lines.extend(["", "## Suppressed / Not Reported", ""])
    lines.append("Repeated, low-novelty, or hype-heavy topics are suppressed.")
    lines.extend(["", "## No Material Updates", ""])
    if material_updates:
        if len(material_updates) < 6:
            lines.append(
                "Low-signal day: fewer than six high-quality primary items passed "
                "the inclusion threshold, so the brief is intentionally short."
            )
        else:
            lines.append(
                "Not applicable; primary artifacts passed the inclusion threshold."
            )
    else:
        lines.append("No primary artifact passed the daily inclusion threshold.")
    lines.extend(["", "## Watchlist Quiet", ""])
    if quiet_sources:
        for source in quiet_sources:
            lines.append(f"- {source}")
    else:
        lines.append("No quiet-source summary was provided for this run.")
    lines.extend(["", "## Feedback Targets", ""])
    lines.extend(["| Target | ID | Suggested feedback command |", "|---|---|---|"])
    for cluster in clusters[:5]:
        lines.append(
            "| "
            f"{cluster.title} | `{cluster.cluster_id}` | "
            "`research-pipeline brief feedback "
            f"--cluster {cluster.cluster_id} --signal keep` |"
        )
    if not clusters:
        lines.append("| No material updates | `none` | Re-run later |")
    lines.append("")
    return "\n".join(lines)


def render_weekly_synthesis(
    daily_reports: list[str], *, week_id: str, max_links: int = 25
) -> str:
    """Render a lightweight weekly trend memo from daily reports."""
    links: list[str] = []
    for report in daily_reports:
        links.extend(_markdown_links(report))
    links = links[:max_links]
    lines = [
        "---",
        "type: weekly-briefing-synthesis",
        f"week: {week_id}",
        "---",
        "",
        f"# Weekly AI Intelligence Synthesis - {week_id}",
        "",
        "## Agent Read Map",
        "",
        "| Field | Value |",
        "|---|---|",
        "| Use | Trend memo from validated daily briefs |",
        "| Evidence | Links copied from daily briefing artifacts |",
        "",
        "## What changed this week",
        "",
        f"- Reviewed {len(daily_reports)} daily briefing artifacts.",
        "",
        "## Themes that strengthened",
        "",
        "- See linked daily brief clusters for repeated high-rank topics.",
        "",
        "## What was noise",
        "",
        "- Items suppressed by daily validators or explicit feedback remain excluded.",
        "",
        "## Watchlist updates",
        "",
        "- Review source registry weights before adding new sources.",
        "",
        "## Feedback and source-quality notes",
        "",
    ]
    if links:
        lines.extend(f"- {link}" for link in links)
    else:
        lines.append("- No evidence links were present in the supplied daily reports.")
    lines.append("")
    return "\n".join(lines)


def _render_cluster_item(
    index: int,
    cluster: BriefingCluster,
    *,
    topic_memory: TopicMemoryStore | None = None,
) -> list[str]:
    topic = cluster.topic_ids[0] if cluster.topic_ids else "topic_general"
    source_class = (
        cluster.source_classes[0].value if cluster.source_classes else "unknown"
    )
    evidence_url = cluster.canonical_urls[0]
    primary = _primary_event(cluster)
    summary = _summary_text(primary, cluster)
    label = _evidence_label(cluster.evidence_type)
    prior_context = _prior_context_text(cluster, topic_memory)
    lines = [
        f"### {index}. {cluster.title} `{cluster.cluster_id}`",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| Topic | `{topic}` / [[{topic.removeprefix('topic_')}]] |",
        f"| Source class | {source_class} |",
        f"| Novelty | {cluster.novelty_type} |",
        f"| Confidence | {cluster.confidence} |",
        f"| Suggested action | {cluster.suggested_action} |",
        "",
        f"**What changed:** {label} {summary[:300] or cluster.title}",
        "",
        f"**Why it matters:** {_why_it_matters(cluster)}",
        "",
        f"**Evidence:** [{cluster.title}]({evidence_url})",
        "",
        f"**Source trace:** {_source_trace(cluster)}",
        "",
        "**Agent note:** "
        f"cluster_id={cluster.cluster_id}; topic_id={topic}; "
        f"evidence_type={cluster.evidence_type}.",
        "",
    ]
    if prior_context:
        lines.insert(-2, f"**Prior context:** {prior_context}")
        lines.insert(-2, "")
    return lines


def _prior_context_text(
    cluster: BriefingCluster,
    topic_memory: TopicMemoryStore | None,
) -> str | None:
    if topic_memory is None:
        return None
    memories = lookup_recent_topic_context(topic_memory, cluster)
    if not memories:
        return None
    memory = memories[0]
    if cluster.novelty_type == "resurfaced" or memory.status == "resurfaced":
        return (
            f"resurfaced topic; last reported {memory.last_reported_at}; "
            f"seen {memory.report_count_30d} times in 30d."
        )
    if cluster.fatigue_penalty > 0.0 or cluster.novelty_type == "cooling":
        return (
            f"repeated topic; last reported {memory.last_reported_at}; "
            f"fatigue={memory.fatigue_score:.2f}."
        )
    return None


def _section_for_class(
    heading: str,
    clusters: list[BriefingCluster],
    source_class: SourceClass,
    limit: int,
) -> list[str]:
    lines = [f"## {heading}", ""]
    selected = [
        cluster for cluster in clusters if source_class in cluster.source_classes
    ][:limit]
    if not selected:
        lines.append("No items in this category passed the daily budget.")
    for cluster in selected:
        if cluster.canonical_urls:
            url = cluster.canonical_urls[0]
            lines.append(f"- `{cluster.cluster_id}` — [{cluster.title}]({url})")
        else:
            lines.append(f"- `{cluster.cluster_id}` — {cluster.title}")
    lines.append("")
    return lines


def _source_mix(clusters: list[BriefingCluster]) -> Counter[str]:
    mix: Counter[str] = Counter(
        {
            "academic_source": 0,
            "implementation_source": 0,
            "media_news": 0,
            "newsletter": 0,
            "primary_artifact": 0,
            "social_signal": 0,
            "technical_discussion": 0,
            "video_audio": 0,
        }
    )
    for cluster in clusters:
        for source_class in cluster.source_classes:
            mix[source_class.value] += 1
    return mix


def _primary_event(cluster: BriefingCluster) -> IntelligenceEvent:
    return next(
        (
            event
            for event in cluster.events
            if event.event_id == cluster.primary_event_id
        ),
        cluster.events[0],
    )


def _summary_text(primary: IntelligenceEvent, cluster: BriefingCluster) -> str:
    for event in (primary, *cluster.events):
        summary = (event.summary_hint or event.excerpt).strip()
        if summary and "duplicate" not in summary.lower():
            return summary
    return cluster.title


def _evidence_label(evidence_type: str) -> str:
    labels = {
        "supported_fact": "[FACT]",
        "inference": "[INFERENCE]",
        "speculation_or_watch_item": "[WATCH]",
    }
    return labels.get(evidence_type, "[EVIDENCE]")


def _why_it_matters(cluster: BriefingCluster) -> str:
    if cluster.suggested_action == "try":
        action = "It has implementation evidence worth testing locally."
    elif cluster.suggested_action == "watch":
        action = "It is discussion-level signal; watch for primary corroboration."
    elif cluster.suggested_action == "ignore":
        action = "It is likely fatigue-heavy or hype-heavy unless new evidence appears."
    else:
        action = "It has primary evidence worth reading."
    return action


def _source_trace(cluster: BriefingCluster) -> str:
    parts = [
        f"{event.source_name}/{event.collection_method.value}/{event.confidence}"
        for event in cluster.events
    ]
    return "; ".join(parts)


def _markdown_links(markdown: str) -> list[str]:
    import re

    return re.findall(r"\[[^\]]+\]\([^)]+\)", markdown)
