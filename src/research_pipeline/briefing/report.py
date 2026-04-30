"""Markdown renderers for daily and weekly briefing reports."""

from __future__ import annotations

import re
from collections import Counter

from research_pipeline.briefing.memory_lookup import lookup_recent_topic_context
from research_pipeline.briefing.models import (
    BriefingCluster,
    IntelligenceEvent,
    SourceClass,
)
from research_pipeline.briefing.topic_memory_store import TopicMemoryStore

_ACTION_ICONS = {
    "read": "📥 read",
    "try": "🛠️ try",
    "watch": "👁️ watch",
    "ignore": "🚫 ignore",
}
_CONFIDENCE_ICONS = {
    "high": "🟢 high",
    "medium": "🟡 medium",
    "low": "🔴 low",
}
_EVIDENCE_ICONS = {
    "supported_fact": "✨",
    "inference": "🧠",
    "speculation_or_watch_item": "👀",
}
_CLASS_HEADINGS = (
    ("📦 Repos & Releases", SourceClass.IMPLEMENTATION_SOURCE, 5),
    ("📄 Papers", SourceClass.ACADEMIC_SOURCE, 5),
    ("💬 Discussions", SourceClass.TECHNICAL_DISCUSSION, 5),
)


def render_daily_brief(
    clusters: list[BriefingCluster],
    *,
    run_date: str,
    status: str = "draft",
    quiet_sources: list[str] | None = None,
    topic_memory: TopicMemoryStore | None = None,
    dossier_links: list[tuple[str, str]] | None = None,
    full_detail_limit: int = 5,
    previous_brief_link: str | None = None,
) -> str:
    """Render a focused daily Markdown brief with icons and internal anchors.

    Compared to earlier versions this layout removes static boilerplate
    (Agent Read Map, Follow-up Queue, Suppressed/Not Reported, No Material
    Updates, Watchlist Quiet) and replaces verbose per-item field tables with
    a single badge line. A ``📑 Contents`` section provides intra-document
    navigation, Executive Signal items deep-link to their detailed Top Items
    entry, and ``previous_brief_link`` (when supplied) renders a 🔗 cross-link
    to the previous day's brief.
    """
    quiet_sources = quiet_sources or []
    item_count = len(clusters)
    link_count = sum(len(cluster.canonical_urls[:1]) for cluster in clusters)
    source_mix = _source_mix(clusters)
    rich_clusters = [c for c in clusters if _has_meaningful_summary(c)]
    thin_clusters = [c for c in clusters if not _has_meaningful_summary(c)]
    full_detail = rich_clusters[:full_detail_limit]
    compact = [*rich_clusters[full_detail_limit:], *thin_clusters]
    full_detail_anchors = {
        cluster.cluster_id: _heading_anchor(_top_item_heading(index, cluster))
        for index, cluster in enumerate(full_detail, start=1)
    }

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
        f"# 🧠 Daily AI Intelligence Brief — {run_date}",
        "",
    ]
    header_facts = _header_summary(item_count, source_mix)
    if previous_brief_link:
        lines.append(f"🔗 [← Previous brief]({previous_brief_link})")
        lines.append("")
    if header_facts:
        lines.append(header_facts)
        lines.append("")

    extra_sections = _present_class_sections(clusters)
    lines.extend(_render_contents(extra_sections, has_also_tracked=bool(compact)))
    lines.append("")

    lines.extend(["## 🔥 Executive Signal", ""])
    if full_detail:
        for cluster in full_detail[:3]:
            lines.append(_render_executive_bullet(cluster, full_detail_anchors))
    else:
        lines.append("- No primary artifact passed the daily inclusion threshold.")
    lines.append("")

    lines.extend(["## ⭐ Top Items", ""])
    if full_detail:
        for index, cluster in enumerate(full_detail, start=1):
            lines.extend(
                _render_cluster_item(index, cluster, topic_memory=topic_memory)
            )
        if compact:
            lines.extend(["### Also tracked", ""])
            for offset, cluster in enumerate(compact, start=len(full_detail) + 1):
                lines.append(_render_compact_bullet(offset, cluster))
            lines.append("")
    elif compact:
        lines.extend(["### Also tracked", ""])
        for offset, cluster in enumerate(compact, start=1):
            lines.append(_render_compact_bullet(offset, cluster))
        lines.append("")
    else:
        lines.append("No ranked items passed the inclusion threshold today.")
        lines.append("")

    for heading, source_class, limit in _CLASS_HEADINGS:
        section_lines = _render_class_section(heading, clusters, source_class, limit)
        if section_lines:
            lines.extend(section_lines)

    if dossier_links:
        lines.extend(["## 📚 Linked Dossiers", ""])
        for title, link in dossier_links:
            lines.append(f"- [{title}]({link})")
        lines.append("")

    if item_count and item_count < 6:
        lines.extend(
            [
                "> ℹ️ Low-signal day: fewer than six high-quality primary items "
                "passed the inclusion threshold, so the brief is intentionally short.",
                "",
            ]
        )

    if quiet_sources:
        lines.extend(["## 🔇 Quiet sources", ""])
        for source in quiet_sources:
            lines.append(f"- {source}")
        lines.append("")

    lines.extend(["## 🗒️ Feedback Targets", ""])
    if clusters:
        lines.extend(["| Cluster | Quick command |", "|---|---|"])
        for cluster in clusters[:5]:
            lines.append(
                f"| {cluster.title} | "
                f"`research-pipeline brief feedback "
                f"--cluster {cluster.cluster_id} --signal keep` |"
            )
    else:
        lines.append("Re-run after the next scheduled source cadence.")
    lines.append("")
    return "\n".join(lines)


def _render_contents(
    extra_section_titles: list[str],
    *,
    has_also_tracked: bool,
) -> list[str]:
    """Build the ``## 📑 Contents`` block.

    ``extra_section_titles`` lists the optional class-scoped sections that
    will actually be rendered (Repos & Releases, Papers, Discussions). The
    Also-tracked sub-section is exposed as a sub-bullet under Top Items when
    present.
    """
    lines = ["## 📑 Contents", ""]
    lines.append("- [🔥 Executive Signal](#executive-signal)")
    lines.append("- [⭐ Top Items](#top-items)")
    if has_also_tracked:
        lines.append("  - [Also tracked](#also-tracked)")
    for heading in extra_section_titles:
        lines.append(f"- [{heading}](#{_heading_anchor(heading)})")
    lines.append("- [🗒️ Feedback Targets](#feedback-targets)")
    return lines


def _render_executive_bullet(
    cluster: BriefingCluster,
    full_detail_anchors: dict[str, str],
) -> str:
    icon = _EVIDENCE_ICONS.get(cluster.evidence_type, "•")
    action = _ACTION_ICONS.get(cluster.suggested_action, cluster.suggested_action)
    summary = _summary_text(_primary_event(cluster), cluster)
    snippet = _shorten(_flatten_summary(summary), 200, fallback=cluster.title)
    anchor = full_detail_anchors.get(cluster.cluster_id)
    title_link = f"[{cluster.title}](#{anchor})" if anchor else cluster.title
    return f"- {icon} **{title_link}** — {action} · {snippet}"


def _render_compact_bullet(offset: int, cluster: BriefingCluster) -> str:
    action = _ACTION_ICONS.get(cluster.suggested_action, cluster.suggested_action)
    url = cluster.canonical_urls[0] if cluster.canonical_urls else ""
    title_md = f"[{cluster.title}]({url})" if url else cluster.title
    snippet = ""
    summary = _summary_text(_primary_event(cluster), cluster)
    if summary and summary != cluster.title:
        snippet = f" — {_shorten(_flatten_summary(summary), 140, fallback='')}"
    return f"{offset}. {action} · {title_md}{snippet} (`{cluster.cluster_id}`)"


def _render_class_section(
    heading: str,
    clusters: list[BriefingCluster],
    source_class: SourceClass,
    limit: int,
) -> list[str]:
    selected = [
        cluster for cluster in clusters if source_class in cluster.source_classes
    ][:limit]
    if not selected:
        return []
    lines = [f"## {heading}", ""]
    for cluster in selected:
        action = _ACTION_ICONS.get(cluster.suggested_action, cluster.suggested_action)
        url = cluster.canonical_urls[0] if cluster.canonical_urls else ""
        title_md = f"[{cluster.title}]({url})" if url else cluster.title
        summary = _summary_text(_primary_event(cluster), cluster)
        if summary and summary != cluster.title:
            snippet = _shorten(_flatten_summary(summary), 220, fallback="")
            lines.append(
                f"- {action} · **{title_md}** — {snippet} (`{cluster.cluster_id}`)"
            )
        else:
            lines.append(f"- {action} · **{title_md}** (`{cluster.cluster_id}`)")
    lines.append("")
    return lines


def _present_class_sections(clusters: list[BriefingCluster]) -> list[str]:
    return [
        heading
        for heading, source_class, _ in _CLASS_HEADINGS
        if any(source_class in cluster.source_classes for cluster in clusters)
    ]


def _header_summary(item_count: int, source_mix: Counter[str]) -> str:
    if item_count == 0:
        return ""
    parts: list[str] = [f"📊 **{item_count} items**"]
    pretty = {
        "primary_artifact": "primary",
        "implementation_source": "impl",
        "academic_source": "papers",
        "technical_discussion": "discussion",
        "media_news": "news",
        "newsletter": "newsletter",
        "social_signal": "social",
        "video_audio": "media",
    }
    mix_parts = [
        f"{count} {pretty.get(key, key)}" for key, count in source_mix.items() if count
    ]
    if mix_parts:
        parts.append(" · ".join(mix_parts))
    return " · ".join(parts)


def _shorten(text: str, limit: int, *, fallback: str = "") -> str:
    text = (text or "").strip()
    if not text:
        return fallback
    if len(text) <= limit:
        return text
    truncated = text[: limit - 1].rstrip()
    return f"{truncated}…"


def _flatten_summary(text: str) -> str:
    """Collapse multi-line markdown / HTML release notes into a single
    readable line suitable for inline use in a list item.

    Strips HTML tags, drops markdown heading hashes, removes link/image
    syntax (keeping the visible text), and squashes whitespace so the
    output never breaks list rendering.
    """
    if not text:
        return ""
    flat = re.sub(r"<[^>]+>", " ", text)
    flat = re.sub(r"!\[([^\]]*)\]\([^)]*\)", r"\1", flat)
    flat = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", flat)
    flat = re.sub(r"^\s*#+\s*", "", flat, flags=re.MULTILINE)
    flat = re.sub(r"^\s*[-*]\s+", "", flat, flags=re.MULTILINE)
    flat = flat.replace("`", "")
    flat = re.sub(r"\s+", " ", flat).strip()
    return flat


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
    """Render a focused Top Items entry as a heading + badge line + summary."""
    topic = cluster.topic_ids[0] if cluster.topic_ids else "topic_general"
    source_class = (
        cluster.source_classes[0].value if cluster.source_classes else "unknown"
    )
    evidence_url = cluster.canonical_urls[0]
    primary = _primary_event(cluster)
    summary = _summary_text(primary, cluster) or cluster.title
    label = _evidence_label(cluster.evidence_type)
    icon = _EVIDENCE_ICONS.get(cluster.evidence_type, "")
    action = _ACTION_ICONS.get(cluster.suggested_action, cluster.suggested_action)
    confidence = _CONFIDENCE_ICONS.get(cluster.confidence, cluster.confidence)
    novelty = f"🆕 {cluster.novelty_type}" if cluster.novelty_type else ""
    badges = " · ".join(
        part
        for part in (
            f"`{action}`",
            f"`{confidence}`",
            f"`📍 {source_class}`",
            f"`{novelty}`" if novelty else "",
            f"`🏷️ {topic}`",
        )
        if part
    )
    heading = _top_item_heading(index, cluster)
    lines = [
        f"### {heading}",
        "",
        badges,
        "",
        f"{icon} {label} {_shorten(summary, 400, fallback=cluster.title)}".strip(),
        "",
        f"🔗 [{_primary_event_label(primary, cluster)}]({evidence_url})",
        "",
        f"<sub>`{cluster.cluster_id}`</sub>",
        "",
    ]
    prior_context = _prior_context_text(cluster, topic_memory)
    if prior_context:
        lines.insert(4, f"_Prior context: {prior_context}_")
        lines.insert(5, "")
    return lines


def _top_item_heading(index: int, cluster: BriefingCluster) -> str:
    return f"{index}. {cluster.title}"


def _heading_anchor(heading: str) -> str:
    """Slugify a Markdown heading the way GitHub Flavored Markdown does.

    GFM lowercases, replaces spaces with hyphens, drops punctuation, and keeps
    unicode letters and emojis intact. Browsers URL-encode the resulting
    fragment automatically.
    """
    slug = heading.strip().lower()
    slug = re.sub(r"[^\w\- ]+", "", slug, flags=re.UNICODE)
    slug = slug.replace(" ", "-")
    slug = re.sub(r"-+", "-", slug).strip("-")
    return slug


def _primary_event_label(primary: IntelligenceEvent, cluster: BriefingCluster) -> str:
    name = (primary.source_name or "").strip()
    return name or cluster.title


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


def _has_meaningful_summary(cluster: BriefingCluster) -> bool:
    """A cluster has a meaningful summary when its excerpt actually
    describes the item rather than just echoing a category label or the title.

    A category label like ``"Data Mining & Modeling"`` is title-cased and
    word-only, while real prose contains at least one lowercase token after
    the first word (articles, verbs, prepositions).
    """
    summary = _summary_text(_primary_event(cluster), cluster).strip()
    if not summary or summary == cluster.title:
        return False
    if any(c in summary for c in ".!?"):
        return True
    tokens = summary.split()
    return any(token[:1].islower() for token in tokens[1:])


def _evidence_label(evidence_type: str) -> str:
    labels = {
        "supported_fact": "[FACT]",
        "inference": "[INFERENCE]",
        "speculation_or_watch_item": "[WATCH]",
    }
    return labels.get(evidence_type, "[EVIDENCE]")


def _markdown_links(markdown: str) -> list[str]:
    return re.findall(r"\[[^\]]+\]\([^)]+\)", markdown)
