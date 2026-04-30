from __future__ import annotations

from typing import Literal

from research_pipeline.briefing.models import (
    AccessMethod,
    BriefingCluster,
    IntelligenceEvent,
    SourceClass,
    SourcePolicy,
)
from research_pipeline.briefing.report import (
    render_daily_brief,
    render_weekly_synthesis,
)


def _event(
    event_id: str,
    source_id: str,
    source_class: SourceClass,
    *,
    summary_hint: str = "Detailed implementation notes",
) -> IntelligenceEvent:
    return IntelligenceEvent(
        event_id=event_id,
        source_name=f"Source {source_id}",
        source_id=source_id,
        source_type=source_class,
        source_policy=SourcePolicy.PUBLIC_OFFICIAL,
        item_type="release",
        canonical_url=f"https://example.com/{event_id}",
        title=f"Event {event_id}",
        retrieved_at="2026-06-12T00:00:00Z",
        collection_method=AccessMethod.API,
        content_hash=f"content-{event_id}",
        dedup_key=f"dedup-{event_id}",
        published_at="2026-06-12T00:00:00Z",
        summary_hint=summary_hint,
        confidence="high",
    )


def _cluster(
    cluster_id: str,
    source_class: SourceClass,
    *,
    suggested_action: Literal["read", "try", "watch", "ignore"] = "read",
    primary_artifact_present: bool = True,
) -> BriefingCluster:
    event = _event(
        event_id=f"{cluster_id}:e1",
        source_id=f"src-{cluster_id}",
        source_class=source_class,
    )
    return BriefingCluster(
        cluster_id=cluster_id,
        title=f"Cluster {cluster_id}",
        primary_event_id=event.event_id,
        event_ids=(event.event_id,),
        topic_ids=(f"topic_{cluster_id}",),
        canonical_urls=(event.canonical_url,),
        first_seen_at="2026-06-12T00:00:00Z",
        last_seen_at="2026-06-12T00:00:00Z",
        source_classes=(source_class,),
        primary_artifact_present=primary_artifact_present,
        suggested_action=suggested_action,
        ranking_explanation="class=3.00; trust=1.00",
        events=(event,),
    )


def test_render_daily_brief_contains_required_sections_and_items() -> None:
    cluster = _cluster("c1", SourceClass.PRIMARY_ARTIFACT, suggested_action="try")

    markdown = render_daily_brief([cluster], run_date="2026-06-12")

    assert "# 🧠 Daily AI Intelligence Brief — 2026-06-12" in markdown
    assert "## 📑 Contents" in markdown
    assert "## 🔥 Executive Signal" in markdown
    assert "## ⭐ Top Items" in markdown
    assert "## 🗒️ Feedback Targets" in markdown
    assert "Cluster c1" in markdown
    assert "research-pipeline brief feedback --cluster c1 --signal keep" in markdown
    # Old boilerplate should be gone.
    assert "## Agent Read Map" not in markdown
    assert "## Follow-up Queue" not in markdown
    assert "## Suppressed / Not Reported" not in markdown


def test_render_daily_brief_links_executive_signal_to_top_items() -> None:
    cluster = _cluster("c1", SourceClass.PRIMARY_ARTIFACT, suggested_action="read")

    markdown = render_daily_brief([cluster], run_date="2026-06-12")

    # The executive-signal bullet should anchor-link into the Top Items entry.
    assert "[Cluster c1](#1-cluster-c1)" in markdown


def test_render_daily_brief_includes_previous_brief_link_when_provided() -> None:
    cluster = _cluster("c1", SourceClass.PRIMARY_ARTIFACT)

    markdown = render_daily_brief(
        [cluster],
        run_date="2026-06-12",
        previous_brief_link="../../2026-06-11/reports/daily.md",
    )

    assert "← Previous brief" in markdown
    assert "../../2026-06-11/reports/daily.md" in markdown


def test_render_daily_brief_low_signal_day_message_when_under_six_items() -> None:
    clusters = [
        _cluster(f"c{index}", SourceClass.PRIMARY_ARTIFACT) for index in range(3)
    ]

    markdown = render_daily_brief(clusters, run_date="2026-06-12")

    assert "Low-signal day" in markdown


def test_render_daily_brief_no_news_variant_for_empty_clusters() -> None:
    markdown = render_daily_brief([], run_date="2026-06-12", quiet_sources=["arxiv"])

    assert "No ranked items passed the inclusion threshold today." in markdown
    assert "No primary artifact passed the daily inclusion threshold." in markdown
    assert "Re-run after the next scheduled source cadence." in markdown
    assert "🔇 Quiet sources" in markdown


def test_render_weekly_synthesis_includes_links_and_applies_cap() -> None:
    reports = [
        "[A](https://example.com/a) [B](https://example.com/b)",
        "[C](https://example.com/c)",
    ]

    markdown = render_weekly_synthesis(reports, week_id="2026-W24", max_links=2)

    assert "# Weekly AI Intelligence Synthesis - 2026-W24" in markdown
    assert "[A](https://example.com/a)" in markdown
    assert "[B](https://example.com/b)" in markdown
    assert "[C](https://example.com/c)" not in markdown
