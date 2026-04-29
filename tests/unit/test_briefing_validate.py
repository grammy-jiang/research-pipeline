from __future__ import annotations

from research_pipeline.briefing.models import (
    AccessMethod,
    BriefingCluster,
    IntelligenceEvent,
    SourceClass,
    SourcePolicy,
)
from research_pipeline.briefing.validate import (
    validate_daily_report,
    validate_dossier_report,
    validate_obsidian_note,
    validation_to_json,
)


def _event(
    event_id: str,
    source_id: str,
    source_class: SourceClass,
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
        summary_hint="Detailed reproducibility package",
        confidence="high",
    )


def _cluster(
    cluster_id: str,
    source_class: SourceClass,
    evidence_type: str = "supported_fact",
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
        primary_artifact_present=True,
        evidence_type=evidence_type,
        ranking_explanation="class=3.00; trust=1.00",
        events=(event,),
    )


def test_validate_daily_report_passes_valid_report() -> None:
    clusters = [_cluster(f"c{i}", SourceClass.PRIMARY_ARTIFACT) for i in range(6)]
    words = " ".join(["word"] * 1000)
    markdown = (
        "---\ntype: daily-brief\n---\n"
        "# Daily Brief\n"
        "## Agent Read Map\n"
        "| Field | Value |\n|---|---|\n"
        "| Brief | Active day with six items |\n"
        "## Executive Signal\n"
        + " ".join(f"[FACT] Signal {i}: " for i in range(3))
        + "https://example.com/c0:e1 item.\n"
        "## Top Items\n"
        + "\n".join(
            f"### {i + 1}. {cluster.title}\n"
            f"Evidence: [{cluster.title}]({cluster.canonical_urls[0]})\n"
            for i, cluster in enumerate(clusters)
        )
        + "## Suppressed / Not Reported\nNone.\n"
        "## Follow-up Queue\n"
        + "\n".join(
            f"- [{cluster.title}]({cluster.canonical_urls[0]})"
            for cluster in clusters[:3]
        )
        + "\n"
        "## Feedback Targets\n"
        "| target | id | cmd |\n|---|---|---|\n"
        + " ".join(
            f"| {cluster.title} | {cluster.cluster_id} | cmd |" for cluster in clusters
        )
        + "\n"
        + words
    )

    result = validate_daily_report(markdown, clusters)

    assert result.passed, f"Errors: {result.errors}, Warnings: {result.warnings}"


def test_validate_daily_report_fails_missing_required_sections() -> None:
    cluster = _cluster("c1", SourceClass.PRIMARY_ARTIFACT)
    markdown = "# Daily Brief\n## Top Items\nNo content.\n"

    result = validate_daily_report(markdown, [cluster])

    assert not result.passed
    assert any("missing required section" in err for err in result.errors)


def test_validate_daily_report_enforces_link_budget() -> None:
    cluster = _cluster("c1", SourceClass.PRIMARY_ARTIFACT)
    markdown = (
        "---\ntype: daily-brief\n---\n"
        "# Daily Brief\n"
        + "\n".join(f"## Section {i}\n" for i in range(6))
        + "\n"
        + " ".join(f"[{i}](https://example.com/{i})" for i in range(20))
        + "\n"
        + ("word " * 300)
    )

    result = validate_daily_report(markdown, [cluster], max_links=15)

    assert not result.passed
    assert any("link count" in err and "exceeds budget" in err for err in result.errors)


def test_validate_daily_report_enforces_word_count_on_active_days() -> None:
    clusters = [_cluster(f"c{i}", SourceClass.PRIMARY_ARTIFACT) for i in range(6)]
    markdown = (
        "---\ntype: daily-brief\n---\n"
        "# Brief\n"
        "## Agent Read Map\n"
        "| Field | Value |\n|---|---|\n"
        "## Executive Signal\nSignal.\n"
        "## Top Items\nItems.\n"
        "## Suppressed / Not Reported\nNone.\n"
        "## Follow-up Queue\nQueue.\n"
        "## Feedback Targets\n"
        "| target | id | cmd |\n|---|---|---|\n"
    )

    result = validate_daily_report(markdown, clusters, active_min_words=900)

    assert not result.passed
    assert any("word count" in err for err in result.errors)


def test_validate_daily_report_detects_duplicate_cluster_titles() -> None:
    cluster1 = _cluster("c1", SourceClass.PRIMARY_ARTIFACT)
    cluster2_dup = _cluster("c2", SourceClass.PRIMARY_ARTIFACT)
    cluster2_dup = cluster2_dup.model_copy(update={"title": cluster1.title})

    markdown = (
        "---\ntype: daily-brief\n---\n"
        "# Brief\n"
        "## Agent Read Map\n"
        "| Field | Value |\n|---|---|\n"
        "## Executive Signal\nSignal.\n"
        "## Top Items\n"
        f"[{cluster1.title}](https://example.com/c1:e1)\n"
        f"[{cluster2_dup.title}](https://example.com/c2:e1)\n"
        "## Suppressed / Not Reported\nNone.\n"
        "## Follow-up Queue\nQueue.\n"
        "## Feedback Targets\n| target | id | cmd |\n|---|---|---|\n" + ("word " * 300)
    )

    result = validate_daily_report(markdown, [cluster1, cluster2_dup])

    assert not result.passed
    assert any("duplicate cluster titles" in err for err in result.errors)


def test_validate_daily_report_verifies_evidence_urls_in_markdown() -> None:
    cluster = _cluster("c1", SourceClass.PRIMARY_ARTIFACT)
    markdown = (
        "---\ntype: daily-brief\n---\n"
        "# Brief\n"
        "## Agent Read Map\n"
        "| Field | Value |\n|---|---|\n"
        "## Executive Signal\nSignal.\n"
        "## Top Items\nNo URL here.\n"
        "## Suppressed / Not Reported\nNone.\n"
        "## Follow-up Queue\nQueue.\n"
        "## Feedback Targets\n| target | id | cmd |\n|---|---|---|\n" + ("word " * 300)
    )

    result = validate_daily_report(markdown, [cluster])

    assert not result.passed
    assert any("evidence URL missing" in err for err in result.errors)


def test_validate_daily_report_checks_evidence_type_labels() -> None:
    cluster = _cluster("c1", SourceClass.PRIMARY_ARTIFACT, evidence_type="inference")
    markdown = (
        "---\ntype: daily-brief\n---\n"
        "# Brief\n"
        "## Agent Read Map\n"
        "| Field | Value |\n|---|---|\n"
        "## Executive Signal\n[FACT] Signal https://example.com/c1:e1.\n"
        "## Top Items\nContent https://example.com/c1:e1.\n"
        "## Suppressed / Not Reported\nNone.\n"
        "## Follow-up Queue\nQueue.\n"
        "## Feedback Targets\n| target | id | cmd |\n|---|---|---|\n"
        "| Cluster c1 | c1 | cmd |\n" + ("word " * 300)
    )

    result = validate_daily_report(markdown, [cluster])

    assert not result.passed
    assert any("[INFERENCE]" in err for err in result.errors)


def test_validate_daily_report_flags_low_signal_day_without_no_material_updates() -> (
    None
):
    clusters = [_cluster(f"c{i}", SourceClass.PRIMARY_ARTIFACT) for i in range(3)]
    markdown = (
        "---\ntype: daily-brief\n---\n"
        "# Brief\n"
        "## Agent Read Map\n"
        "| Field | Value |\n|---|---|\n"
        "## Executive Signal\nSignal.\n"
        "## Top Items\nItems.\n"
        "## Suppressed / Not Reported\nNone.\n"
        "## Follow-up Queue\nQueue.\n"
        "## Feedback Targets\n| target | id | cmd |\n|---|---|---|\n"
    )

    result = validate_daily_report(markdown, clusters)

    assert not result.passed
    assert any("No Material Updates" in err for err in result.errors)


def test_validate_daily_report_rejects_boilerplate_content() -> None:
    clusters = [_cluster(f"c{i}", SourceClass.PRIMARY_ARTIFACT) for i in range(6)]
    markdown = (
        "---\ntype: daily-brief\n---\n"
        "# Brief\n"
        "## Agent Read Map\n"
        "| Field | Value |\n|---|---|\n"
        "## Executive Signal\nDuplicate release mention here.\n"
        "## Top Items\nItems.\n"
        "## Suppressed / Not Reported\nNone.\n"
        "## Follow-up Queue\nQueue.\n"
        "## Feedback Targets\n| target | id | cmd |\n|---|---|---|\n" + ("word " * 300)
    )

    result = validate_daily_report(markdown, clusters)

    assert not result.passed
    assert any("boilerplate" in err for err in result.errors)


def test_validate_dossier_report_passes_valid_dossier() -> None:
    markdown = (
        "---\ntype: dossier\n---\n"
        "# Dossier\n"
        "## Agent Read Map\n"
        "| Field | Value |\n|---|---|\n"
        "## One-paragraph Summary\nSummary.\n"
        "## What Changed\nChanged.\n"
        "## Why It Matters Technically\nMatters.\n"
        "## Evidence Timeline\nTimeline.\n"
        "## Artifacts To Open\n[Link](https://example.com/artifact)\n"
        "## Open Questions\nQuestions.\n"
        "## Agent Notes\nfactuality_label=supported_fact\n"
    )

    result = validate_dossier_report(markdown)

    assert result.passed


def test_validate_dossier_report_fails_missing_sections() -> None:
    markdown = "# Dossier\n## What Changed\nChanged.\n"

    result = validate_dossier_report(markdown)

    assert not result.passed
    assert any("missing required section" in err for err in result.errors)


def test_validate_obsidian_note_passes_valid_note() -> None:
    markdown = (
        "---\ntype: topic-brief\n---\n"
        "# Topic\n"
        "## Agent Read Map\n"
        "| Field | Value |\n|---|---|\n"
    )

    result = validate_obsidian_note(markdown, "topic-brief")

    assert result.passed


def test_validate_obsidian_note_fails_missing_frontmatter() -> None:
    markdown = "# Topic\n## Agent Read Map\nContent.\n"

    result = validate_obsidian_note(markdown, "topic-brief")

    assert not result.passed
    assert any("frontmatter" in err for err in result.errors)


def test_validation_result_serializes_to_json() -> None:
    cluster = _cluster("c1", SourceClass.PRIMARY_ARTIFACT)
    markdown = (
        "---\ntype: daily-brief\n---\n"
        "# Brief\n"
        "## Agent Read Map\n"
        "| Field | Value |\n|---|---|\n"
        "## Executive Signal\n[FACT] Signal https://example.com/c1:e1.\n"
        "## Top Items\nContent.\n"
        "## Suppressed / Not Reported\nNone.\n"
        "## Follow-up Queue\nQueue.\n"
        "## Feedback Targets\n| target | id | cmd |\n|---|---|---|\n" + ("word " * 300)
    )

    result = validate_daily_report(markdown, [cluster])
    json_dict = validation_to_json(result)

    assert "passed" in json_dict
    assert "errors" in json_dict
    assert "metrics" in json_dict
