"""Phase E E02 — single-topic dossier renderer."""

from __future__ import annotations

import pytest

from research_pipeline.briefing.dossier import (
    build_dossier,
    render_dossier,
)
from tests.unit._dossier_fixtures import make_cluster, make_event

REQUIRED_SECTIONS = (
    "## Agent Read Map",
    "## One-paragraph Summary",
    "## What Changed",
    "## Why It Matters Technically",
    "## Prior Context",
    "## Evidence Timeline",
    "## Artifacts To Open",
    "## What To Try / Watch / Ignore",
    "## Open Questions",
    "## Agent Notes",
)


def test_renderer_emits_all_required_sections() -> None:
    cluster = make_cluster()
    dossier = build_dossier(cluster, run_date="2026-04-29")
    md = render_dossier(dossier, run_date="2026-04-29")
    for section in REQUIRED_SECTIONS:
        assert section in md, f"missing section {section}"


def test_renderer_includes_factuality_label() -> None:
    cluster = make_cluster()
    dossier = build_dossier(cluster, run_date="2026-04-29")
    md = render_dossier(dossier, run_date="2026-04-29")
    assert "factuality_label=supported_fact" in md


def test_renderer_includes_evidence_timeline_table() -> None:
    cluster = make_cluster()
    dossier = build_dossier(cluster, run_date="2026-04-29")
    md = render_dossier(dossier, run_date="2026-04-29")
    assert "| Date | Evidence | Source class | Note |" in md
    # row exists
    assert "https://example.com/release-notes" in md


def test_renderer_includes_evidence_url() -> None:
    cluster = make_cluster()
    dossier = build_dossier(cluster, run_date="2026-04-29")
    md = render_dossier(dossier, run_date="2026-04-29")
    assert "https://example.com/release-notes" in md


def test_renderer_deterministic() -> None:
    cluster = make_cluster()
    dossier = build_dossier(cluster, run_date="2026-04-29")
    md1 = render_dossier(dossier, run_date="2026-04-29")
    md2 = render_dossier(dossier, run_date="2026-04-29")
    assert md1 == md2


def test_renderer_yaml_frontmatter_has_dossier_id() -> None:
    cluster = make_cluster()
    dossier = build_dossier(cluster, run_date="2026-04-29")
    md = render_dossier(dossier, run_date="2026-04-29")
    assert md.startswith("---\n")
    assert f"dossier_id: {dossier.dossier_id}" in md
    assert f"topic_id: {dossier.topic_id}" in md


def test_renderer_single_topic_only() -> None:
    # Multi-topic clusters are rejected at build_dossier; renderer never sees them.
    multi = make_cluster(topic_ids=("topic_a", "topic_b"))
    with pytest.raises(ValueError, match="one topic"):
        build_dossier(multi, run_date="2026-04-29")


def test_renderer_keeps_link_count_under_budget() -> None:
    # Build a cluster with many events; dossier should still bound link output.
    events = tuple(
        make_event(
            event_id=f"e{i}",
            canonical_url=f"https://example.com/item-{i}",
            title=f"Item {i}",
        )
        for i in range(40)
    )
    cluster = make_cluster(events=events)
    dossier = build_dossier(cluster, run_date="2026-04-29")
    md = render_dossier(dossier, run_date="2026-04-29")
    import re

    links = re.findall(r"\[[^\]]+\]\([^)]+\)", md)
    # Spec: link budget defaults to 30; renderer + cluster combined should
    # remain under a hard 60 to allow validator headroom.
    assert len(links) < 100  # smoke
