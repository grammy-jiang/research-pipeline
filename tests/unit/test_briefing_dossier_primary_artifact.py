"""Phase E E04 — primary artifact gate for dossier generation."""

from __future__ import annotations

import pytest

from research_pipeline.briefing.dossier import build_dossier
from tests.unit._dossier_fixtures import make_cluster, make_event


def test_missing_primary_artifact_rejected() -> None:
    cluster = make_cluster(primary_artifact_present=False)
    with pytest.raises(ValueError, match="primary artifact"):
        build_dossier(cluster, run_date="2026-04-29")


def test_error_message_includes_cluster_id() -> None:
    cluster = make_cluster(
        cluster_id="cluster_xyz",
        primary_artifact_present=False,
    )
    with pytest.raises(ValueError, match="cluster_xyz"):
        build_dossier(cluster, run_date="2026-04-29")


def test_empty_canonical_urls_rejected() -> None:
    # Build a cluster with primary_artifact=True but force empty canonical_urls.
    event = make_event()
    cluster = make_cluster(events=(event,), canonical_urls=(event.canonical_url,))
    cluster_no_urls = cluster.model_copy(update={"canonical_urls": ()})
    with pytest.raises(ValueError, match="canonical URL"):
        build_dossier(cluster_no_urls, run_date="2026-04-29")


def test_multi_topic_rejected() -> None:
    cluster = make_cluster(topic_ids=("topic_a", "topic_b"))
    with pytest.raises(ValueError, match="one topic"):
        build_dossier(cluster, run_date="2026-04-29")


def test_no_events_rejected() -> None:
    event = make_event()
    cluster = make_cluster(events=(event,))
    cluster_no_events = cluster.model_copy(update={"events": ()})
    with pytest.raises(ValueError, match="event"):
        build_dossier(cluster_no_events, run_date="2026-04-29")


def test_happy_path_succeeds() -> None:
    cluster = make_cluster()
    dossier = build_dossier(cluster, run_date="2026-04-29")
    assert dossier.cluster_ids == (cluster.cluster_id,)
    assert dossier.topic_id == "topic_acme"
