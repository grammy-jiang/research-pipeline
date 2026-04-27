from pathlib import Path

import pytest

from research_pipeline.briefing.feedback import BriefingFeedbackStore
from research_pipeline.briefing.io import read_jsonl
from research_pipeline.briefing.layout import resolve_briefing_paths
from research_pipeline.briefing.models import (
    BriefingCluster,
    FeedbackSignal,
    IntelligenceEvent,
)
from research_pipeline.briefing.normalize import canonicalize_url, stable_hash
from research_pipeline.briefing.obsidian import export_daily_note
from research_pipeline.briefing.preference_update import rollback_preference_adjustment
from research_pipeline.briefing.registry import load_source_registry
from research_pipeline.briefing.topic_memory import TopicMemoryStore
from research_pipeline.briefing.validate import validate_dossier_report
from research_pipeline.briefing.workflow import run_briefing
from research_pipeline.mcp_server import resources

FIXTURES = Path(__file__).parents[1] / "fixtures" / "briefing"


def test_stable_hash_and_url_normalization_are_deterministic() -> None:
    assert stable_hash("a", "b") == stable_hash("a", "b")
    assert (
        canonicalize_url("HTTPS://Example.com/foo/?utm_source=x&b=2&a=1#frag")
        == "https://example.com/foo?a=1&b=2"
    )


def test_offline_briefing_run_writes_valid_artifacts(tmp_path: Path) -> None:
    registry = load_source_registry(FIXTURES / "registry.json")

    paths, validation = run_briefing(
        registry,
        workspace=tmp_path,
        run_date="2026-04-27",
        fixture_base_dir=FIXTURES,
    )

    assert validation["passed"] is True
    assert paths.source_snapshot_path.exists()
    assert paths.events_path.exists()
    assert paths.ranked_clusters_path.exists()
    assert paths.daily_report_path.exists()
    assert paths.validation_path.exists()
    clusters = read_jsonl(paths.ranked_clusters_path, BriefingCluster)
    assert len(clusters) >= 2
    assert "## Agent Read Map" in paths.daily_report_path.read_text(encoding="utf-8")


def test_daily_report_uses_best_evidence_and_quality_labels(tmp_path: Path) -> None:
    registry = load_source_registry(FIXTURES / "registry.json")

    paths, validation = run_briefing(
        registry,
        workspace=tmp_path,
        run_date="2026-04-27",
        fixture_base_dir=FIXTURES,
    )
    markdown = paths.daily_report_path.read_text(encoding="utf-8")

    assert validation["passed"] is True
    assert "[FACT]" in markdown
    assert "| Confidence | high |" in markdown
    assert "| Suggested action | try |" in markdown
    assert "Adds MCP workflow improvements" in markdown
    assert "Duplicate release mention" not in markdown
    assert "Previous release." not in markdown
    assert "status: validated" in markdown


def test_repeated_topic_is_not_reported_as_new_on_second_run(tmp_path: Path) -> None:
    registry = load_source_registry(FIXTURES / "registry.json")
    run_briefing(
        registry,
        workspace=tmp_path,
        run_date="2026-04-27",
        fixture_base_dir=FIXTURES,
    )

    paths, validation = run_briefing(
        registry,
        workspace=tmp_path,
        run_date="2026-04-27",
        fixture_base_dir=FIXTURES,
    )

    assert validation["passed"] is True
    markdown = paths.daily_report_path.read_text(encoding="utf-8")
    assert "| Novelty | active |" in markdown


def test_no_news_day_is_valid_low_signal_output(tmp_path: Path) -> None:
    registry = load_source_registry(FIXTURES / "empty_registry.json")

    paths, validation = run_briefing(
        registry,
        workspace=tmp_path,
        run_date="2026-04-27",
        fixture_base_dir=FIXTURES,
    )

    assert validation["passed"] is True
    assert validation["metrics"]["item_count"] == 0
    assert "No primary artifact passed" in paths.daily_report_path.read_text(
        encoding="utf-8"
    )


def test_feedback_store_records_explicit_feedback(tmp_path: Path) -> None:
    store = BriefingFeedbackStore(tmp_path / "feedback.db")
    try:
        feedback = store.record(
            target_type="cluster",
            target_id="cluster_abc",
            signal=FeedbackSignal.KEEP,
            reason="useful",
        )
        weights = store.weights_by_target()
    finally:
        store.close()

    assert feedback.feedback_id.startswith("feedback_")
    assert weights["cluster:cluster_abc"] > 0


def test_obsidian_export_rejects_path_escape(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        export_daily_note("hello", vault_root=tmp_path / "vault", run_date="../escape")


def test_cli_brief_run_offline(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from research_pipeline.cli.app import app

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "brief",
            "run",
            "--registry",
            str(FIXTURES / "registry.json"),
            "--workspace",
            str(tmp_path),
            "--date",
            "2026-04-27",
            "--fixture-base-dir",
            str(FIXTURES),
        ],
    )

    assert result.exit_code == 0, result.output
    paths = resolve_briefing_paths(tmp_path, "2026-04-27")
    assert paths.daily_report_path.exists()


def test_source_expansion_requires_review_metadata(tmp_path: Path) -> None:
    registry = load_source_registry(FIXTURES / "bad_expansion_registry.json")

    with pytest.raises(ValueError, match="enablement review"):
        run_briefing(
            registry,
            workspace=tmp_path,
            run_date="2026-04-27",
            fixture_base_dir=FIXTURES,
        )


def test_reviewed_source_expansion_adapter_runs_offline(tmp_path: Path) -> None:
    registry = load_source_registry(FIXTURES / "expansion_registry.json")

    paths, validation = run_briefing(
        registry,
        workspace=tmp_path,
        run_date="2026-04-27",
        fixture_base_dir=FIXTURES,
    )

    assert validation["passed"] is True
    assert paths.root.joinpath("workflow_state.json").exists()
    events = read_jsonl(paths.events_path, IntelligenceEvent)
    assert events[0].collection_method.value == "hacker_news"


def test_preference_adjustment_can_rollback(tmp_path: Path) -> None:
    db_path = tmp_path / "feedback.db"
    store = BriefingFeedbackStore(db_path)
    try:
        for index in range(3):
            store.record(
                target_type="source",
                target_id="source-a",
                signal=FeedbackSignal.MORE_LIKE_THIS,
                reason=f"useful-{index}",
            )
        adjustments = store.create_adjustments(min_feedback=3)
    finally:
        store.close()

    assert adjustments
    result = rollback_preference_adjustment(
        db_path, str(adjustments[0]["adjustment_id"])
    )
    assert result["rolled_back"] is True


def test_feedback_conflicts_are_detected(tmp_path: Path) -> None:
    store = BriefingFeedbackStore(tmp_path / "feedback.db")
    try:
        store.record(
            target_type="cluster",
            target_id="cluster-x",
            signal=FeedbackSignal.KEEP,
        )
        store.record(
            target_type="cluster",
            target_id="cluster-x",
            signal=FeedbackSignal.HIDE,
        )
        conflicts = store.conflict_summary()
    finally:
        store.close()

    assert conflicts["cluster:cluster-x"] == {"positive": 1, "negative": 1}


def test_topic_alias_suggestions_are_reviewable(tmp_path: Path) -> None:
    registry = load_source_registry(FIXTURES / "registry.json")
    paths, _validation = run_briefing(
        registry,
        workspace=tmp_path,
        run_date="2026-04-27",
        fixture_base_dir=FIXTURES,
    )
    cluster = read_jsonl(paths.ranked_clusters_path, BriefingCluster)[0]
    store = TopicMemoryStore(paths.root / "memory" / "topics.db")
    try:
        suggestion = store.suggest_alias(cluster.topic_ids[0], "Claude coding agent")
        assert suggestion is not None
        reviewed = store.review_alias_suggestion(
            suggestion.suggestion_id,
            approve=True,
            review_record="test approval",
        )
        memory = store.get(cluster.topic_ids[0])
    finally:
        store.close()

    assert reviewed.status == "approved"
    assert memory is not None
    assert "claude coding agent" in memory.aliases


def test_dossier_validator_requires_evidence_and_labels() -> None:
    result = validate_dossier_report("# Bad dossier\n\n## Agent Read Map\n")

    assert result.passed is False
    assert any("evidence URL" in error for error in result.errors)


def test_cli_auto_dossier_and_resume(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from research_pipeline.cli.app import app

    runner = CliRunner()
    run_result = runner.invoke(
        app,
        [
            "brief",
            "run",
            "--registry",
            str(FIXTURES / "registry.json"),
            "--workspace",
            str(tmp_path),
            "--date",
            "2026-04-27",
            "--fixture-base-dir",
            str(FIXTURES),
        ],
    )
    assert run_result.exit_code == 0, run_result.output
    dossier_result = runner.invoke(
        app,
        [
            "brief",
            "dossier",
            "--auto",
            "--workspace",
            str(tmp_path),
            "--date",
            "2026-04-27",
        ],
    )
    assert dossier_result.exit_code == 0, dossier_result.output
    resume_result = runner.invoke(
        app,
        [
            "brief",
            "resume",
            "--from-stage",
            "validate",
            "--workspace",
            str(tmp_path),
            "--date",
            "2026-04-27",
        ],
    )
    assert resume_result.exit_code == 0, resume_result.output
    assert list(
        (tmp_path / "briefings" / "2026-04-27" / "reports" / "dossiers").glob("*.md")
    )


def test_briefing_mcp_resources_read_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = load_source_registry(FIXTURES / "registry.json")
    run_briefing(
        registry,
        workspace=tmp_path,
        run_date="2026-04-27",
        fixture_base_dir=FIXTURES,
    )
    monkeypatch.setattr(resources, "DEFAULT_WORKSPACE", str(tmp_path))

    assert "2026-04-27" in resources.list_briefings()
    assert "# Daily AI Intelligence Brief" in resources.get_briefing_daily("2026-04-27")
    assert "rank_score" in resources.get_briefing_ranked("2026-04-27")
    assert "workflow_state" not in resources.get_briefing_workflow_state("missing")
