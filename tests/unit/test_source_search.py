"""Search outcomes distinguish missing coverage from a successful empty query."""

from pathlib import Path
from unittest.mock import MagicMock

from research_pipeline.config.models import PipelineConfig
from research_pipeline.models.query_plan import QueryPlan
from research_pipeline.pipeline.source_search import execute_search


def query_plan() -> QueryPlan:
    return QueryPlan(
        topic_raw="email threads",
        topic_normalized="email threads",
        must_terms=["email"],
        query_variants=["all:email"],
        primary_months=6,
        fallback_months=12,
    )


def test_all_failed_is_not_a_success_and_has_no_fallback(tmp_path: Path) -> None:
    call = MagicMock(side_effect=RuntimeError("fixture"))
    report = execute_search(
        query_plan(),
        PipelineConfig(),
        ["arxiv"],
        tmp_path,
        handlers={"arxiv": call},
    )
    assert report.status == "failed"
    assert len(report.attempts) == 1
    assert call.call_count == 1
    assert report.candidates == []


def test_successful_empty_query_uses_fallback_and_remains_distinct(
    tmp_path: Path,
) -> None:
    call = MagicMock(return_value=[])
    report = execute_search(
        query_plan(),
        PipelineConfig(),
        ["arxiv"],
        tmp_path,
        handlers={"arxiv": call},
    )
    assert report.status == "empty"
    assert [a.window_months for a in report.attempts] == [6, 12]
    assert call.call_count == 2


def test_only_selected_sources_execute(tmp_path: Path) -> None:
    selected = MagicMock(return_value=[])
    disabled = MagicMock(side_effect=AssertionError("disabled source invoked"))
    report = execute_search(
        query_plan(),
        PipelineConfig(),
        ["dblp"],
        tmp_path,
        handlers={"dblp": selected, "scholar": disabled},
    )
    assert report.sources == ["dblp"]
    disabled.assert_not_called()
    # DBLP has no upstream date-window filter; do not repeat it as a date fallback.
    assert selected.call_count == 1


def test_explicit_source_queries_are_recorded_and_used(tmp_path: Path) -> None:
    plan = query_plan()
    plan.source_queries = {"semantic_scholar": ["email reassembly", "inline replies"]}
    plan.fallback_months = plan.primary_months
    call = MagicMock(return_value=[])
    report = execute_search(
        plan,
        PipelineConfig(),
        ["semantic_scholar"],
        tmp_path,
        handlers={"semantic_scholar": call},
    )
    assert [c.args[0].topic_raw for c in call.call_args_list] == [
        "email reassembly",
        "inline replies",
    ]
    assert [a.attempted_queries for a in report.attempts] == [
        ["email reassembly"],
        ["inline replies"],
    ]
