"""Release review regressions: configuration and mandatory publication evidence."""

import asyncio
import hashlib
import importlib.util
import json
import re
from pathlib import Path
from unittest.mock import patch

import pytest

from research_pipeline.config.models import PipelineConfig
from research_pipeline.models.summary import CrossPaperSynthesisRecord
from research_pipeline.pipeline.query_planning import build_query_plan
from research_pipeline.summarization.report_templates import render_report
from research_pipeline.summarization.report_validation import validate_report

SKILL = Path(__file__).parents[2] / "src/research_pipeline/skill_data/research-pipeline"


def _report() -> str:
    record = CrossPaperSynthesisRecord(
        topic="Offline release fixture",
        corpus=[
            {
                "paper_id": "p1",
                "title": "Offline paper",
                "year": "2026",
                "venue": "Fixture",
            }
        ],
    )
    return render_report(record, "structured_synthesis")


def test_plan_uses_configured_sparsity_thresholds():
    config = PipelineConfig()
    config.search.min_candidates = 100
    config.search.min_highscore = 12
    config.search.min_downloads = 7
    plan = build_query_plan("email conversation reconstruction", config)
    assert plan.sparsity_thresholds.model_dump() == {
        "min_candidates": 100,
        "min_highscore": 12,
        "min_downloads": 7,
    }


def test_watch_honors_configured_request_interval(tmp_path):
    from research_pipeline.cli.cmd_watch import watch_command

    queries = tmp_path / "queries.json"
    queries.write_text(json.dumps([{"name": "offline", "query": "offline"}]))
    config = tmp_path / "config.toml"
    config.write_text("[arxiv]\nmin_interval_seconds = 70.0\n")
    with patch("research_pipeline.cli.cmd_watch.ArxivClient") as client:
        watch_command(queries_file=queries, config_path=config)
    session = client.call_args.kwargs["session"]
    assert session.budget.min_interval == 70.0


def test_structured_report_meets_required_format(tmp_path):
    report = tmp_path / "report.md"
    report.write_text(_report())
    result = validate_report(report, strict_format=True)
    assert result["verdict"] == "PASS"
    assert result["workflow_format_passed"] is True


@pytest.mark.parametrize("part", ["contents", "round_history", "mermaid", "latex"])
def test_strict_validation_rejects_each_missing_format_part(tmp_path, part):
    text = _report()
    fence = chr(96) * 3
    patterns = {
        "contents": r"(?ms)^## Contents\n.*?(?=^## )",
        "round_history": r"(?ms)^## Round History\n.*?(?=^## )",
        "mermaid": rf"(?s){fence}mermaid.*?{fence}",
        "latex": r"\$[^$]+\$",
    }
    changed, count = re.subn(patterns[part], "", text)
    assert count > 0
    report = tmp_path / "report.md"
    report.write_text(changed)
    result = validate_report(report, strict_format=True)
    assert result["verdict"] == "FAIL"
    assert result["workflow_format_passed"] is False


def test_publication_rechecks_actual_format(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "publication_review", SKILL / "scripts/publish_report.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    draft = tmp_path / "draft.md"
    draft.write_text("An incomplete draft")
    validation = tmp_path / "validation.json"
    validation.write_text(
        json.dumps(
            {
                "passed": True,
                "workflow_format_passed": True,
                "report_sha256": hashlib.sha256(draft.read_bytes()).hexdigest(),
            }
        )
    )
    final = tmp_path / "final.md"
    with pytest.raises(ValueError, match="format"):
        module.publish(draft, validation, final)
    assert not final.exists()


def test_mcp_validation_is_annotated_as_mutating():
    from research_pipeline.mcp_server.server import mcp

    tool = next(
        t for t in asyncio.run(mcp.list_tools()) if t.name == "tool_validate_report"
    )
    assert tool.annotations is not None
    assert tool.annotations.readOnlyHint is False


def test_renderer_preserves_closed_worker_mermaid_blocks(tmp_path):
    fence = chr(96) * 3
    finding = "Two paths.\n\n" + fence + "mermaid\nflowchart TD\n A --> B\n" + fence
    record = CrossPaperSynthesisRecord(
        topic="Fixture Markdown",
        taxonomy=[
            {
                "finding_id": "f1",
                "finding_type": "pattern",
                "finding": finding,
                "confidence": "LOW",
                "supporting_papers": ["p1"],
            }
        ],
    )
    report_text = render_report(record, "structured_synthesis")
    assert fence + " [p1]" not in report_text
    assert "\n" + fence + "\n" in report_text
    report = tmp_path / "report.md"
    report.write_text(report_text)
    assert validate_report(report, strict_format=True)["workflow_format_passed"]


def test_strict_validation_rejects_malformed_mermaid_closing_fence(tmp_path):
    fence = chr(96) * 3
    report = tmp_path / "report.md"
    report.write_text(
        _report()
        + "\n"
        + fence
        + "mermaid\nflowchart TD\n A --> B\n"
        + fence
        + " [p1]\n"
    )
    result = validate_report(report, strict_format=True)
    assert result["verdict"] == "FAIL"
    assert result["workflow_format_passed"] is False
