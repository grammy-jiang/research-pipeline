"""Workflow gates reject stale, invalid, or failed work without live services."""

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2] / "src/research_pipeline/skill_data/research-pipeline"
spec = importlib.util.spec_from_file_location(
    "reliability_runner", ROOT / "runners/runner.py"
)
assert spec and spec.loader
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_schema_is_enforced(tmp_path):
    plan = tmp_path / "plan.json"
    plan.write_text('{"topic":"old incorrect field"}')
    task = {
        "output": {"path": str(plan), "schema": "schemas/query_plan.schema.json"},
        "validation": ["artifact_exists", "schema_valid"],
    }
    assert runner.validation_errors(task, {"cwd": str(tmp_path)}, {"success": True})


def test_empty_candidate_file_cannot_pass(tmp_path):
    path = tmp_path / "candidates.jsonl"
    path.write_text("")
    task = {"output": {"path": str(path)}, "validation": ["non_empty"]}
    assert runner.validation_errors(task, {"cwd": str(tmp_path)}, {"success": True})


def test_missing_execution_receipt_cannot_pass():
    task = {"validation": ["exit_code_zero"]}
    assert runner.validation_errors(task, {}, {})
    assert runner.validation_errors(task, {}, {"success": False, "exit_code": 1})
    assert runner.validation_errors(task, {}, {"success": True, "exit_code": 0}) == []


def test_rejected_reviewer_cannot_pass(tmp_path):
    verdict = tmp_path / "review.json"
    verdict.write_text(
        json.dumps(
            {
                "reviewer_task_id": "review-synthesis",
                "target_artifact": "report.md",
                "status": "rejected",
            }
        )
    )
    task = {
        "id": "review-synthesis",
        "executor": {"kind": "llm_reviewer"},
        "output": {"path": str(verdict)},
        "validation": ["artifact_exists"],
    }
    assert runner.validation_errors(task, {"cwd": str(tmp_path)}, {"success": True})


def test_failed_last_task_is_not_complete(tmp_path):
    manifest = {
        "tasks": [
            {
                "id": "gate",
                "executor": {"kind": "deterministic_script"},
                "failure_policy": {"on_failure": "block"},
            }
        ]
    }
    state = {"context": {"cwd": str(tmp_path)}, "tasks": {"gate": {"status": "failed"}}}
    assert (
        runner.run_workflow(manifest, state, tmp_path / "state.json", "standard", False)
        == 1
    )
    assert state["status"] == "blocked"


def test_delegated_gate_requires_result_even_without_output(tmp_path):
    manifest = {
        "tasks": [
            {
                "id": "verify",
                "executor": {"kind": "deterministic_mcp_tool"},
                "validation": ["exit_code_zero"],
            }
        ]
    }
    state = {
        "context": {"cwd": str(tmp_path)},
        "tasks": {"verify": {"status": "delegated"}},
    }
    assert (
        runner.run_workflow(manifest, state, tmp_path / "state.json", "standard", False)
        == 0
    )
    assert state["tasks"]["verify"]["status"] == "delegated"
    assert state.get("status") != "complete"


def test_publication_requires_matching_passed_validation(tmp_path):
    pub_spec = importlib.util.spec_from_file_location(
        "publish", ROOT / "scripts/publish_report.py"
    )
    assert pub_spec and pub_spec.loader
    pub = importlib.util.module_from_spec(pub_spec)
    pub_spec.loader.exec_module(pub)
    draft = tmp_path / "draft.md"
    draft.write_text("a reviewed draft")
    validation = tmp_path / "validation.json"
    validation.write_text(json.dumps({"passed": True, "report_sha256": "stale"}))
    final = tmp_path / "final.md"
    with pytest.raises(ValueError, match="hash"):
        pub.publish(draft, validation, final)
    assert not final.exists()


def test_validate_cli_fails_for_missing_report(tmp_path):
    from typer.testing import CliRunner

    from research_pipeline.cli.app import app

    result = CliRunner().invoke(
        app, ["validate", "--report", str(tmp_path / "missing.md")]
    )
    assert result.exit_code == 1


def test_failed_plan_records_failure_without_requiring_run_id(tmp_path):
    manifest = {
        "tasks": [
            {
                "id": "plan",
                "executor": {"kind": "deterministic_mcp_tool"},
                "failure_policy": {"on_failure": "block", "retries": 1},
            }
        ]
    }
    state = {
        "context": {"cwd": str(tmp_path)},
        "tasks": {"plan": {"status": "delegated", "attempts": 1}},
    }
    result = {"success": False, "exit_code": 42}
    assert (
        runner.submit_task(manifest, state, tmp_path / "state.json", "plan", result)
        == 1
    )
    assert state["tasks"]["plan"]["status"] == "failed"
    assert state["tasks"]["plan"]["result"]["exit_code"] == 42
    with pytest.raises(ValueError, match="not currently delegated"):
        runner.execute_delegated(manifest, state, tmp_path / "state.json", "plan")


def test_plan_cannot_accept_another_run(tmp_path):
    manifest = {"tasks": [{"id": "plan", "failure_policy": {"on_failure": "block"}}]}
    state = {
        "run_id": "expected",
        "context": {"cwd": str(tmp_path)},
        "tasks": {"plan": {"status": "delegated", "attempts": 1}},
    }
    assert (
        runner.submit_task(
            manifest,
            state,
            tmp_path / "state.json",
            "plan",
            {"success": True, "artifacts": {"run_id": "different"}},
        )
        == 1
    )
    assert state["tasks"]["plan"]["status"] == "failed"
    assert state["run_id"] == "expected"


def test_runner_uses_workspace_environment(monkeypatch, tmp_path):
    workspace = tmp_path / "separate"
    monkeypatch.setenv("RESEARCH_PIPELINE_WORKSPACE", str(workspace))
    assert runner.workflow_context({"run_id": "fixture"})["run_dir"] == str(
        workspace / "fixture"
    )


def test_resume_snapshots_keep_published_report_and_seed_context(tmp_path):
    report = tmp_path / "topic-research-report.md"
    report.write_text(
        "# Prior report\n[2401.12345]\n## Research Gaps\n- [ACADEMIC] unanswered\n"
    )
    manifest = {
        "tasks": [
            {
                "id": "resume-check",
                "executor": {
                    "kind": "deterministic_script",
                    "command": (
                        'python3 "{skill_dir}/scripts/resume_check.py" topic "{cwd}"'
                    ),
                },
                "output": {"path": "{cwd}/resume_context.json"},
                "validation": ["artifact_exists", "exit_code_zero"],
            }
        ]
    }
    state = {"topic_slug": "topic", "context": {"cwd": str(tmp_path)}}
    assert (
        runner.run_workflow(manifest, state, tmp_path / "state.json", "standard", False)
        == 0
    )
    assert state["context"]["prior_paper_ids"] == ["2401.12345"]
    assert state["context"]["prior_gaps"]
    assert report.exists()
    assert len(list(tmp_path.glob("topic-research-report.*.md"))) == 1


def test_synthesis_corpus_must_come_from_shortlist(tmp_path):
    (tmp_path / "screen").mkdir()
    (tmp_path / "screen/shortlist.json").write_text(
        json.dumps([{"paper": {"arxiv_id": "2609.00001", "version": "v1"}}])
    )
    synthesis = tmp_path / "synthesis.json"
    synthesis.write_text(
        json.dumps(
            {
                "corpus": [{"paper_id": "2609.99999"}],
                "taxonomy": [{"supporting_papers": ["2609.99999"]}],
            }
        )
    )
    task = {
        "id": "paper-synthesizer",
        "output": {"path": str(synthesis)},
        "validation": ["evidence_present"],
    }
    errors = runner.validation_errors(
        task, {"run_dir": str(tmp_path)}, {"success": True}
    )
    assert any("outside the shortlist" in error for error in errors)


def test_llm_screener_cannot_accept_untouched_heuristic_output(tmp_path):
    shortlist = tmp_path / "shortlist.json"
    shortlist.write_text(
        json.dumps(
            [{"paper": {"abstract": "An email study"}, "download": True, "llm": None}]
        )
    )
    task = {
        "id": "paper-screener",
        "output": {"path": str(shortlist)},
        "validation": ["evidence_present"],
    }
    errors = runner.validation_errors(task, {}, {"success": True})
    assert any("relevance judgment" in error for error in errors)
