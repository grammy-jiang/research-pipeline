#!/usr/bin/env python3
"""
research-pipeline skill orchestrator.

Reads manifest.json, maintains workflow_state.json, executes deterministic
tasks, and delegates LLM worker/reviewer tasks to sub-agents. The orchestrator
is the single authority for task status transitions.

Usage:
  python3 runner.py "<topic>" [--run-id <ID>] [--profile standard] [--state STATE]
  python3 runner.py --status   # show current workflow state
  python3 runner.py --dry-run "<topic>"  # print task graph without executing
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SKILL_DIR = Path(__file__).parent.parent
MANIFEST_PATH = SKILL_DIR / "manifest.json"
CONTRACTS_DIR = Path(__file__).parent / "subagent_contracts"

# Valid task status values per the workflow state model.
TERMINAL_STATUSES = {"accepted", "skipped_by_policy", "blocked", "failed"}
# Statuses that satisfy a dependency (accepted = ran OK; skipped_by_policy = excluded
# from profile or skipped after non-blocking failure — both count as "done enough").
READY_STATUSES = {"accepted", "skipped_by_policy"}
LLM_KINDS = {"llm_worker", "llm_reviewer"}


# ---------------------------------------------------------------------------
# State I/O
# ---------------------------------------------------------------------------


def load_manifest() -> dict[str, Any]:
    return json.loads(MANIFEST_PATH.read_text())


def load_state(state_path: Path) -> dict[str, Any]:
    if state_path.exists():
        return json.loads(state_path.read_text())
    return {}


def save_state(state: dict[str, Any], state_path: Path) -> None:
    state_path.write_text(json.dumps(state, indent=2))
    state_path.with_suffix(".json.bak").write_text(json.dumps(state, indent=2))
    _write_round_state(state)


def _write_round_state(state: dict[str, Any]) -> None:
    """Write round_state.json into the workflow CWD for lifecycle hooks.

    stop-check.sh and resume-inject.sh look for this file to detect an active
    research session.  Without it both hooks are permanent no-ops.
    """
    cwd_str = state.get("context", {}).get("cwd", "")
    if not cwd_str:
        return
    cwd = Path(cwd_str)
    if not cwd.exists():
        return

    # Load classified gaps if available (written by classify-gaps task).
    open_gaps: list[Any] = []
    gaps_file = cwd / "gaps.json"
    if gaps_file.exists():
        try:
            gaps_data = json.loads(gaps_file.read_text())
            open_gaps = [
                g
                for g in gaps_data.get("gaps", [])
                if g.get("classification") != "OUT_OF_SCOPE"
            ]
        except (json.JSONDecodeError, OSError):
            pass

    round_state: dict[str, Any] = {
        "workflow_id": state.get("workflow_id", "research-pipeline"),
        "run_id": state.get("run_id", ""),
        "topic": state.get("topic", ""),
        "topic_slug": state.get("topic_slug", ""),
        "round": state.get("round", 1),
        "status": state.get("status", "running"),
        "profile": state.get("profile", "standard"),
        "open_gaps": open_gaps,
        "updated_at": datetime.now(UTC).isoformat(),
    }
    with contextlib.suppress(OSError):
        (cwd / "round_state.json").write_text(json.dumps(round_state, indent=2))


# ---------------------------------------------------------------------------
# DAG helpers
# ---------------------------------------------------------------------------


def task_ready(task: dict[str, Any], task_states: dict[str, Any]) -> bool:
    """Return True when all declared dependencies have reached a satisfied state.

    Both 'accepted' and 'skipped_by_policy' count as satisfied: a dep that was
    excluded from the current profile (or skipped after a non-blocking failure)
    should not block downstream tasks.
    """
    for dep in task.get("depends_on", []):
        if task_states.get(dep, {}).get("status") not in READY_STATUSES:
            return False
    return True


def profile_includes(task_id: str, profile: str, manifest: dict[str, Any]) -> bool:
    """Return True when a task is included in the requested profile."""
    profiles = manifest.get("profiles", {})
    if not profiles:
        return True
    task_list = profiles.get(profile, profiles.get("standard", []))
    return task_id in task_list


# ---------------------------------------------------------------------------
# Artifact validation
# ---------------------------------------------------------------------------


def artifact_paths(output: dict[str, Any], ctx: dict[str, str]) -> list[Path]:
    template = output.get("path") or output.get("glob")
    if not template:
        return []
    rendered = template.format(**ctx)
    path = Path(rendered)
    if not path.is_absolute():
        path = Path(ctx.get("cwd", ".")) / path
    if "glob" in output:
        import glob

        return [Path(p) for p in sorted(glob.glob(str(path), recursive=True))]
    return [path]


def _documents(path: Path) -> list[Any]:
    text = path.read_text()
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    return [json.loads(text)]


def validation_errors(
    task: dict[str, Any], ctx: dict[str, str], result: dict[str, Any]
) -> list[str]:
    """Enforce declared gates, including tool success and reviewer rejection."""
    errors: list[str] = []
    output = task.get("output", {})
    checks = task.get("validation", [])
    try:
        paths = artifact_paths(output, ctx)
        for check in checks:
            if check == "exit_code_zero":
                if result.get("success") is not True:
                    errors.append("Missing or unsuccessful execution result")
                if "exit_code" in result and result["exit_code"] != 0:
                    errors.append("Command exit code was not zero")
            elif check in ("artifact_exists", "non_empty"):
                if not paths or any(not p.exists() for p in paths):
                    errors.append("Required output artifact is missing")
                elif check == "non_empty" and any(
                    not p.is_file() or not p.read_bytes().strip() for p in paths
                ):
                    errors.append("Required output artifact is empty")
            elif check == "schema_valid":
                if not paths or not output.get("schema"):
                    errors.append("Schema validation requires an artifact and schema")
                    continue
                import jsonschema

                schema = json.loads((SKILL_DIR / output["schema"]).read_text())
                validator_cls = jsonschema.validators.validator_for(schema)
                validator_cls.check_schema(schema)
                validator = validator_cls(schema)
                for path in paths:
                    documents = _documents(path)
                    if not documents:
                        errors.append(f"No records in {path.name}")
                    for document in documents:
                        errors.extend(
                            f"{path.name}: {err.message}"
                            for err in validator.iter_errors(document)
                        )
            elif check == "evidence_present":
                for path in paths:
                    docs = _documents(path)
                    if len(docs) == 1 and isinstance(docs[0], list):
                        docs = docs[0]
                    for doc in docs:
                        if task["id"] == "paper-analyzer":
                            findings = doc.get("key_findings", [])
                            if not findings or any(
                                not (f.get("section") or f.get("quote"))
                                for f in findings
                            ):
                                errors.append(
                                    "Every paper finding needs a source locator"
                                )
                        elif task["id"] == "paper-synthesizer":
                            corpus = {
                                p.get("paper_id", p.get("arxiv_id", ""))
                                for p in doc.get("corpus", [])
                            }
                            if not corpus:
                                errors.append("Synthesis corpus is empty")
                            shortlist = json.loads(
                                (
                                    Path(ctx["run_dir"]) / "screen/shortlist.json"
                                ).read_text()
                            )
                            allowed = set()
                            for entry in shortlist:
                                paper = entry.get("paper", entry)
                                paper_id = paper.get(
                                    "arxiv_id", paper.get("paper_id", "")
                                )
                                if paper_id:
                                    allowed.add(paper_id)
                                    allowed.add(paper_id + paper.get("version", ""))
                            if not corpus <= allowed:
                                errors.append(
                                    "Synthesis corpus contains papers "
                                    "outside the shortlist"
                                )

                            for field in (
                                "taxonomy",
                                "recurring_patterns",
                                "evidence_strength_map",
                                "operational_implications",
                                "production_readiness",
                                "design_implications",
                                "risk_register",
                            ):
                                for finding in doc.get(field, []):
                                    cited = set(finding.get("supporting_papers", []))
                                    if not cited or not cited <= corpus:
                                        errors.append(
                                            f"{field}: missing or unknown paper IDs"
                                        )
                        elif task["id"] == "paper-screener":
                            paper = doc.get("paper", doc)
                            if not paper.get("abstract", "").strip():
                                errors.append("Screened paper has no abstract evidence")
                            judgment = doc.get("llm") or {}
                            if doc.get("download", True) and (
                                judgment.get("llm_score", 0) < 0.6
                                or not judgment.get("rationale")
                                or not judgment.get("evidence_quotes")
                            ):
                                errors.append(
                                    "LLM shortlist lacks a supported relevance judgment"
                                )

                        else:
                            errors.append("Unknown evidence contract")
            elif check == "per_paper_complete":
                run_dir = Path(ctx["run_dir"])
                expected = {
                    p.stem
                    for folder in (
                        "convert/markdown",
                        "convert",
                        "convert_rough",
                        "convert_fine",
                    )
                    for p in (run_dir / folder).glob("*.md")
                }
                actual = {p.name.removesuffix(".analysis.json") for p in paths}
                if not expected or expected - actual:
                    errors.append(
                        f"Missing paper analyses: {sorted(expected - actual)}"
                    )
            elif check == "validation_passed":
                for path in paths:
                    doc = json.loads(path.read_text())
                    if doc.get("passed") is not True:
                        errors.append("Report validation did not pass")
                    if doc.get("workflow_format_passed") is not True:
                        errors.append("Required report format did not pass")
                    target = Path(ctx["draft_report"])
                    if (
                        doc.get("report_sha256")
                        != hashlib.sha256(target.read_bytes()).hexdigest()
                    ):
                        errors.append("Report validation hash is stale")
            else:
                errors.append(f"Unknown validator: {check}")
        if task.get("executor", {}).get("kind") == "llm_reviewer":
            if not paths:
                errors.append("Reviewer verdict is missing")
            for path in paths:
                verdict = json.loads(path.read_text())
                if verdict.get("reviewer_task_id") != task.get("id"):
                    errors.append("Reviewer task ID does not match")
                if verdict.get("status") not in ("accepted", "accepted_with_issues"):
                    errors.append("Reviewer rejected this artifact")
                if verdict.get("required_fixes") or verdict.get("unsupported_claims"):
                    errors.append("Reviewer still requires fixes")
                target = Path(ctx.get("draft_report", ""))
                if verdict.get("target_artifact") != str(target):
                    errors.append("Reviewer did not review the rendered draft")
                if (
                    target.is_file()
                    and verdict.get("target_sha256")
                    != hashlib.sha256(target.read_bytes()).hexdigest()
                ):
                    errors.append("Reviewer target hash is stale")
                scores = verdict.get("scores", {})
                for key, threshold in (
                    ("faithfulness", 0.75),
                    ("coherence", 0.70),
                    ("gap_completeness", 0.60),
                ):
                    value = scores.get(key)
                    if not isinstance(value, int | float) or value < threshold:
                        errors.append(f"Reviewer {key} is below the contract threshold")
                if scores.get("citation_integrity") is not True:
                    errors.append("Reviewer citation integrity did not pass")
    except (OSError, ValueError, KeyError, TypeError, ImportError) as exc:
        errors.append(f"Cannot validate artifacts: {exc}")
    return errors


def validate_artifact(output: dict[str, Any], ctx: dict[str, str]) -> bool:
    checks = ["artifact_exists"] + (["schema_valid"] if output.get("schema") else [])
    return not validation_errors({"output": output, "validation": checks}, ctx, {})


def output_fingerprints(task: dict[str, Any], ctx: dict[str, str]) -> dict[str, str]:
    """Bind report/review acceptance to the actual files consumed and produced."""
    if not task.get("bind_artifacts"):
        return {}
    paths = artifact_paths(task.get("output", {}), ctx)
    paths += [Path(p.format(**ctx)) for p in task.get("inputs", [])]
    return {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.is_file()
    }


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


def run_deterministic(task: dict[str, Any], ctx: dict[str, str]) -> tuple[bool, str]:
    """Execute a deterministic_script task. Returns (success, message)."""
    executor = task.get("executor", {})
    cmd = executor.get("command", "")
    if not cmd:
        return True, "no command — MCP tool invocation handled by agent"
    try:
        result = subprocess.run(
            command_args(cmd, ctx),
            cwd=ctx.get("cwd", "."),
            capture_output=True,
            text=True,
            timeout=300,
        )
    except (subprocess.TimeoutExpired, OSError):
        return False, f"command timed out after 300s: {cmd[:120]}"
    if result.returncode != 0:
        return False, (result.stderr or result.stdout).strip()
    return True, result.stdout.strip()


def print_llm_delegation(task: dict[str, Any], ctx: dict[str, Any]) -> None:
    """Print the sub-agent contract for the delegated LLM task."""
    # Prefer the manifest-declared contract path; fall back to derived name.
    manifest_contract = task.get("executor", {}).get("contract", "")
    if manifest_contract:
        contract_path = SKILL_DIR / manifest_contract
    else:
        contract_name = task["id"].replace("-", "_") + ".yaml"
        contract_path = CONTRACTS_DIR / contract_name
    print(f"\n{'=' * 60}")
    print(f"DELEGATE TO SUB-AGENT: {task['id']}")
    print(f"  label   : {task.get('label', '')}")
    print(f"  kind    : {task['executor']['kind']}")
    print(f"  name    : {task['executor'].get('name', task['id'])}")
    if contract_path.exists():
        print(f"  contract: {contract_path}")
        contract_text = contract_path.read_text()
        # Substitute all known context variables so the sub-agent sees
        # concrete paths rather than literal template placeholders.
        for key, val in ctx.items():
            contract_text = contract_text.replace(f"{{{key}}}", str(val))
        print(f"\n{contract_text}")
    print("=" * 60)
    print(
        f"Submit the result with --complete-task {task['id']} --result-file RESULT.json"
    )
    print("The runner validates the output. Do not edit accepted status by hand.\n")


# ---------------------------------------------------------------------------
# Main orchestrator loop
# ---------------------------------------------------------------------------


def capture_run_id(
    state: dict[str, Any], ctx: dict[str, str], cwd_str: str
) -> str | None:
    """Discover and record the run id produced by the plan stage (#17).

    The plan MCP tool generates a fresh ``runs/<id>/`` directory but never
    reports the id back into workflow state, so downstream ``{run_id}``
    substitutions render empty. When the state carries no run id, adopt the
    newest ``runs/*`` directory and write it into both the persisted state and
    the live context so subsequent tasks resolve correctly.
    """
    if state.get("run_id"):
        return str(state["run_id"])
    runs_dir = Path(cwd_str) / "runs"
    if not runs_dir.is_dir():
        return None
    candidates = [
        d for d in runs_dir.iterdir() if d.is_dir() and not d.name.startswith(".")
    ]
    if not candidates:
        return None
    newest = max(candidates, key=lambda d: d.stat().st_mtime)
    run_id = newest.name
    state["run_id"] = run_id
    ctx["run_id"] = run_id
    ctx["run_dir"] = str(newest)
    return run_id


def workflow_context(state: dict[str, Any]) -> dict[str, str]:
    context = state.get("context", {})
    cwd = str(context.get("cwd", Path.cwd()))
    workspace = context.get("workspace") or os.environ.get(
        "RESEARCH_PIPELINE_WORKSPACE"
    )
    if not workspace:
        config_path = context.get("config")
        config = {}
        if config_path:
            import tomllib

            config = tomllib.loads(Path(config_path).read_text())
        workspace = config.get("workspace", "./runs")
    workspace_path = Path(workspace).expanduser()
    if not workspace_path.is_absolute():
        workspace_path = Path(cwd) / workspace_path
    run_id = state.get("run_id", "")
    run_dir = workspace_path / run_id
    return {
        "skill_dir": str(SKILL_DIR),
        "python_executable": sys.executable,
        "cwd": cwd,
        "workspace": str(workspace_path),
        "run_id": run_id,
        "run_dir": str(run_dir),
        "topic": state.get("topic", ""),
        "topic_slug": state.get("topic_slug", ""),
        "config": str(context.get("config", "")),
        "prior_paper_ids": ",".join(context.get("prior_paper_ids", [])),
        "fine_paper_ids": ",".join(context.get("fine_paper_ids", [])),
        "synthesis_path": str(
            run_dir
            / (
                "analysis/synthesis.json"
                if state.get("profile") == "deep"
                else "summarize/synthesis_report.json"
            )
        ),
        "draft_report": str(run_dir / "report/draft.md"),
        "draft_sha256": hashlib.sha256(
            (run_dir / "report/draft.md").read_bytes()
        ).hexdigest()
        if (run_dir / "report/draft.md").is_file()
        else "",
        "validation_path": str(run_dir / "validate/validation_result.json"),
        "final_report": str(
            Path(cwd) / f"{state.get('topic_slug', '')}-research-report.md"
        ),
    }


def command_args(template: str, ctx: dict[str, str]) -> list[str]:
    # Split BEFORE substitution: a quoted topic remains exactly one argument.
    tokens = [token.format(**ctx) for token in shlex.split(template)]
    for flag in ("--config", "--run-id"):
        if flag in tokens:
            i = tokens.index(flag)
            if i + 1 < len(tokens) and not tokens[i + 1]:
                del tokens[i : i + 2]
    return tokens


def task_failed(task: dict[str, Any], current: dict[str, Any], reason: str) -> bool:
    skip = task.get("failure_policy", {}).get("on_failure") == "skip"
    current.update(status="skipped_by_policy" if skip else "failed", reason=reason)
    return not skip


def submit_task(
    manifest: dict[str, Any],
    state: dict[str, Any],
    state_path: Path,
    task_id: str,
    result: dict[str, Any],
) -> int:
    task = next(t for t in manifest["tasks"] if t["id"] == task_id)
    current = state["tasks"].get(task_id, {})
    if current.get("status") != "delegated":
        raise ValueError("Only a delegated task can receive a result")
    if result.get("task_id", task_id) != task_id:
        raise ValueError("Result task ID does not match")
    current["result"] = result
    if task_id == "plan" and result.get("success") is True:
        rid = result.get("artifacts", {}).get("run_id")
        problem = ""
        if not isinstance(rid, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", rid):
            problem = "Plan result must include artifacts.run_id"
        elif state.get("run_id") and state["run_id"] != rid:
            problem = "Plan result run ID does not match this workflow"
        if problem:
            task_failed(task, current, problem)
            state["status"] = "blocked"
            save_state(state, state_path)
            return 1
        state["run_id"] = rid
    ctx = workflow_context(state)
    current["result"] = result
    errors = validation_errors(task, ctx, result)
    if result.get("success") is not True:
        errors.insert(0, "Task execution was not successful")
    if errors:
        blocking = task_failed(task, current, "; ".join(errors))
        state["status"] = "blocked" if blocking else "running"
        save_state(state, state_path)
        return int(blocking)
    current.update(
        status="accepted",
        ended_at=datetime.now(UTC).isoformat(),
        fingerprints=output_fingerprints(task, ctx),
    )
    state["status"] = "running"
    save_state(state, state_path)
    return 0


def execute_delegated(
    manifest: dict[str, Any], state: dict[str, Any], state_path: Path, task_id: str
) -> int:
    task = next(t for t in manifest["tasks"] if t["id"] == task_id)
    if state["tasks"].get(task_id, {}).get("status") != "delegated":
        raise ValueError("Task is not currently delegated")
    if task["executor"]["kind"] != "deterministic_mcp_tool":
        raise ValueError("LLM work requires the printed contract and a result file")
    ctx = workflow_context(state)
    try:
        process = subprocess.run(
            command_args(task["executor"]["cli"], ctx),
            cwd=ctx["cwd"],
            env={**os.environ, "RESEARCH_PIPELINE_WORKSPACE": ctx["workspace"]},
            capture_output=True,
            text=True,
            timeout=task["executor"].get("timeout_seconds", 3600),
        )
        result = {
            "success": process.returncode == 0,
            "exit_code": process.returncode,
            "task_id": task_id,
            "artifacts": {},
        }
        receipt_dir = state_path.parent / "execution"
        receipt_dir.mkdir(parents=True, exist_ok=True)
        log_path = (
            receipt_dir / f"{task_id}-{state['tasks'][task_id].get('attempts', 1)}.log"
        )
        log_path.write_text(process.stdout + process.stderr)
        result["log_path"] = str(log_path)
        if task_id == "plan":
            match = re.search(
                r"^Run ID: ([A-Za-z0-9_-]+)$", process.stdout, re.MULTILINE
            )
            if match:
                result["artifacts"]["run_id"] = match.group(1)
    except (OSError, subprocess.TimeoutExpired) as exc:
        result = {
            "success": False,
            "task_id": task_id,
            "error_type": type(exc).__name__,
        }
    return submit_task(manifest, state, state_path, task_id, result)


def retry_task(manifest: dict[str, Any], state: dict[str, Any], task_id: str) -> None:
    tasks = {t["id"]: t for t in manifest["tasks"]}
    if task_id not in tasks:
        raise ValueError("Unknown task")
    affected = {task_id}
    while True:
        expanded = affected | {
            tid
            for tid, task in tasks.items()
            if set(task.get("depends_on", [])) & affected
        }
        if expanded == affected:
            break
        affected = expanded
    for tid in affected:
        current = state["tasks"].get(tid, {})
        policy = tasks[tid].get("failure_policy", {})
        retries = policy.get(
            "retries", 1 if policy.get("on_failure") == "retry_once_then_block" else 0
        )
        if tid == task_id and current.get("attempts", 0) >= retries + 1:
            raise ValueError(
                "Task retry budget exhausted; investigate before starting a new run"
            )
        if tid == "review-synthesis" and current.get("attempts", 0) >= retries + 1:
            raise ValueError("Reviewer retry budget exhausted")
    for tid in affected:
        current = state["tasks"].get(tid, {})
        history = current.get("history", [])
        if current.get("status") != "pending":
            history = [*history, {k: v for k, v in current.items() if k != "history"}]
        state["tasks"][tid] = {
            "status": "pending",
            "attempts": current.get("attempts", 0),
            "history": history,
        }
    state.update(status="running", completed_at=None)


def run_workflow(
    manifest: dict[str, Any],
    state: dict[str, Any],
    state_path: Path,
    profile: str,
    dry_run: bool,
) -> int:
    task_states = state.setdefault("tasks", {})
    ctx = workflow_context(state)
    included = {
        t["id"]
        for t in manifest["tasks"]
        if profile_includes(t["id"], profile, manifest)
    }
    if set(manifest.get("mandatory_gates", [])) - included:
        state["status"] = "blocked"
        save_state(state, state_path)
        return 1
    for task in manifest["tasks"]:
        tid = task["id"]
        current = task_states.setdefault(tid, {"status": "pending"})
        if tid not in included:
            current.update(
                status="skipped_by_policy", reason=f"not in profile '{profile}'"
            )
            continue
        if current["status"] in ("failed", "blocked"):
            state["status"] = "blocked"
            save_state(state, state_path)
            print(f"BLOCKED: {tid}: {current.get('reason', '')}", file=sys.stderr)
            return 1
        if current["status"] == "accepted":
            errors = validation_errors(task, ctx, current.get("result", {}))
            if not current.get("result"):
                errors.append("Legacy accepted task has no execution receipt")
            if current.get("fingerprints", {}) != output_fingerprints(task, ctx):
                errors.append("Accepted report inputs or outputs changed")
            if errors:
                current.update(status="blocked", reason="; ".join(errors))
                state["status"] = "blocked"
                save_state(state, state_path)
                return 1
    changed = True
    while changed:
        changed = False
        for task in manifest["tasks"]:
            tid = task["id"]
            current = task_states[tid]
            if tid not in included or current["status"] in READY_STATUSES:
                continue
            if not task_ready(task, task_states):
                continue
            if current["status"] == "delegated":
                state["status"] = "paused"
                save_state(state, state_path)
                print(
                    f"Awaiting result for {tid}; use --execute-task "
                    "or --complete-task/--result-file."
                )
                return 0
            if tid in ("convert-fine", "expand") and not ctx.get(
                "fine_paper_ids" if tid == "convert-fine" else "prior_paper_ids"
            ):
                current.update(
                    status="skipped_by_policy", reason="No paper IDs selected"
                )
                changed = True
                continue
            kind = task.get("executor", {}).get("kind", "deterministic_script")
            if dry_run:
                print(f"READY [{kind}] {tid}")
                continue
            current.update(
                status="delegated" if kind != "deterministic_script" else "running",
                started_at=datetime.now(UTC).isoformat(),
                attempts=current.get("attempts", 0) + 1,
            )
            save_state(state, state_path)
            if kind in LLM_KINDS:
                print_llm_delegation(task, ctx)
                return 0
            if kind == "deterministic_mcp_tool":
                print(f"[MCP TOOL] {tid}: {task.get('label', tid)}")
                print(
                    "  CLI: "
                    + shlex.join(command_args(task["executor"].get("cli", ""), ctx))
                )
                print(f"  Record actual CLI execution: --execute-task {tid}")
                return 0
            success, message = run_deterministic(task, ctx)
            if tid == "resume-check" and success:
                resume = json.loads(
                    (Path(ctx["cwd"]) / "resume_context.json").read_text()
                )
                context = state.setdefault("context", {})
                context["prior_paper_ids"] = sorted(
                    set(
                        context.get("prior_paper_ids", [])
                        + resume.get("prior_paper_ids", [])
                    )
                )
                context["prior_gaps"] = context.get("prior_gaps", []) or resume.get(
                    "open_gaps_raw", []
                )
                ctx = workflow_context(state)
            result = {"success": success, "exit_code": 0 if success else 1}
            errors = validation_errors(task, ctx, result)
            current["result"] = result
            if not success or errors:
                blocking = task_failed(task, current, message or "; ".join(errors))
                state["status"] = "blocked" if blocking else "running"
                save_state(state, state_path)
                if blocking:
                    return 1
            else:
                current.update(
                    status="accepted",
                    ended_at=datetime.now(UTC).isoformat(),
                    fingerprints=output_fingerprints(task, ctx),
                )
            changed = True
            save_state(state, state_path)
    complete = all(task_states[tid]["status"] in READY_STATUSES for tid in included)
    state["status"] = "complete" if complete else "paused"
    if complete:
        state["completed_at"] = datetime.now(UTC).isoformat()
        state["final_report_path"] = ctx["final_report"]
        print(f"Workflow COMPLETE. Final report: {ctx['final_report']}")
    save_state(state, state_path)
    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def slug(topic: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", topic.lower()).strip("-")


def main() -> int:
    parser = argparse.ArgumentParser(description="research-pipeline skill orchestrator")
    parser.add_argument("topic", nargs="?", default="", help="Research topic")
    parser.add_argument("--run-id", default="", help="Existing run ID (resume)")
    parser.add_argument(
        "--profile",
        default="standard",
        choices=["quick", "standard", "deep", "auto"],
        help="Pipeline profile",
    )
    parser.add_argument("--config", default="", help="Path to config.toml")
    parser.add_argument(
        "--state", default="workflow_state.json", help="Workflow state file path"
    )
    parser.add_argument(
        "--status", action="store_true", help="Print current workflow state and exit"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print ready tasks without executing"
    )
    parser.add_argument(
        "--execute-task",
        default="",
        help="Execute a delegated CLI task and capture its actual exit code",
    )
    parser.add_argument(
        "--complete-task", default="", help="Validate a delegated worker/MCP result"
    )
    parser.add_argument(
        "--result-file",
        type=Path,
        help="Saved result JSON with success and artifacts.run_id for planning",
    )
    parser.add_argument(
        "--retry-task",
        default="",
        help="Invalidate this task and dependents within the retry budget",
    )
    args = parser.parse_args()

    manifest = load_manifest()
    state_path = Path(args.state)
    state = load_state(state_path)

    if args.status:
        if not state:
            print("No workflow state found.")
            return 1
        print(f"Workflow : {state.get('workflow_id')}  run={state.get('run_id')}")
        print(f"Status   : {state.get('status')}")
        print(f"Topic    : {state.get('topic')}")
        print()
        for task_id, ts in state.get("tasks", {}).items():
            status = ts.get("status", "pending")
            reason = (
                f"  [{ts.get('reason', ts.get('message', ''))}]"
                if status not in ("pending", "accepted")
                else ""
            )
            print(f"  {task_id:40s} {status}{reason}")
        return 0

    if not state:
        if not args.topic:
            print("ERROR: provide a topic for a new run.", file=sys.stderr)
            return 1
        topic_slug = slug(args.topic)
        state = {
            "workflow_id": manifest["workflow_id"],
            "run_id": args.run_id,
            "topic": args.topic,
            "topic_slug": topic_slug,
            "profile": args.profile,
            "status": "running",
            "round": 1,
            "max_rounds": 4,
            "started_at": datetime.now(UTC).isoformat(),
            "completed_at": None,
            "tasks": {t["id"]: {"status": "pending"} for t in manifest["tasks"]},
            "context": {
                "config": args.config,
                "cwd": str(Path.cwd()),
                "skill_dir": str(SKILL_DIR),
                "prior_paper_ids": [],
                "prior_gaps": [],
                "fine_paper_ids": [],
            },
            "rounds": [],
            "final_report_path": None,
        }
        save_state(state, state_path)
        print(f"New workflow started. Topic: {args.topic!r}  Profile: {args.profile}")
        print(f"State file: {state_path}")
    else:
        if args.topic:
            state["topic"] = args.topic
            state["topic_slug"] = slug(args.topic)
        if args.run_id:
            state["run_id"] = args.run_id
        if args.config:
            state.setdefault("context", {})["config"] = args.config

    try:
        if args.retry_task:
            retry_task(manifest, state, args.retry_task)
            save_state(state, state_path)
        if args.execute_task:
            code = execute_delegated(manifest, state, state_path, args.execute_task)
            if code:
                return code
        if args.complete_task:
            if not args.result_file:
                raise ValueError("--complete-task requires --result-file")
            code = submit_task(
                manifest,
                state,
                state_path,
                args.complete_task,
                json.loads(args.result_file.read_text()),
            )
            if code:
                return code
    except (ValueError, KeyError, OSError, StopIteration) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if args.dry_run:
        print(f"\nDRY RUN — profile: {state.get('profile', 'standard')}")
        print("Tasks that would execute (in dependency order):\n")

    return run_workflow(
        manifest, state, state_path, state.get("profile", "standard"), args.dry_run
    )


if __name__ == "__main__":
    sys.exit(main())
