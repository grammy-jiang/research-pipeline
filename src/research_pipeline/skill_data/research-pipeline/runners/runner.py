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
import json
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
                if g.get("gap_type") != "OUT_OF_SCOPE"
            ]
        except (json.JSONDecodeError, OSError):
            pass

    round_state: dict[str, Any] = {
        "workflow_id": state.get("workflow_id", "research-pipeline"),
        "run_id": state.get("run_id", ""),
        "topic": state.get("topic", ""),
        "topic_slug": state.get("topic_slug", ""),
        "current_round": state.get("round", 1),
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


def validate_artifact(output: dict[str, Any], ctx: dict[str, str]) -> bool:
    """Check that the expected output artifact exists on disk."""
    path_template = output.get("path", "")
    if not path_template:
        return True
    path = Path(path_template.format(**ctx))
    return path.exists()


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


def run_deterministic(task: dict[str, Any], ctx: dict[str, str]) -> tuple[bool, str]:
    """Execute a deterministic_script task. Returns (success, message)."""
    executor = task.get("executor", {})
    cmd = executor.get("command", "")
    if not cmd:
        return True, "no command — MCP tool invocation handled by agent"
    for key, val in ctx.items():
        cmd = cmd.replace(f"{{{key}}}", str(val))
    try:
        result = subprocess.run(
            shlex.split(cmd), capture_output=True, text=True, timeout=300
        )
    except subprocess.TimeoutExpired:
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
    print("After the sub-agent completes and the artifact exists,")
    print(f"update workflow_state.json: tasks.{task['id']}.status = 'accepted'")
    print("Then re-run runner.py to continue the workflow.\n")


# ---------------------------------------------------------------------------
# Main orchestrator loop
# ---------------------------------------------------------------------------


def run_workflow(
    manifest: dict[str, Any],
    state: dict[str, Any],
    state_path: Path,
    profile: str,
    dry_run: bool,
) -> int:
    task_states = state.setdefault("tasks", {})
    cwd_str = state.get("context", {}).get("cwd", ".")
    run_id = state.get("run_id", "")
    ctx = {
        "skill_dir": str(SKILL_DIR),
        "run_id": run_id,
        "topic": state.get("topic", ""),
        "topic_slug": state.get("topic_slug", ""),
        "config": state.get("context", {}).get("config", ""),
        "cwd": cwd_str,
        # Derived: default workspace location for this run (./runs/<run_id>).
        "run_dir": str(Path(cwd_str) / "runs" / run_id) if run_id else "",
        "prior_paper_ids": ",".join(
            state.get("context", {}).get("prior_paper_ids", [])
        ),
        "fine_paper_ids": ",".join(state.get("context", {}).get("fine_paper_ids", [])),
    }

    changed = True
    while changed:
        changed = False
        for task in manifest["tasks"]:
            task_id = task["id"]
            current = task_states.get(task_id, {})

            # Skip tasks already in a terminal state.
            if current.get("status") in TERMINAL_STATUSES:
                continue

            # Skip tasks excluded from the requested profile.
            if not profile_includes(task_id, profile, manifest):
                if current.get("status") != "skipped_by_policy":
                    task_states[task_id] = {
                        "status": "skipped_by_policy",
                        "reason": f"not in profile '{profile}'",
                    }
                    save_state(state, state_path)
                continue

            # Skip if dependencies are not yet accepted.
            if not task_ready(task, task_states):
                continue

            # Guard: skip convert-fine when no paper IDs have been selected.
            # Passing --paper-ids "" to the CLI causes an error; skipping is safe
            # because convert-fine is an enrichment step, not a core stage.
            if task_id == "convert-fine" and not ctx.get("fine_paper_ids"):
                if current.get("status") != "skipped_by_policy":
                    task_states[task_id] = {
                        "status": "skipped_by_policy",
                        "reason": "fine_paper_ids not set in workflow context",
                    }
                    save_state(state, state_path)
                    changed = True
                continue

            kind = task.get("executor", {}).get("kind", "deterministic_script")
            label = task.get("label", task_id)

            if dry_run:
                print(f"  READY  [{kind:30s}]  {task_id}")
                continue

            # ---- LLM worker / reviewer — delegate to sub-agent ----
            if kind in LLM_KINDS:
                if current.get("status") == "delegated":
                    # Already delegated; waiting for agent to mark accepted.
                    continue
                print(f"\n[DELEGATING] {task_id}: {label}")
                print_llm_delegation(task, ctx)
                task_states[task_id] = {
                    "status": "delegated",
                    "started_at": datetime.now(UTC).isoformat(),
                }
                save_state(state, state_path)
                changed = True
                # Pause: agent must complete the sub-agent and re-run runner.
                return 0

            # ---- MCP tool — print the invocation for the agent ----
            if kind == "deterministic_mcp_tool":
                cli = task.get("executor", {}).get("cli", "")
                for key, val in ctx.items():
                    cli = cli.replace(f"{{{key}}}", str(val))
                print(f"\n[MCP TOOL] {task_id}: {label}")
                if cli:
                    print(f"  CLI equivalent: {cli}")
                # Optimistically accept MCP tool tasks; the agent will validate.
                task_states[task_id] = {
                    "status": "accepted",
                    "ended_at": datetime.now(UTC).isoformat(),
                    "note": "MCP tool invoked by agent",
                }
                save_state(state, state_path)
                changed = True
                continue

            # ---- Deterministic script ----
            print(f"\n[RUNNING] {task_id}: {label}")
            task_states[task_id] = {
                "status": "running",
                "started_at": datetime.now(UTC).isoformat(),
            }
            save_state(state, state_path)

            success, msg = run_deterministic(task, ctx)
            if not success:
                policy = task.get("failure_policy", {}).get("on_failure", "block")
                task_states[task_id].update({"status": "failed", "reason": msg})
                save_state(state, state_path)
                print(f"  FAILED: {msg}", file=sys.stderr)
                if policy in ("block", "retry_once_then_block"):
                    return 1
                task_states[task_id]["status"] = "skipped_by_policy"
                save_state(state, state_path)
                changed = True
                continue

            # Validate artifact existence.
            output = task.get("output", {})
            if output and not validate_artifact(output, ctx):
                policy = task.get("failure_policy", {}).get(
                    "on_artifact_missing",
                    task.get("failure_policy", {}).get("on_failure", "block"),
                )
                reason = f"artifact not found: {output.get('path', '?')}"
                task_states[task_id].update({"status": "failed", "reason": reason})
                save_state(state, state_path)
                print(f"  BLOCKED — {reason}", file=sys.stderr)
                if policy in ("block", "retry_once_then_block"):
                    return 1
                task_states[task_id]["status"] = "skipped_by_policy"

            else:
                task_states[task_id].update(
                    {
                        "status": "accepted",
                        "ended_at": datetime.now(UTC).isoformat(),
                        "message": msg[:200] if msg else "",
                    }
                )
                print("  ACCEPTED")

            save_state(state, state_path)
            changed = True

    # ---- Check overall completion ----
    terminal = {
        tid for tid, ts in task_states.items() if ts.get("status") in TERMINAL_STATUSES
    }
    in_scope = {
        t["id"]
        for t in manifest["tasks"]
        if profile_includes(t["id"], profile, manifest)
    }
    delegated = {
        tid for tid, ts in task_states.items() if ts.get("status") == "delegated"
    }

    if delegated:
        print(f"\nWorkflow paused. Delegated tasks: {sorted(delegated)}")
        print(
            "Complete the sub-agent task and update workflow_state.json, then re-run."
        )
        return 0

    remaining = in_scope - terminal
    if not remaining:
        state["status"] = "complete"
        state["completed_at"] = datetime.now(UTC).isoformat()
        save_state(state, state_path)
        print("\nWorkflow COMPLETE. Read workflow_state.json for full status.")
        print(f"Final report: {ctx['cwd']}/{ctx['topic_slug']}-research-report.md")
    else:
        pending_blocked = [
            f"{tid}={task_states.get(tid, {}).get('status', 'pending')}"
            for tid in sorted(remaining)
        ]
        print(f"\nWorkflow paused. Remaining: {pending_blocked}")

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

    if args.dry_run:
        print(f"\nDRY RUN — profile: {state.get('profile', 'standard')}")
        print("Tasks that would execute (in dependency order):\n")

    return run_workflow(
        manifest, state, state_path, state.get("profile", "standard"), args.dry_run
    )


if __name__ == "__main__":
    sys.exit(main())
