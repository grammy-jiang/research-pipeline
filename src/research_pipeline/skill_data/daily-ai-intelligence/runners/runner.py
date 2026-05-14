#!/usr/bin/env python3
"""
daily-ai-intelligence skill orchestrator.

Reads manifest.json, maintains workflow_state.json, and executes the
briefing pipeline in dependency order. The orchestrator is the single
authority for task status transitions.

Usage:
  python runner.py --registry <REG> --workspace <WS> [--date YYYY-MM-DD]
  python runner.py --status                           # show current state
  python runner.py --dry-run --registry <REG> ...     # print task graph
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SKILL_DIR = Path(__file__).parent.parent
MANIFEST_PATH = SKILL_DIR / "manifest.json"
CONTRACTS_DIR = Path(__file__).parent / "subagent_contracts"

TERMINAL_STATUSES = {"accepted", "skipped_by_policy", "blocked", "failed"}
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


# ---------------------------------------------------------------------------
# DAG helpers
# ---------------------------------------------------------------------------


def task_ready(task: dict[str, Any], task_states: dict[str, Any]) -> bool:
    for dep in task.get("depends_on", []):
        if task_states.get(dep, {}).get("status") not in READY_STATUSES:
            return False
    return True


def validate_artifact(output: dict[str, Any], ctx: dict[str, str]) -> bool:
    path_template = output.get("path", "")
    if not path_template:
        return True
    path = Path(path_template.format(**ctx))
    return path.exists()


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


def run_deterministic(task: dict[str, Any], ctx: dict[str, str]) -> tuple[bool, str]:
    executor = task.get("executor", {})
    cmd = executor.get("command", "")
    if not cmd:
        return True, "no command — MCP tool invocation handled by agent"
    for key, val in ctx.items():
        cmd = cmd.replace(f"{{{key}}}", str(val))
    result = subprocess.run(shlex.split(cmd), capture_output=True, text=True)
    if result.returncode != 0:
        return False, (result.stderr or result.stdout).strip()
    return True, result.stdout.strip()


def print_llm_delegation(task: dict[str, Any]) -> None:
    contract_name = task["id"].replace("-", "_") + ".yaml"
    contract_path = CONTRACTS_DIR / contract_name
    print(f"\n{'=' * 60}")
    print(f"DELEGATE TO REVIEWER: {task['id']}")
    print(f"  label: {task.get('label', '')}")
    if contract_path.exists():
        print(f"  contract: {contract_path}")
        print(f"\n{contract_path.read_text()}")
    print("=" * 60)
    print("After the reviewer completes, update workflow_state.json:")
    print(f"  tasks.{task['id']}.status = 'accepted'")
    print("Then re-run runner.py to continue.\n")


# ---------------------------------------------------------------------------
# Main orchestrator loop
# ---------------------------------------------------------------------------


def run_workflow(
    manifest: dict[str, Any],
    state: dict[str, Any],
    state_path: Path,
    dry_run: bool,
) -> int:
    task_states = state.setdefault("tasks", {})
    ctx = {
        "skill_dir": str(SKILL_DIR),
        "registry": state.get("context", {}).get("registry", ""),
        "workspace": state.get("context", {}).get("workspace", ""),
        "date": state.get("date", ""),
        "cluster_id": state.get("context", {}).get("cluster_id", ""),
        "vault": state.get("context", {}).get("vault", ""),
        "week": state.get("context", {}).get("week", ""),
    }

    changed = True
    while changed:
        changed = False
        for task in manifest["tasks"]:
            task_id = task["id"]
            current = task_states.get(task_id, {})

            if current.get("status") in TERMINAL_STATUSES:
                continue
            if not task_ready(task, task_states):
                continue

            # Skip optional tasks that lack a required context value.
            if task.get("optional"):
                trigger = task.get("trigger_condition", "")
                if trigger and not _optional_trigger_met(task_id, ctx):
                    task_states[task_id] = {
                        "status": "skipped_by_policy",
                        "reason": "optional, not triggered",
                    }
                    save_state(state, state_path)
                    changed = True
                    continue

            kind = task.get("executor", {}).get("kind", "deterministic_script")
            label = task.get("label", task_id)

            if dry_run:
                opt = " [optional]" if task.get("optional") else ""
                print(f"  READY  [{kind:30s}]  {task_id}{opt}")
                continue

            # ---- LLM reviewer — delegate ----
            if kind in LLM_KINDS:
                if current.get("status") == "delegated":
                    continue
                print(f"\n[DELEGATING] {task_id}: {label}")
                print_llm_delegation(task)
                task_states[task_id] = {
                    "status": "delegated",
                    "started_at": datetime.now(UTC).isoformat(),
                }
                save_state(state, state_path)
                changed = True
                return 0

            # ---- MCP tool — print invocation for agent ----
            if kind == "deterministic_mcp_tool":
                cli = task.get("executor", {}).get("cli", "")
                for key, val in ctx.items():
                    cli = cli.replace(f"{{{key}}}", str(val))
                print(f"\n[MCP TOOL] {task_id}: {label}")
                if cli:
                    print(f"  CLI equivalent: {cli}")
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
                fm = task.get("failure_policy", {}).get("message", "")
                if fm:
                    print(f"  {fm}", file=sys.stderr)
                if policy == "block":
                    return 1
                task_states[task_id]["status"] = "skipped_by_policy"
                save_state(state, state_path)
                changed = True
                continue

            output = task.get("output", {})
            if output and not validate_artifact(output, ctx):
                policy = task.get("failure_policy", {}).get("on_failure", "block")
                reason = f"artifact not found: {output.get('path', '?')}"
                task_states[task_id].update({"status": "failed", "reason": reason})
                save_state(state, state_path)
                print(f"  BLOCKED — {reason}", file=sys.stderr)
                if policy == "block":
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

    # ---- Completion check ----
    delegated = {
        tid for tid, ts in task_states.items() if ts.get("status") == "delegated"
    }
    if delegated:
        print(f"\nWorkflow paused. Delegated: {sorted(delegated)}")
        return 0

    remaining = {
        t["id"]
        for t in manifest["tasks"]
        if task_states.get(t["id"], {}).get("status") not in TERMINAL_STATUSES
        and not t.get("optional")
    }
    if not remaining:
        state["status"] = "complete"
        state["completed_at"] = datetime.now(UTC).isoformat()
        brief_path = Path(ctx["workspace"]) / ctx["date"] / "reports" / "daily.md"
        state["final_brief_path"] = str(brief_path)
        save_state(state, state_path)
        print("\nWorkflow COMPLETE.")
        print(f"Brief: {brief_path}")
        print(
            "Validation: "
            + str(
                Path(ctx["workspace"]) / ctx["date"] / "validation" / "validation.json"
            )
        )
    else:
        print(f"\nWorkflow paused. Remaining non-optional: {sorted(remaining)}")

    return 0


def _optional_trigger_met(task_id: str, ctx: dict[str, str]) -> bool:
    """Check whether an optional task's required context is available."""
    requires: dict[str, list[str]] = {
        "dossier": ["cluster_id"],
        "export-obsidian": ["vault"],
        "weekly-synthesis": ["week"],
        "feedback": [],
        "preferences": [],
    }
    needed = requires.get(task_id, [])
    return all(ctx.get(k) for k in needed)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="daily-ai-intelligence skill orchestrator"
    )
    parser.add_argument("--registry", default="", help="Path to registry config TOML")
    parser.add_argument(
        "--workspace", default="./workspace/briefing", help="Workspace root"
    )
    parser.add_argument(
        "--date", default="", help="Date YYYY-MM-DD (default: today UTC)"
    )
    parser.add_argument("--cluster-id", default="", help="Cluster ID for dossier task")
    parser.add_argument("--vault", default="", help="Obsidian vault path")
    parser.add_argument("--week", default="", help="ISO week for weekly synthesis")
    parser.add_argument(
        "--state",
        default="",
        help="Workflow state file (default: <workspace>/<date>/workflow_state.json)",
    )
    parser.add_argument("--status", action="store_true", help="Print workflow state")
    parser.add_argument(
        "--dry-run", action="store_true", help="Print tasks without executing"
    )
    args = parser.parse_args()

    manifest = load_manifest()

    date = args.date or datetime.now(UTC).strftime("%Y-%m-%d")
    workspace = args.workspace or "./workspace/briefing"
    state_file = args.state or str(Path(workspace) / date / "workflow_state.json")
    state_path = Path(state_file)
    state_path.parent.mkdir(parents=True, exist_ok=True)

    state = load_state(state_path)

    if args.status:
        if not state:
            print("No workflow state found.")
            return 1
        print(f"Workflow : {state.get('workflow_id')}  date={state.get('date')}")
        print(f"Status   : {state.get('status')}")
        for task_id, ts in state.get("tasks", {}).items():
            status = ts.get("status", "pending")
            reason = f"  [{ts.get('reason', '')}]" if ts.get("reason") else ""
            print(f"  {task_id:40s} {status}{reason}")
        return 0

    if not state:
        if not args.registry:
            print("ERROR: --registry is required for a new run.", file=sys.stderr)
            return 1
        state = {
            "workflow_id": manifest["workflow_id"],
            "date": date,
            "status": "running",
            "started_at": datetime.now(UTC).isoformat(),
            "completed_at": None,
            "tasks": {t["id"]: {"status": "pending"} for t in manifest["tasks"]},
            "context": {
                "skill_dir": str(SKILL_DIR),
                "registry": args.registry,
                "workspace": workspace,
                "cluster_id": args.cluster_id,
                "vault": args.vault,
                "week": args.week,
            },
            "final_brief_path": None,
            "validation_path": None,
        }
        save_state(state, state_path)
        print(f"New workflow started. Date: {date}")
        print(f"State file: {state_path}")
    else:
        ctx = state.setdefault("context", {})
        if args.registry:
            ctx["registry"] = args.registry
        if args.workspace:
            ctx["workspace"] = workspace
        if args.cluster_id:
            ctx["cluster_id"] = args.cluster_id
        if args.vault:
            ctx["vault"] = args.vault
        if args.week:
            ctx["week"] = args.week

    if args.dry_run:
        print(f"\nDRY RUN — date: {date}")
        print("Tasks that would execute:\n")

    return run_workflow(manifest, state, state_path, args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
