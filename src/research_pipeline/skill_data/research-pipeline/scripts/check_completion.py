#!/usr/bin/env python3
"""check_completion.py — Validate that a research-pipeline run is complete.

Checks required artifacts exist and validation passed before the skill
declares success to the user. This is a deterministic guard — it does
not invoke any LLM.

Usage:
    python3 check_completion.py --run-id RUN_ID --slug TOPIC_SLUG
    python3 check_completion.py --run-id RUN_ID --slug TOPIC_SLUG \\
        --workspace ./runs --cwd .

Exit codes:
    0 — All required artifacts present; run is complete.
    1 — Missing or invalid artifacts; do not declare complete.

Outputs:
    JSON result to stdout.
    Human-readable status to stderr.
"""

import argparse
import json
import sys
from pathlib import Path

# Artifacts that MUST exist for a run to be considered complete.
# (relative to runs/<run_id>/, label, blocking)
REQUIRED_ARTIFACTS: list[tuple[str, str, bool]] = [
    ("plan/query_plan.json", "plan", True),
    ("search/candidates.jsonl", "search", True),
    ("screen/screened.jsonl", "screen", True),
]

# At least one of these must exist (summarize stage).
SUMMARIZE_CANDIDATES = [
    "summarize/synthesis_report.json",
    "summarize/synthesis.json",
]

# Validation file locations to probe (in order of preference).
# `research-pipeline validate` writes ``validation_result.json`` next to the
# report; the plain ``validation.json`` names are kept for older runs (#32).
VALIDATION_CANDIDATES = [
    "validate/validation_result.json",
    "validation_result.json",
    "summarize/validation_result.json",
    "validate/validation.json",
    "validation.json",
    "summarize/validation.json",
]


def _find_first(run_dir: Path, paths: list[str]) -> "Path | None":
    for rel in paths:
        p = run_dir / rel
        if p.exists() and p.stat().st_size > 0:
            return p
    return None


def check(run_id: str, slug: str, workspace: str, cwd: str) -> dict:
    """Run all completion checks and return a structured result dict."""
    runs_dir = Path(workspace).expanduser() if workspace else Path("runs")

    run_dir = runs_dir / run_id
    result: dict = {
        "run_id": run_id,
        "slug": slug,
        "checks": {},
        "blocking": [],
        "warnings": [],
        "all_passed": False,
    }

    # ── 1. Run directory ───────────────────────────────────────────────────
    if not run_dir.exists():
        result["checks"]["run_dir"] = {
            "status": "FAIL",
            "message": f"Run directory not found: {run_dir}",
        }
        result["blocking"].append("run_dir")
        return result
    result["checks"]["run_dir"] = {"status": "PASS", "path": str(run_dir)}

    # ── 2. Core pipeline artifacts ─────────────────────────────────────────
    for rel_path, stage, blocking in REQUIRED_ARTIFACTS:
        artifact = run_dir / rel_path
        if artifact.exists() and artifact.stat().st_size > 0:
            result["checks"][stage] = {"status": "PASS", "path": str(artifact)}
        else:
            result["checks"][stage] = {
                "status": "FAIL",
                "path": str(artifact),
                "message": "Missing or empty",
            }
            if blocking:
                result["blocking"].append(stage)

    # ── 3. Summarize artifacts ─────────────────────────────────────────────
    summarize_path = _find_first(run_dir, SUMMARIZE_CANDIDATES)
    if summarize_path:
        result["checks"]["summarize"] = {"status": "PASS", "path": str(summarize_path)}
    else:
        result["checks"]["summarize"] = {
            "status": "FAIL",
            "message": (
                "Neither summarize/synthesis_report.json nor summarize/synthesis.json "
                "found. Run 'research-pipeline summarize' first."
            ),
        }
        result["blocking"].append("summarize")

    # ── 4. Final report in CWD ─────────────────────────────────────────────
    report_path = Path(cwd).expanduser() / f"{slug}-research-report.md"
    if report_path.exists() and report_path.stat().st_size > 100:
        result["checks"]["final_report"] = {"status": "PASS", "path": str(report_path)}
    else:
        result["checks"]["final_report"] = {
            "status": "FAIL",
            "path": str(report_path),
            "message": (
                f"Final report not written to CWD or is empty. Expected: {report_path}"
            ),
        }
        result["blocking"].append("final_report")

    # ── 5. Validation result ───────────────────────────────────────────────
    val_path = _find_first(run_dir, VALIDATION_CANDIDATES)
    if val_path:
        try:
            with open(val_path) as f:
                val_data = json.load(f)
            # Accept various schemas: passed, valid, status == "pass"
            passed = val_data.get(
                "passed",
                val_data.get("valid", val_data.get("status") == "pass"),
            )
            if passed:
                result["checks"]["validation"] = {
                    "status": "PASS",
                    "path": str(val_path),
                }
            else:
                reasons = val_data.get("reasons", val_data.get("errors", []))
                result["checks"]["validation"] = {
                    "status": "FAIL",
                    "path": str(val_path),
                    "message": f"Validation failed: {reasons}",
                }
                result["blocking"].append("validation")
        except (json.JSONDecodeError, OSError) as exc:
            result["checks"]["validation"] = {
                "status": "FAIL",
                "message": f"Cannot read validation file: {exc}",
            }
            result["blocking"].append("validation")
    else:
        result["checks"]["validation"] = {
            "status": "WARN",
            "message": (
                "validation.json not found. "
                "Run 'research-pipeline validate --report <path>'"
                " before declaring complete."
            ),
        }
        result["warnings"].append("validation_missing")

    result["all_passed"] = len(result["blocking"]) == 0
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate research-pipeline run completion before declaring done."
    )
    parser.add_argument("--run-id", required=True, help="Pipeline run ID")
    parser.add_argument(
        "--slug",
        required=True,
        help="Topic slug for the final report filename (e.g. transformer-time-series)",
    )
    parser.add_argument(
        "--workspace",
        default="",
        help="Path to the runs directory (default: ./runs)",
    )
    parser.add_argument(
        "--cwd",
        default=".",
        help="Working directory where the final report is expected (default: .)",
    )
    args = parser.parse_args()

    result = check(args.run_id, args.slug, args.workspace, args.cwd)
    print(json.dumps(result, indent=2))

    if result["all_passed"]:
        print(
            f"\n✅ Run {args.run_id} complete — all required artifacts present.",
            file=sys.stderr,
        )
        if result["warnings"]:
            print(f"   Warnings: {result['warnings']}", file=sys.stderr)
        return 0
    else:
        print(
            f"\n❌ Run {args.run_id} is NOT complete. Blocking: {result['blocking']}",
            file=sys.stderr,
        )
        print(
            "   Do not declare this run complete until all blocking checks pass.",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
