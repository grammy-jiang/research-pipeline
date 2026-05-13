#!/usr/bin/env bash
# check_completion.sh — Verify that a daily-ai-intelligence brief run is
# complete and validated before the skill surfaces results to the user.
#
# Usage: check_completion.sh --workspace WS --date DATE
# Exit 0: brief is complete and passed validation.
# Exit 1: brief is incomplete, not yet validated, or validation failed.

set -euo pipefail

WS=""
DATE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --workspace) WS="${2:-}"; shift 2 ;;
        --date)      DATE="${2:-}"; shift 2 ;;
        *)
            echo "Unknown argument: $1" >&2
            echo "Usage: check_completion.sh --workspace WS --date DATE" >&2
            exit 1
            ;;
    esac
done

WS="${WS:-./workspace/briefing}"
DATE="${DATE:-$(date -u +%F)}"

# Expand ~ in path
WS="${WS/#\~/$HOME}"

DAY_DIR="${WS}/${DATE}"
VALIDATION="${DAY_DIR}/validation/validation.json"
REPORT="${DAY_DIR}/reports/daily.md"
FAILED=0

echo "Checking brief completion for date: $DATE" >&2
echo "Workspace: $WS" >&2

# ── 1. Day directory ──────────────────────────────────────────────────────
if [[ ! -d "$DAY_DIR" ]]; then
    echo "FAIL: Day directory not found: $DAY_DIR" >&2
    echo "  Run 'research-pipeline brief run --workspace $WS --date $DATE' first." >&2
    exit 1
fi

# ── 2. Daily report ───────────────────────────────────────────────────────
if [[ ! -f "$REPORT" ]]; then
    echo "FAIL: Daily report not found: $REPORT" >&2
    FAILED=1
elif [[ ! -s "$REPORT" ]]; then
    echo "FAIL: Daily report is empty: $REPORT" >&2
    FAILED=1
else
    echo "PASS: Daily report present: $REPORT" >&2
fi

# ── 3. Validation file ────────────────────────────────────────────────────
if [[ ! -f "$VALIDATION" ]]; then
    echo "FAIL: Validation file not found: $VALIDATION" >&2
    echo "  Run 'research-pipeline brief validate --workspace $WS --date $DATE'" >&2
    FAILED=1
else
    # Parse validation result with Python (handles both JSON schemas)
    if python3 - "$VALIDATION" <<'PYEOF'
import json, sys
path = sys.argv[1]
try:
    with open(path) as f:
        v = json.load(f)
    passed = v.get("passed", v.get("valid", v.get("status") == "pass"))
    if not passed:
        reasons = v.get("reasons", v.get("errors", v.get("failures", [])))
        print(f"FAIL: Validation did not pass. Reasons: {reasons}", file=sys.stderr)
        sys.exit(1)
    print("PASS: Validation passed.", file=sys.stderr)
    sys.exit(0)
except Exception as e:
    print(f"FAIL: Cannot read validation file: {e}", file=sys.stderr)
    sys.exit(1)
PYEOF
    then
        : # pass — already printed by python
    else
        FAILED=1
    fi
fi

if [[ "$FAILED" -eq 1 ]]; then
    echo "" >&2
    echo "❌  Brief for ${DATE} is NOT ready to surface." >&2
    echo "    Fix the issues above before delivering to the user." >&2
    echo "    Low-signal / no-news days are valid outputs — do not pad the brief." >&2
    exit 1
fi

echo "" >&2
echo "✅  Brief for ${DATE} is complete and validated." >&2
echo "    Report  : $REPORT" >&2
echo "    Validated: $VALIDATION" >&2
exit 0
