#!/usr/bin/env bash
# run-copilot-loop.sh
# Runs GitHub Copilot CLI in non-interactive mode 5 times with a fixed prompt.
#
# Key flags used:
#   -p / --prompt   non-interactive mode; process exits after task completes
#   --allow-all     grant all tool/path/URL permissions (same as --yolo)
#   --autopilot     autonomous mode; agent works without per-step confirmations

set -euo pipefail

ITERATIONS=5
PROMPT='please review the skills provided by this repository, understand it, explain to me the workflow/steps inside; then evaludate it if there is any bug, or conflict between steps, mark them up; then please check the local CHANGELOG.md to see if you have fixed them before, when it was, what you fixed, and why it is still failed, understand the change history, then fix it, update the CHANGELOG.md, commit'

LOG_DIR="copilot-loop-logs"
mkdir -p "$LOG_DIR"

# Verify copilot is on PATH
if ! command -v copilot &>/dev/null; then
    echo "ERROR: 'copilot' not found on PATH. Install via: curl -fsSL https://gh.io/copilot-install | bash" >&2
    exit 1
fi

run_iteration() {
    local iter=$1
    local timestamp
    timestamp=$(date '+%Y%m%d-%H%M%S')
    local log_file="$LOG_DIR/iteration-$(printf '%02d' "$iter")-${timestamp}.log"

    echo ""
    echo "════════════════════════════════════════════════════════"
    printf "  Iteration %d / %d   started %s\n" "$iter" "$ITERATIONS" "$(date '+%Y-%m-%d %H:%M:%S')"
    echo "════════════════════════════════════════════════════════"

    copilot \
        --prompt "$PROMPT" \
        --allow-all \
        --autopilot \
        2>&1 | tee "$log_file"

    local status=${PIPESTATUS[0]}
    echo ""
    if [[ $status -eq 0 ]]; then
        echo "✓  Iteration $iter complete — log: $log_file"
    else
        echo "⚠  Iteration $iter exited with status $status — log: $log_file"
    fi
    return $status
}

failed=0
for i in $(seq 1 $ITERATIONS); do
    if ! run_iteration "$i"; then
        (( failed++ )) || true
    fi
done

echo ""
echo "════════════════════════════════════════════════════════"
echo "  All $ITERATIONS iterations finished."
echo "  Failures: $failed"
echo "  Logs in:  $LOG_DIR/"
echo "════════════════════════════════════════════════════════"

exit "$failed"
