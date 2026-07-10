#!/usr/bin/env bash
# run-copilot-loop.sh
# Runs GitHub Copilot CLI in non-interactive mode 5 times with a fixed prompt.
#
# The agent runs autonomously (--allow-all --autopilot) and commits directly on
# the checked-out branch. That autopilot commit is NOT reviewed by a human or by
# CI before the next iteration runs, so this driver enforces a DETERMINISTIC
# gate the agent cannot self-attest around (issue #106):
#
#   * before each iteration: refuse to start on a dirty working tree, so
#     iteration N+1 never builds on iteration N's uncommitted / broken state;
#   * after each iteration: verify the commits the agent just made against the
#     repo's Hard Constraints (HC1/HC2/HC3) and quality gates, and STOP the loop
#     on any failure so no further autopilot iterations pile onto a bad commit.
#
# Key flags used:
#   -p / --prompt   non-interactive mode; process exits after task completes
#   --allow-all     grant all tool/path/URL permissions (same as --yolo)
#   --autopilot     autonomous mode; agent works without per-step confirmations
#
# Env:
#   SKIP_GATE=1            disable the deterministic gate (NOT recommended)
#   ALLOWED_PATHS_REGEX    override the HC2 path allowlist (default below)

set -euo pipefail

ITERATIONS=5
PROMPT='please review the skills provided by this repository, understand it, explain to me the workflow/steps inside; then evaludate it if there is any bug, or conflict between steps, mark them up; then please check the local CHANGELOG.md to see if you have fixed them before, when it was, what you fixed, and why it is still failed, understand the change history, then fix it, update the CHANGELOG.md, commit'

LOG_DIR="copilot-loop-logs"

# Deterministic gate config (issue #106).
SKIP_GATE="${SKIP_GATE:-0}"
# Mirrors AGENTS.md HC2, plus the maintained root CHANGELOG.md (docs/changelog.md
# is a symlink to it). Every changed file in an autopilot commit must match.
ALLOWED_PATHS_REGEX="${ALLOWED_PATHS_REGEX:-^(src/|tests/|docs/|\.github/|pyproject\.toml$|\.pre-commit-config\.yaml$|Makefile$|AGENTS\.md$|CLAUDE\.md$|CHANGELOG\.md$)}"

mkdir -p "$LOG_DIR"

# Verify copilot is on PATH
if ! command -v copilot &>/dev/null; then
    echo "ERROR: 'copilot' not found on PATH. Install via: curl -fsSL https://gh.io/copilot-install | bash" >&2
    exit 1
fi

# The gate needs a git repository.
if ! git rev-parse --is-inside-work-tree &>/dev/null; then
    echo "ERROR: run-copilot-loop.sh must run inside a git repository." >&2
    exit 1
fi

require_clean_tree() {
    if [[ -n "$(git status --porcelain)" ]]; then
        echo "ERROR: working tree is dirty — refusing to start an iteration on an" >&2
        echo "unclean tree (a prior iteration left uncommitted changes)." >&2
        git status --short >&2
        return 1
    fi
}

# Deterministic verification of the commits made since $1 (HEAD before the run).
run_gate() {
    local base=$1

    if [[ "$SKIP_GATE" == "1" ]]; then
        echo "⚠  SKIP_GATE=1 — deterministic gate DISABLED (not recommended)."
        return 0
    fi

    if [[ "$(git rev-parse HEAD)" == "$(git rev-parse "$base")" ]]; then
        echo "ℹ  No new commits this iteration — nothing to gate."
        return 0
    fi

    local changed
    changed=$(git diff --name-only "$base" HEAD)
    echo "── Gate: files changed in ${base:0:8}..HEAD ──"
    echo "$changed"

    # HC2 — path allowlist: no agent-authored write outside the allowlist.
    local offenders=""
    while IFS= read -r f; do
        [[ -z "$f" ]] && continue
        if ! [[ "$f" =~ $ALLOWED_PATHS_REGEX ]]; then
            offenders+="  $f"$'\n'
        fi
    done <<< "$changed"
    if [[ -n "$offenders" ]]; then
        echo "GATE FAIL (HC2): commit writes outside the path allowlist:" >&2
        printf '%s' "$offenders" >&2
        return 1
    fi

    # HC3 — destructive-command scan on added lines.
    if git diff "$base" HEAD -- . | grep -E '^\+' \
        | grep -Eq 'rm[[:space:]]+-rf|git[[:space:]]+push[[:space:]]+--force|git[[:space:]]+reset[[:space:]]+--hard|DROP[[:space:]]+TABLE'; then
        echo "GATE FAIL (HC3): a destructive command appears in the diff." >&2
        return 1
    fi

    # HC1 — detect-secrets over exactly the changed set.
    if ! uv run pre-commit run detect-secrets --from-ref "$base" --to-ref HEAD; then
        echo "GATE FAIL (HC1): detect-secrets flagged the diff." >&2
        return 1
    fi

    # Quality gates — the same checks CI runs, but BEFORE the next iteration.
    uv run ruff check . || { echo "GATE FAIL: ruff check" >&2; return 1; }
    uv run ruff format --check . || { echo "GATE FAIL: ruff format" >&2; return 1; }
    uv run mypy src/ || { echo "GATE FAIL: mypy" >&2; return 1; }
    uv run pytest tests/unit/ -x -q || { echo "GATE FAIL: pytest" >&2; return 1; }

    echo "✓  Gate passed for ${base:0:8}..HEAD"
}

run_iteration() {
    local iter=$1
    local timestamp iter_padded log_file
    timestamp=$(date '+%Y%m%d-%H%M%S')
    iter_padded=$(printf '%02d' "$iter")
    log_file="$LOG_DIR/iteration-${iter_padded}-${timestamp}.log"

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
    return "$status"
}

failed=0
for i in $(seq 1 $ITERATIONS); do
    if ! require_clean_tree; then
        echo "Aborting loop before iteration $i." >&2
        exit 1
    fi

    base=$(git rev-parse HEAD)

    if ! run_iteration "$i"; then
        (( failed++ )) || true
    fi

    if ! run_gate "$base"; then
        echo "" >&2
        echo "════════════════════════════════════════════════════════" >&2
        echo "  Deterministic gate FAILED after iteration $i — stopping." >&2
        echo "  No further autopilot iterations run so nothing builds on an" >&2
        echo "  unverified commit. Review ${base:0:8}..HEAD by hand; the commit" >&2
        echo "  is NOT auto-reverted (a reset is a destructive op — HC3)." >&2
        echo "════════════════════════════════════════════════════════" >&2
        exit 1
    fi
done

echo ""
echo "════════════════════════════════════════════════════════"
echo "  All $ITERATIONS iterations finished."
echo "  Failures: $failed"
echo "  Logs in:  $LOG_DIR/"
echo "════════════════════════════════════════════════════════"

exit "$failed"
