#!/usr/bin/env bash
# research-pipeline stop-check hook
#
# Blocks the agent from declaring completion if required research artifacts
# are missing (synthesis_report.json, validation results, final report, etc.).
#
# Compatible with:
#   Claude Code   — Stop event   (exit 2 → prevents Claude from stopping)
#   Copilot CLI   — agentStop    (exit 2 → prevents agent from stopping)
#   Codex CLI     — Stop event   (exit 2 → sends stderr as continuation prompt)
#
# The hook is a no-op when no active research session is detected in CWD
# (i.e., no round_state.json), so it is safe to register globally.

# Drain stdin; all three agents pipe hook event JSON to stdin.
# Reading prevents SIGPIPE errors from large transcript payloads.
if [[ ! -t 0 ]]; then cat > /dev/null 2>/dev/null || true; fi

# ── 1. Detect active research session ────────────────────────────────────────
ROUND_STATE="${PWD}/round_state.json"
[[ -f "$ROUND_STATE" ]] || exit 0

# ── 2. Require Python ────────────────────────────────────────────────────────
command -v python3 &>/dev/null || exit 0   # fail open if Python not available

# ── 3. Extract run_id and topic_slug from round_state.json ───────────────────
RUN_ID="$(python3 - <<EOF 2>/dev/null
import json
try:
    d = json.load(open("${ROUND_STATE}"))
    print(d.get("run_id", ""))
except Exception:
    pass
EOF
)"

SLUG="$(python3 - <<EOF 2>/dev/null
import json
try:
    d = json.load(open("${ROUND_STATE}"))
    print(d.get("topic_slug", ""))
except Exception:
    pass
EOF
)"

[[ -n "$RUN_ID" ]] || exit 0
[[ -n "$SLUG"   ]] || exit 0

# ── 4. Find check_completion.py in known skill install locations ──────────────
CHECK_SCRIPT=""
for skill_dir in \
    "${HOME}/.claude/skills/research-pipeline" \
    "${HOME}/.copilot/skills/research-pipeline" \
    "${HOME}/.agents/skills/research-pipeline" \
    "${HOME}/.codex/skills/research-pipeline"; do
    candidate="${skill_dir}/scripts/check_completion.py"
    if [[ -f "$candidate" ]]; then
        CHECK_SCRIPT="$candidate"
        break
    fi
done

[[ -n "$CHECK_SCRIPT" ]] || exit 0   # skill not installed locally; fail open

# ── 5. Run completion check ───────────────────────────────────────────────────
if ! OUTPUT="$(python3 "$CHECK_SCRIPT" --run-id "$RUN_ID" --slug "$SLUG" 2>&1)"; then
    cat >&2 <<MSG
=== research-pipeline: Stop blocked ===
Completion check failed for run '$RUN_ID' (slug: '$SLUG').

$OUTPUT

Fix the missing artifacts listed above, then respond with "done" or run:
  python3 $CHECK_SCRIPT --run-id $RUN_ID --slug $SLUG
MSG
    exit 2
fi

exit 0
