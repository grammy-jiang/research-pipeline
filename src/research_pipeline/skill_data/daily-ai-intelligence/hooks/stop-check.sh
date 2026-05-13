#!/usr/bin/env bash
# daily-ai-intelligence stop-check hook
#
# Blocks the agent from declaring completion if today's brief failed validation
# or was never validated.
#
# Compatible with:
#   Claude Code   — Stop event   (exit 2 → prevents Claude from stopping)
#   Copilot CLI   — agentStop    (exit 2 → prevents agent from stopping)
#   Codex CLI     — Stop event   (exit 2 → sends stderr as continuation prompt)
#
# The hook is a no-op when no brief workspace is found for today, so it is safe
# to register globally.

# Drain stdin; agents pipe hook event JSON to stdin.
if [[ ! -t 0 ]]; then cat > /dev/null 2>/dev/null || true; fi

# ── 1. Require Python or bash check_completion.sh ────────────────────────────
CHECK_SCRIPT=""
for skill_dir in \
    "${HOME}/.claude/skills/daily-ai-intelligence" \
    "${HOME}/.copilot/skills/daily-ai-intelligence" \
    "${HOME}/.agents/skills/daily-ai-intelligence" \
    "${HOME}/.codex/skills/daily-ai-intelligence"; do
    candidate="${skill_dir}/scripts/check_completion.sh"
    if [[ -f "$candidate" ]]; then
        CHECK_SCRIPT="$candidate"
        break
    fi
done

[[ -n "$CHECK_SCRIPT" ]] || exit 0   # skill not installed locally; fail open

# ── 2. Run the brief completion check ────────────────────────────────────────
# check_completion.sh auto-detects today's workspace by convention (./workspace).
# It exits 0 when no brief has been run today (nothing to check), exits 1 when
# a brief exists but fails validation.
if ! OUTPUT="$(bash "$CHECK_SCRIPT" 2>&1)"; then
    cat >&2 <<MSG
=== daily-ai-intelligence: Stop blocked ===
Completion check failed: today's brief is incomplete or failed validation.

$OUTPUT

Resolve the validation issues above. When the brief passes, the hook will
allow the agent to stop. To re-check manually:
  bash $CHECK_SCRIPT
MSG
    exit 2
fi

exit 0
