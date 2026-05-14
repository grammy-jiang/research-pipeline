#!/usr/bin/env bash
# research-pipeline resume-inject hook  (UserPromptSubmit / userPromptSubmitted)
#
# At each user prompt, checks whether an active research session exists in
# CWD (round_state.json). If so, injects that session's state as
# additionalContext for the model BEFORE it processes the prompt.
#
# This converts the advisory resume-check.sh into an enforced context
# injection: the model will always know about a prior session without
# the user having to mention it.
#
# Compatible with:
#   Claude Code   — UserPromptSubmit  (exit 0 + JSON stdout → additionalContext)
#   Copilot CLI   — userPromptSubmitted
#   Codex CLI     — UserPromptSubmit
#
# SCOPE: Safe to register as a user-global hook because it is a no-op
# (exits 0, no output) when round_state.json does not exist in CWD.

# Drain stdin only when piped (avoids blocking when run interactively or TTY)
if [[ ! -t 0 ]]; then
    cat > /dev/null 2>/dev/null || true
fi

command -v python3 &>/dev/null || exit 0

ROUND_STATE="${PWD}/round_state.json"
[[ -f "$ROUND_STATE" ]] || exit 0   # no active session → no-op

# Build and emit the additionalContext JSON using Python to handle escaping.
python3 - "$ROUND_STATE" <<'EOF'
import json, sys

try:
    d = json.load(open(sys.argv[1]))
    run_id = d.get("run_id", "")
    if not run_id:
        sys.exit(0)

    slug        = d.get("topic_slug",     "")
    round_num   = d.get("round",          "?")
    status      = d.get("status",         "unknown")
    gaps        = d.get("open_gaps",      [])

    lines = [
        "=== PRIOR RESEARCH SESSION DETECTED (round_state.json in CWD) ===",
        f"  run_id:         {run_id}",
        f"  topic_slug:     {slug}",
        f"  round:          {round_num}",
        f"  status:         {status}",
    ]
    if gaps:
        lines.append(f"  open_gaps:      {len(gaps)} gap(s) still open")
        for g in gaps[:3]:               # show up to 3 gap titles
            title = g if isinstance(g, str) else g.get("title", str(g))
            lines.append(f"    - {title}")
        if len(gaps) > 3:
            lines.append(f"    ... and {len(gaps) - 3} more")
    lines += [
        "",
        "ACTION: If the user is asking to continue, resume, or iterate this",
        "research session, load references/iterative-synthesis.md and resume",
        "from round_state.json. Do NOT start fresh.",
        "If the request is unrelated to this research session, ignore the",
        "above context.",
    ]

    print(json.dumps({
        "hookSpecificOutput": {
            "hookEventName": "UserPromptSubmit",
            "additionalContext": "\n".join(lines)
        }
    }))
except Exception:
    sys.exit(0)
EOF
exit 0
