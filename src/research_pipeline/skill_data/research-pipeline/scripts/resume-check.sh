#!/usr/bin/env bash
# resume-check.sh — Detect a prior research report and prepare resume context.
#
# Usage: resume-check.sh <topic-slug> [<working-dir>]
#
# If a prior report is found:
#   - Renames it to <slug>-research-report.<YYYY-MM-DD>.md (snapshot)
#   - Extracts prior arXiv paper IDs and raw gap lines
#   - Writes resume_context.json to the working directory
#   - Exits 0
#
# If no prior report exists:
#   - Writes resume_context.json with resume=false
#   - Exits 0
#
# This script never fails with exit code 1 — it is detection only.
# The LLM reads resume_context.json to decide how to seed the next run.

set -euo pipefail

SLUG="${1:-}"
CWD="${2:-.}"

if [[ -z "$SLUG" ]]; then
    echo "Usage: resume-check.sh <topic-slug> [<working-dir>]" >&2
    exit 1
fi

# Expand ~ in CWD
CWD="${CWD/#\~/$HOME}"

REPORT_FILE="${CWD}/${SLUG}-research-report.md"
OUTPUT="${CWD}/resume_context.json"
DATE="$(date -u +%F)"

if [[ ! -f "$REPORT_FILE" ]]; then
    python3 -c "
import json
ctx = {
    'resume': False,
    'snapshot': None,
    'original_report': '${REPORT_FILE}',
    'prior_paper_ids': [],
    'open_gaps_raw': [],
    'instructions': 'No prior report found. Run a fresh pipeline.'
}
with open('${OUTPUT}', 'w') as f:
    json.dump(ctx, f, indent=2)
print('No prior report — fresh run.')
"
    echo "No prior report found for '${SLUG}' — fresh run." >&2
    exit 0
fi

SNAPSHOT="${CWD}/${SLUG}-research-report.${DATE}.md"

# Rename the prior report to snapshot (mv, not cp — the original slot is now free)
mv "$REPORT_FILE" "$SNAPSHOT"
echo "Prior report snapshot: $SNAPSHOT" >&2

# Extract arXiv IDs: patterns like 2401.12345, 2401.12345v2
PRIOR_IDS="$(grep -oE '[0-9]{4}\.[0-9]{4,5}(v[0-9]+)?' "$SNAPSHOT" \
    | grep -vE '^[0-9]{4}\.[0-9]{1,3}$' \
    | sort -u \
    | tr '\n' ',' \
    | sed 's/,$//' || true)"

# Extract open gap lines: lines mentioning [ACADEMIC], [ENGINEERING], "GAP:", "gap:", or "Open:"
GAPS_JSON="$(python3 - "$SNAPSHOT" <<'PYEOF'
import sys, re, json
path = sys.argv[1]
patterns = [
    re.compile(r'\[ACADEMIC\]|\[ENGINEERING\]', re.I),
    re.compile(r'^\s*[-*]?\s*(GAP|gap|Open|Unresolved):', re.M),
    re.compile(r'Research Gaps?|Unresolved Questions?', re.I),
]
gaps = []
try:
    with open(path, encoding='utf-8', errors='replace') as f:
        in_gap_section = False
        for line in f:
            stripped = line.strip()
            # Enter gap sections
            if re.search(r'^#{1,3}\s+(Research Gaps?|Unresolved|Assumption Map|Risk Register)', stripped, re.I):
                in_gap_section = True
                continue
            # Leave gap sections at next heading
            if in_gap_section and re.match(r'^#{1,3}\s+', stripped):
                in_gap_section = False
            # Collect items from gap sections or matching inline patterns
            if in_gap_section and stripped and stripped not in ('#', '##', '###'):
                if stripped.startswith(('-', '*', '|')) or stripped[0].isalpha():
                    gaps.append(stripped[:200])
            elif any(p.search(stripped) for p in patterns[:2]):
                gaps.append(stripped[:200])
except Exception as e:
    pass
print(json.dumps(gaps[:30]))
PYEOF
)"

python3 - <<PYEOF
import json, sys

prior_ids_str = """${PRIOR_IDS}"""
gaps_json = """${GAPS_JSON}"""

prior_ids = [pid.strip() for pid in prior_ids_str.split(",") if pid.strip()]
try:
    gaps = json.loads(gaps_json) if gaps_json.strip() else []
except Exception:
    gaps = []

ctx = {
    "resume": True,
    "snapshot": "${SNAPSHOT}",
    "original_report": "${REPORT_FILE}",
    "date": "${DATE}",
    "prior_paper_ids": prior_ids,
    "open_gaps_raw": gaps,
    "instructions": (
        "1. Seed the new pipeline run: pass prior_paper_ids to "
        "'research-pipeline expand --paper-ids \"<ids>\"' so the new run extends "
        "rather than duplicates the earlier shortlist. "
        "2. Add open_gaps_raw as extra query_variants in the new query_plan.json. "
        "3. Regenerate the final report from scratch using combined evidence — "
        "do NOT append to or reference the snapshot. "
        "4. The snapshot is preserved in CWD as a read-only backup."
    ),
}

with open("${OUTPUT}", "w") as f:
    json.dump(ctx, f, indent=2)

print(f"Resume context written to: ${OUTPUT}")
print(f"Prior paper IDs extracted: {len(prior_ids)}")
print(f"Open gaps extracted: {len(gaps)}")
PYEOF

echo "Resume check complete. Read resume_context.json before starting the new pipeline run." >&2
exit 0
