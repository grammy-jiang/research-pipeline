#!/usr/bin/env bash
# validate-registry.sh — Check that a daily-ai-intelligence registry file
# exists and has at least one enabled source.
#
# Usage: validate-registry.sh <registry-path>
# Exit 0: registry is valid and has at least one enabled source.
# Exit 1: registry is missing, unreadable, or has no enabled sources.

set -euo pipefail

REG="${1:-}"

if [[ -z "$REG" ]]; then
    echo "Usage: validate-registry.sh <registry-path>" >&2
    exit 1
fi

# Expand ~ in path
REG="${REG/#\~/$HOME}"

if [[ ! -f "$REG" ]]; then
    echo "ERROR: Registry file not found: $REG" >&2
    echo "" >&2
    echo "Copy the template and edit it:" >&2
    echo "  cp ~/.claude/skills/daily-ai-intelligence/config.toml ~/my-daily-registry.toml" >&2
    echo "  # Edit ~/my-daily-registry.toml: enable real sources, set cadences" >&2
    exit 1
fi

# Check for at least one enabled = true source
if ! grep -qE 'enabled[[:space:]]*=[[:space:]]*true' "$REG"; then
    echo "ERROR: No enabled sources found in: $REG" >&2
    echo "" >&2
    echo "Edit the registry and set 'enabled = true' for at least one [[briefing.sources]] entry." >&2
    echo "Tip: The bundled config.toml has only a disabled example source by design — copy and edit it." >&2
    exit 1
fi

# Count enabled sources
ENABLED_COUNT="$(grep -cE 'enabled[[:space:]]*=[[:space:]]*true' "$REG" || echo 0)"

# Warn if only the bundled example source is enabled
if grep -qE 'source_id[[:space:]]*=[[:space:]]*"example-manual"' "$REG" 2>/dev/null && \
   [[ "$ENABLED_COUNT" -le 1 ]]; then
    echo "WARNING: Only the 'example-manual' placeholder source appears enabled." >&2
    echo "         Replace it with your actual sources before running a live brief." >&2
    # Warn only — fixtures-only runs are valid for testing
fi

echo "Registry valid: $REG" >&2
echo "  Enabled sources: $ENABLED_COUNT" >&2
exit 0
