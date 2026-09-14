#!/usr/bin/env bash
# Compatibility entry point; arguments are passed as data.
set -euo pipefail
exec python3 "$(dirname "$0")/resume_check.py" "$@"
