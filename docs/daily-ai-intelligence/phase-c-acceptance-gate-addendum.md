# Phase C Acceptance Gate Addendum

Append to `acceptance-gates.md`.

## Phase C Gate

Phase C is complete only when all C01-C08 are audit_pass; no Phase C ticket is pending/in_progress/failed/reopened/blocked; A/B tests still pass; daily/topic/source notes export; frontmatter valid; unsafe paths rejected; human notes not overwritten; wiki-links preserved; second export idempotent; dry-run reports changes without writing.

Required commands:

```bash
uv run pytest tests/unit/test_briefing_obsidian*.py tests/unit/test_briefing_validate_obsidian.py tests/unit/test_briefing_cli_export_obsidian.py -xvs
uv run pytest tests/integration_offline/test_briefing_phase_c_obsidian_e2e.py -xvs
uv run pytest tests/integration_offline/test_briefing_phase_a_e2e.py tests/integration_offline/test_briefing_phase_b_memory_e2e.py -xvs
uv run ruff format .
uv run ruff check . --fix
uv run mypy src/
```
