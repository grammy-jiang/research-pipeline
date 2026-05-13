# Phase E Acceptance Gate Addendum

Append to `acceptance-gates.md`.

## Phase E Gate

E01-E08 audit_pass; no pending/in_progress/failed/reopened/blocked E tickets; A/B/C/D tests still pass; manual dossier CLI works; missing primary artifact rejected; evidence timeline generated; claims carry evidence/inference/speculation labels; overlong or under-evidenced dossiers rejected; daily brief can link to manual dossier output without bloat.

Required commands:

```bash
uv run pytest tests/unit/test_briefing_*dossier*.py tests/unit/test_briefing_validate_dossier.py -xvs
uv run pytest tests/integration_offline/test_briefing_phase_e_dossier_e2e.py -xvs
uv run pytest tests/integration_offline/test_briefing_phase_a_e2e.py tests/integration_offline/test_briefing_phase_b_memory_e2e.py tests/integration_offline/test_briefing_phase_c_obsidian_e2e.py tests/integration_offline/test_briefing_phase_d_feedback_e2e.py -xvs
uv run ruff format .
uv run ruff check . --fix
uv run mypy src/
```
