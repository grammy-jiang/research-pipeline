# Phase D Acceptance Gate Addendum

Append to `acceptance-gates.md`.

## Phase D Gate

D01-D08 audit_pass; no pending/in_progress/failed/reopened/blocked D tickets; A/B/C tests still pass; feedback events persist; valid CLI feedback accepted; malformed target IDs rejected; insufficient/conflicting feedback does not change durable ranking; reversible preference updates affect future ranking; rollback restores previous weights; weekly feedback/source-quality section works.

```bash
uv run pytest tests/unit/test_briefing_*feedback*.py tests/unit/test_briefing_preference_update.py tests/unit/test_briefing_rank_feedback.py tests/unit/test_briefing_weekly_feedback.py -xvs
uv run pytest tests/integration_offline/test_briefing_phase_d_feedback_e2e.py -xvs
uv run pytest tests/integration_offline/test_briefing_phase_a_e2e.py tests/integration_offline/test_briefing_phase_b_memory_e2e.py tests/integration_offline/test_briefing_phase_c_obsidian_e2e.py -xvs
uv run ruff format .
uv run ruff check . --fix
uv run mypy src/
```
