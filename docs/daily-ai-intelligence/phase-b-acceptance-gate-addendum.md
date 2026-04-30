# Phase B Acceptance Gate Addendum

Append this section to `docs/daily-ai-intelligence/acceptance-gates.md`.

## Phase B Gate

Phase B is complete only when:

- all B01-B08 tickets are `audit_pass`;
- no Phase B ticket is `pending`, `in_progress`, `verification_failed`, `needs_reaudit`, `reopened`, or `blocked`;
- Phase A tests still pass;
- repeated low-novelty topics can be suppressed;
- resurfaced topics can be detected;
- false topic merges are rejected or sent to review;
- durable aliases/merges cannot be applied without review record;
- report output remains within Phase A report budgets;
- memory writes include trigger, effect, rollback metadata, source IDs, and timestamp.

Required commands:

```bash
uv run pytest tests/unit/test_briefing_*memory*.py tests/unit/test_briefing_lifecycle.py tests/unit/test_briefing_rank_memory.py tests/unit/test_briefing_report_memory.py -xvs
uv run pytest tests/integration_offline/test_briefing_phase_b_memory_e2e.py -xvs
uv run pytest tests/integration_offline/test_briefing_phase_a_e2e.py -xvs
uv run ruff format .
uv run ruff check . --fix
uv run mypy src/
```
