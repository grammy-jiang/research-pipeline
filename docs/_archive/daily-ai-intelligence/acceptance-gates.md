# Acceptance Gates

## Ticket Complete

A ticket is complete only when:

- acceptance contract exists
- tests and fixtures exist where required
- implementation is scoped to the ticket
- relevant unit tests pass
- relevant integration/e2e tests pass when applicable
- ruff checks pass
- mypy checks pass or justified exceptions are documented
- telemetry/artifact expectations are verified where relevant
- proof pack exists
- `phase-status.yaml` is updated

## Phase A Gate

Phase A is complete only when:

- all A01-A12 tickets are `audit_pass`
- no Phase A ticket is `pending`, `in_progress`, `verification_failed`, `needs_reaudit`, `reopened`, or `blocked`
- CLI commands follow the artifact contracts
- normal, low-signal, and no-news offline e2e tests pass
- report validation catches malformed or over-budget reports
- fixed artifact layout is used
- no forbidden Phase B+ functionality is introduced
- no new dependency is introduced without explicit justification

Required commands:

```bash
uv run pytest tests/unit/test_briefing_*.py -xvs --cov=src/research_pipeline/briefing --cov-report=term-missing
uv run pytest tests/integration_offline/ -xvs
uv run ruff format .
uv run ruff check . --fix
uv run mypy src/
```

## Definition of Done

The phrase "done" means "acceptance gates pass".

It does not mean "the code looks complete".
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
