# Feature Acceptance Contract: Extend daily report with prior-topic context

## Feature ID

`B06_report_prior_context`

## Goal

Extend daily report with prior-topic context for Phase B memory and fatigue.

Specifically ensure report rendering can:

- omit prior-context text when no matching memory exists
- add concise prior context for resurfaced included items
- add concise prior context when repeated-topic fatigue changes why an item is included
- avoid introducing a verbose standalone history section by default

## In Scope

- Implement only this ticket's Phase B behavior.
- Reuse Phase A briefing surfaces where appropriate.
- Add/update tests before implementation.
- Keep normal tests offline.
- Treat memory as evidence, not truth.
- Preserve Phase A report budgets.
- Write/update proof pack after verification passes.
- Keep prior context concise and inline with the relevant item.
- Do not perform durable topic-memory writes from report rendering.

## Out of Scope

- Phase C+ functionality.
- Obsidian export.
- Hot-topic dossiers.
- Social-source ingestion.
- MCP expansion.
- UI/dashboard.
- Automatic durable aliases or merges without review.
- New dependencies unless explicitly justified.

## Expected Owned Files

- `src/research_pipeline/briefing/report.py`
- `tests/unit/test_briefing_report_memory.py`

## Required Tests

- `tests/unit/test_briefing_report_memory.py`
	- reports without matching memory omit prior-context text
	- resurfaced topics render concise prior-context text
	- cooling/repeated topics render concise prior-context text tied to fatigue
	- no verbose `## Prior Context` history section is added by default

## Required Fixtures

- `Add focused fixtures only if required by the test design.`

## Failure Cases To Cover

- Missing memory records.
- Empty memory store.
- Repeated low-novelty topic.
- Resurfaced dormant topic.
- Verbose prior-history blocks added unconditionally.
- False merge or ambiguous alias suggestion.
- Durable alias/merge attempted without review record.
- Current evidence contradicts stale memory.
- No network calls in normal tests.

## Verification Commands

```bash
uv run pytest tests/unit/test_briefing_report_memory.py -xvs
uv run ruff check src/research_pipeline/briefing tests/
uv run mypy src/research_pipeline/briefing
```

## Acceptance Criteria

This feature is complete only if:

- all required tests pass;
- verification commands pass;
- no later-phase functionality is introduced;
- no forbidden dependency is added;
- memory writes include trigger/effect/rollback metadata where applicable;
- current evidence remains authoritative over stale memory;
- `phase-status.yaml` records the proof pack and verification commands;
- a proof pack is written under `docs/daily-ai-intelligence/proofs/`.

## Proof Pack Path

```text
docs/daily-ai-intelligence/proofs/B06_report_prior_context.md
```
