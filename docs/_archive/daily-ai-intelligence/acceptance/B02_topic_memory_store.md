# Feature Acceptance Contract: Add SQLite-backed topic memory store

## Feature ID

`B02_topic_memory_store`

## Goal

Add SQLite-backed topic memory store for Phase B memory and fatigue.

This ticket reconciles the already-present store behavior into the
ticket-owned module surface and verifies deterministic persistence and
review-safety behavior.

## In Scope

- Implement only this ticket's Phase B behavior.
- Reuse Phase A briefing surfaces where appropriate.
- Add/update tests before implementation.
- Keep normal tests offline.
- Treat memory as evidence, not truth.
- Preserve Phase A report budgets.
- Write/update proof pack after verification passes.
- Provide a ticket-owned import surface at
	`src/research_pipeline/briefing/topic_memory_store.py`.
- Keep backward compatibility for existing imports from
	`research_pipeline.briefing.topic_memory`.
- Enforce review safety so alias approvals/rejections require a
	non-empty review record before durable state changes.

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

- `src/research_pipeline/briefing/topic_memory_store.py`
- `tests/unit/test_briefing_topic_memory_store.py`
- `tests/fixtures/briefing/memory/topic_memory_empty.sqlite`

## Required Tests

- `tests/unit/test_briefing_topic_memory_store.py`
	- missing topic returns `None`
	- empty store has no pending alias suggestions
	- repeated low-novelty reporting increases fatigue and marks topic active
	- resurfaced topic after dormancy is marked resurfaced
	- ambiguous/duplicate alias suggestion is not duplicated
	- approving/rejecting alias without review record is rejected
	- when current evidence arrives, durable topic timestamps and
	  canonical cluster linkage are updated from that evidence

## Required Fixtures

- `tests/fixtures/briefing/memory/topic_memory_empty.sqlite`
	- initialized empty SQLite file used by unit tests

## Failure Cases To Cover

- Missing memory records.
- Empty memory store.
- Repeated low-novelty topic.
- Resurfaced dormant topic.
- False merge or ambiguous alias suggestion.
- Durable alias/merge attempted without review record.
- Current evidence contradicts stale memory.
- No network calls in normal tests.

## Verification Commands

```bash
uv run pytest tests/unit/test_briefing_topic_memory_store.py -xvs
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
docs/daily-ai-intelligence/proofs/B02_topic_memory_store.md
```
