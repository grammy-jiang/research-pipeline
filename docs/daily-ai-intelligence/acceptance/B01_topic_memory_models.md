# Feature Acceptance Contract: Add topic memory models

## Feature ID

`B01_topic_memory_models`

## Goal

Add topic memory models for Phase B memory and fatigue.

Specifically ensure the model layer defines immutable, JSON-serializable
contracts for:

- durable topic state (`TopicMemory`)
- reviewable alias suggestions (`TopicAliasSuggestion`)
- auditable memory writes (`TopicMemoryWriteRecord`)

## In Scope

- Implement only this ticket's Phase B behavior.
- Reuse Phase A briefing surfaces where appropriate.
- Add/update tests before implementation.
- Keep normal tests offline.
- Treat memory as evidence, not truth.
- Preserve Phase A report budgets.
- Write/update proof pack after verification passes.

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

- `src/research_pipeline/briefing/models.py`
- `src/research_pipeline/briefing/__init__.py`
- `tests/unit/test_briefing_topic_memory.py`

## Required Tests

- `tests/unit/test_briefing_topic_memory.py`
	- `TopicMemory` construct → serialize → deserialize roundtrip
	- `TopicMemoryWriteRecord` requires trigger/effect/rollback metadata,
	  source IDs, timestamp, owner, and review state
	- `TopicAliasSuggestion` requires a review record once status becomes
	  `approved` or `rejected`

## Required Fixtures

- `Add focused fixtures only if required by the test design.`

## Failure Cases To Cover

- Missing trigger/effect/rollback metadata.
- Missing source cluster and event IDs for a memory write.
- Missing timestamp or owning ticket/command for a memory write.
- Approved/rejected alias suggestion without a review record.
- JSON roundtrip changes immutable model values.
- No network calls in normal tests.

## Verification Commands

```bash
uv run pytest tests/unit/test_briefing_topic_memory.py -xvs
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
docs/daily-ai-intelligence/proofs/B01_topic_memory_models.md
```
