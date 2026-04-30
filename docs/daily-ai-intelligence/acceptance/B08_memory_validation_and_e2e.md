# Feature Acceptance Contract: Add memory validation and offline Phase B e2e tests

## Feature ID

`B08_memory_validation_and_e2e`

## Goal

Add memory validation and offline Phase B e2e tests for Phase B memory and fatigue.

Specifically ensure Phase B can prove, offline and deterministically, that:

- repeated or cooling topics have a consistent backing memory record
- resurfaced topics are only accepted when their prior dormant memory state is present
- ambiguous fallback matches do not silently collapse into a false durable merge
- review-queue records remain explicit and non-authoritative until reviewed

## In Scope

- Implement only this ticket's Phase B behavior.
- Reuse Phase A briefing surfaces where appropriate.
- Add/update tests before implementation.
- Keep normal tests offline.
- Treat memory as evidence, not truth.
- Preserve Phase A report budgets.
- Write/update proof pack after verification passes.
- Add a dedicated Phase B memory validator without changing Phase A report validation behavior.
- Prove repeated-topic, resurfaced-topic, and false-merge scenarios with offline tests.

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

- `src/research_pipeline/briefing/validate_memory.py`
- `tests/unit/test_briefing_validate_memory.py`
- `tests/integration_offline/test_briefing_phase_b_memory_e2e.py`
- `tests/fixtures/briefing/e2e/memory_repeated_topic/`
- `tests/fixtures/briefing/e2e/memory_resurfaced_topic/`
- `tests/fixtures/briefing/e2e/memory_false_merge/`

## Required Tests

- `tests/unit/test_briefing_validate_memory.py`
- `tests/integration_offline/test_briefing_phase_b_memory_e2e.py`
	- valid repeated-topic state passes validation
	- valid resurfaced-topic state passes validation
	- empty store with only new topics remains valid
	- missing memory for a memory-annotated cluster fails validation
	- ambiguous title/alias fallback match fails validation

## Required Fixtures

- `tests/fixtures/briefing/e2e/memory_repeated_topic/`
- `tests/fixtures/briefing/e2e/memory_resurfaced_topic/`
- `tests/fixtures/briefing/e2e/memory_false_merge/`

## Failure Cases To Cover

- Missing memory records.
- Empty memory store.
- Repeated low-novelty topic.
- Resurfaced dormant topic.
- False merge or ambiguous alias suggestion.
- Durable alias/merge attempted without review record.
- Current evidence contradicts stale memory.
- Memory-annotated cluster has no matching durable topic record.
- Multiple fallback matches exist for the same cluster title/alias.
- No network calls in normal tests.

## Verification Commands

```bash
uv run pytest tests/unit/test_briefing_validate_memory.py tests/integration_offline/test_briefing_phase_b_memory_e2e.py -xvs
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
- the memory validator is read-only and deterministic;
- repeated, resurfaced, and false-merge scenarios are proven offline;
- current evidence remains authoritative over stale memory;
- `phase-status.yaml` records the proof pack and verification commands;
- a proof pack is written under `docs/daily-ai-intelligence/proofs/`.

## Proof Pack Path

```text
docs/daily-ai-intelligence/proofs/B08_memory_validation_and_e2e.md
```
