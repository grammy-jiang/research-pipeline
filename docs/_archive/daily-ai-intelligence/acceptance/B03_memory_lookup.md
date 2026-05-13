# Feature Acceptance Contract: Add memory lookup for recent clusters

## Feature ID

`B03_memory_lookup`

## Goal

Add memory lookup for recent clusters for Phase B memory and fatigue.

Specifically ensure lookup works by:

- explicit topic IDs on incoming clusters
- normalized topic-title matching against existing memory names/aliases
- canonical URL overlap via canonical cluster linkage

## In Scope

- Implement only this ticket's Phase B behavior.
- Reuse Phase A briefing surfaces where appropriate.
- Add/update tests before implementation.
- Keep normal tests offline.
- Treat memory as evidence, not truth.
- Preserve Phase A report budgets.
- Write/update proof pack after verification passes.
- Keep alias suggestions reviewable-only (no durable merge/alias writes).
- Prefer current cluster evidence over stale memory when conflicts exist.

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

- `src/research_pipeline/briefing/memory_lookup.py`
- `tests/unit/test_briefing_memory_lookup.py`

## Required Tests

- `tests/unit/test_briefing_memory_lookup.py`
	- missing memory records return empty lookup results
	- empty store returns empty lookup results
	- lookup by topic ID returns matching memory
	- lookup by normalized title/alias returns matching memory
	- lookup by canonical-cluster linkage returns matching memory
	- ambiguous title-only matches yield reviewable alias suggestions, not writes
	- conflicting stale memory does not override current cluster evidence fields

## Required Fixtures

- `Add focused fixtures only if required by the test design.`

## Failure Cases To Cover

- Missing memory records.
- Empty memory store.
- Repeated low-novelty topic.
- Resurfaced dormant topic.
- False merge or ambiguous alias suggestion.
- Durable alias/merge attempted without review record.
- Current evidence contradicts stale memory.
- No network calls in normal tests.
- Alias suggestion generation must not perform durable store mutations.

## Verification Commands

```bash
uv run pytest tests/unit/test_briefing_memory_lookup.py -xvs
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
docs/daily-ai-intelligence/proofs/B03_memory_lookup.md
```
