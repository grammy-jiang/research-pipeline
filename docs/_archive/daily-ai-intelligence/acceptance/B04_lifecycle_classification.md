# Feature Acceptance Contract: Add topic lifecycle classification

## Feature ID

`B04_lifecycle_classification`

## Goal

Add topic lifecycle classification for Phase B memory and fatigue.

Specifically provide deterministic classification for:

- `new` when no matching prior memory exists
- `active` when prior memory exists and freshness remains high
- `cooling` when repetition/fatigue is elevated but still recent
- `dormant` when prior memory is old and no strong new evidence exists
- `resurfaced` when a dormant/old topic reappears with fresh evidence

## In Scope

- Implement only this ticket's Phase B behavior.
- Reuse Phase A briefing surfaces where appropriate.
- Add/update tests before implementation.
- Keep normal tests offline.
- Treat memory as evidence, not truth.
- Preserve Phase A report budgets.
- Write/update proof pack after verification passes.
- Add a pure lifecycle module that does not write to durable store state.
- Keep explicit topic ID evidence authoritative over title-only ambiguity.

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

- `src/research_pipeline/briefing/lifecycle.py`
- `tests/unit/test_briefing_lifecycle.py`

## Required Tests

- `tests/unit/test_briefing_lifecycle.py`
	- missing memory records classify as `new`
	- empty store classifies as `new`
	- repeated low-novelty topic classifies as `cooling`
	- resurfaced dormant topic classifies as `resurfaced`
	- old topic without strong current evidence classifies as `dormant`
	- lifecycle classification does not create alias suggestions or mutate store
	- conflicting stale-memory status is overridden by current explicit evidence

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
- Lifecycle helpers must be read-only and deterministic.

## Verification Commands

```bash
uv run pytest tests/unit/test_briefing_lifecycle.py -xvs
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
docs/daily-ai-intelligence/proofs/B04_lifecycle_classification.md
```
