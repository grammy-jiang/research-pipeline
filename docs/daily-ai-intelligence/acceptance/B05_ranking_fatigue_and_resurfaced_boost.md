# Feature Acceptance Contract: Extend ranking with fatigue penalty and resurfaced-topic boost

## Feature ID

`B05_ranking_fatigue_and_resurfaced_boost`

## Goal

Extend ranking with fatigue penalty and resurfaced-topic boost for Phase B memory and fatigue.

Specifically ensure deterministic ranking behavior for:

- no-memory clusters remaining unpenalized
- repeated low-novelty topics receiving a fatigue penalty
- resurfaced topics receiving a positive boost and `resurfaced` novelty type
- strong resurfaced primary evidence remaining rank-eligible despite fatigue

## In Scope

- Implement only this ticket's Phase B behavior.
- Reuse Phase A briefing surfaces where appropriate.
- Add/update tests before implementation.
- Keep normal tests offline.
- Treat memory as evidence, not truth.
- Preserve Phase A report budgets.
- Write/update proof pack after verification passes.
- Keep ranking deterministic and bounded to the existing Phase A scoring surface.
- Do not perform durable topic-memory writes from ranking.

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

- `src/research_pipeline/briefing/rank.py`
- `tests/unit/test_briefing_rank_memory.py`

## Required Tests

- `tests/unit/test_briefing_rank_memory.py`
	- clusters without memory keep zero fatigue/resurfaced adjustments
	- repeated low-novelty topic ranks below an equivalent fresh topic
	- resurfaced topic receives a positive boost and `resurfaced` novelty type
	- strong resurfaced primary evidence is not filtered out solely by fatigue

## Required Fixtures

- `Add focused fixtures only if required by the test design.`

## Failure Cases To Cover

- Missing memory records.
- Empty memory store.
- Repeated low-novelty topic.
- Resurfaced dormant topic.
- Strong primary resurfacing evidence suppressed below ranking threshold.
- False merge or ambiguous alias suggestion.
- Durable alias/merge attempted without review record.
- Current evidence contradicts stale memory.
- No network calls in normal tests.

## Verification Commands

```bash
uv run pytest tests/unit/test_briefing_rank_memory.py -xvs
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
docs/daily-ai-intelligence/proofs/B05_ranking_fatigue_and_resurfaced_boost.md
```
