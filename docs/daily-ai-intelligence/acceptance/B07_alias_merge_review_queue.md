# Feature Acceptance Contract: Add review queue for topic alias and merge suggestions

## Feature ID

`B07_alias_merge_review_queue`

## Goal

Add review queue for topic alias and merge suggestions for Phase B memory and fatigue.

Specifically ensure the review queue can:

- persist reviewable alias suggestions into the durable pending queue
- list pending or decided review items deterministically
- deduplicate repeated queue submissions for the same topic/alias pair
- require an explicit review record before approval or rejection changes durable state

## In Scope

- Implement only this ticket's Phase B behavior.
- Reuse Phase A briefing surfaces where appropriate.
- Add/update tests before implementation.
- Keep normal tests offline.
- Treat memory as evidence, not truth.
- Preserve Phase A report budgets.
- Write/update proof pack after verification passes.
- Keep B07 as a thin queue/review layer over the existing topic-memory store.
- Do not auto-approve aliases or perform durable merges without review.

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

- `src/research_pipeline/briefing/topic_review.py`
- `tests/unit/test_briefing_topic_review.py`

## Required Tests

- `tests/unit/test_briefing_topic_review.py`
	- empty store returns an empty review queue
	- queueing alias suggestions persists pending items
	- repeated submissions for the same alias are deduplicated
	- approving or rejecting without a review record is rejected
	- approving with a review record updates alias state and queue status

## Required Fixtures

- `Add focused fixtures only if required by the test design.`

## Failure Cases To Cover

- Missing memory records.
- Empty memory store.
- Repeated low-novelty topic.
- Resurfaced dormant topic.
- False merge or ambiguous alias suggestion.
- Duplicate queue items for the same alias suggestion.
- Durable alias/merge attempted without review record.
- Current evidence contradicts stale memory.
- No network calls in normal tests.

## Verification Commands

```bash
uv run pytest tests/unit/test_briefing_topic_review.py -xvs
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
docs/daily-ai-intelligence/proofs/B07_alias_merge_review_queue.md
```
