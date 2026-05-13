# Daily AI Intelligence — Phase B Spec

## Purpose

Phase B adds memory and fatigue to the daily briefing pipeline.

The goal is to stop repeated low-novelty topics from appearing unnecessarily while correctly surfacing resurfaced topics with prior context.

## Preconditions

Do not start Phase B unless:

- A01-A12 are `audit_pass`;
- Phase A gate passes;
- `phases.A.status` is `complete`;
- `phases.B.status` is `in_progress`.

## Scope

Phase B includes:

- `TopicMemory`
- lifecycle values: `new`, `active`, `cooling`, `dormant`, `resurfaced`
- topic appearance tracking
- SQLite-backed topic memory store
- memory lookup by topic ID, title tokens, canonical URL, and explicit links
- fatigue penalty
- resurfaced-topic boost
- prior-topic references in daily brief
- reviewable topic alias/merge suggestions
- validation that no durable alias/merge is applied without review
- offline e2e tests for repeated, resurfaced, and false-merge scenarios

## Non-goals

Phase B does not include:

- Obsidian export
- hot-topic dossiers
- feedback learning
- behavioral tracking
- social-source expansion
- HN/X/Reddit/Bluesky
- MCP tool expansion
- UI/dashboard

## Core principle

Memory is evidence, not truth.

Current evidence must always be checked before ranking or reporting.

## Memory write rule

Every memory write must record:

- trigger condition
- observed effect
- rollback metadata
- source cluster/event IDs
- timestamp
- owning ticket or command
- whether review is required

## Alias/merge rule

A new `TopicMemory` entry may be created automatically.

Durable aliases and topic merges must not be applied automatically.

Suggested aliases/merges must be stored as reviewable suggestions before promotion.

## Ranking extension

```text
rank_score =
  phase_a_rank_score
- fatigue_penalty
+ resurfaced_topic_boost
+ new_evidence_boost
```

Fatigue must not suppress strong new primary evidence.

## Report extension

Show prior context only when it changes why the item is included, suppressed, or marked as resurfaced.

Do not add verbose history blocks by default.

## Completion gate

Phase B is complete only when:

- B01-B08 are `audit_pass`;
- repeated low-novelty topics can be suppressed;
- resurfaced topics can be detected;
- false merges are rejected or sent to review;
- report output remains within Phase A budgets;
- Phase A normal/low-signal/no-news tests still pass.
