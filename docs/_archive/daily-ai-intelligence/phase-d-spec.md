# Phase D Spec — Explicit Feedback Loop

## Preconditions

A01-A12, B01-B08, C01-C08 are `audit_pass`; phases A/B/C are complete; D is in_progress.

## Scope

- `FeedbackEvent`
- `FeedbackStore`
- `brief feedback --cluster/--topic/--source --signal ...`
- manual review labels as feedback records
- reversible source/topic preference updates
- negative preferences
- rollback mechanism
- feedback audit validation
- weekly feedback/source-quality section
- offline e2e tests

## Signals

`keep`, `hide`, `more_like_this`, `less_like_this`, `too_noisy`, `already_known`, `not_actionable`

## Non-goals

No behavioral feedback, read-time tracking, link-click tracking, Obsidian telemetry, dossiers, social sources, MCP expansion, UI.

## Promotion rule

A durable preference change requires: trigger, procedure, observable effect, rollback path, review record, before/after values.

Conflicting or insufficient feedback must be a no-op or require review.

## Ranking extension

rank_score = phase_c_rank_score + explicit_topic_adjustment + explicit_source_adjustment - explicit_negative_penalty

## Completion

D01-D08 audit_pass; feedback persists; CLI records valid feedback; malformed IDs rejected; insufficient/conflicting feedback does not change durable ranking; rollback restores weights; weekly feedback section works; A/B/C tests still pass.
