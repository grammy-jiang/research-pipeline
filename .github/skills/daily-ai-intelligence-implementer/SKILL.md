---
name: daily-ai-intelligence-implementer
description: Use this skill when implementing the Daily AI Intelligence pipeline, especially Phase A thin daily briefing tickets.
---

# Daily AI Intelligence Implementer

You are implementing the Daily AI Intelligence pipeline as an audited, ticket-driven workflow.

Your job is not to finish vaguely. Your job is to advance the implementation state machine.

## Always read first

- `docs/daily-ai-intelligence/phase-status.yaml`
- `docs/daily-ai-intelligence/implementation-backlog.md`
- `docs/daily-ai-intelligence/acceptance-gates.md`
- `docs/daily-ai-intelligence/startup-audit-policy.md`
- `docs/daily-ai-intelligence/copilot-execution-rules.md`
- `docs/daily-ai-intelligence/phase-a-spec.md`

## Mandatory startup audit

Every session begins with startup audit.

Do not trust `done`, `verified`, or `audit_pass` blindly.

If a completed ticket fails audit, mark it `reopened` and fix the earliest reopened ticket first.

## Mandatory acceptance contract

Before implementation, create or update the current ticket's acceptance contract.

The contract must say how the feature will be proven complete.

## Mandatory test-first behavior

Write or update tests and fixtures before implementation whenever the ticket changes behavior.

## Mandatory proof pack

After implementation, write a proof pack proving what passed.

## Stop condition

Stop only when the current ticket's deterministic checks pass, `phase-status.yaml` is updated, and the next pending/reopened ticket is identified.

## Phase A focus

Implement only Phase A until all A01-A12 tickets pass audit.

## Do not

- start Phase B early
- implement Obsidian
- implement dossiers
- add social sources
- add MCP tools before CLI stabilizes
- use browser scraping
- rely on LLM ranking
- mark project complete from intuition

# Verification Policy

Verification must be deterministic.

Prefer these layers:

1. Unit tests.
2. Offline integration tests.
3. Offline end-to-end tests.
4. Report validator.
5. Artifact layout checks.
6. Linting/type checks.
7. Manual review only after deterministic checks pass.

Do not rely only on snapshot tests.

Tests must assert:

- schema fields
- stable IDs
- link counts
- item counts
- required report sections
- ranking scores
- tie-breaker ordering
- validation errors
- telemetry records where relevant

# Phase F Skill Addendum — Source Expansion

Use this when `phase-status.yaml` shows Phase F is active.

## Core rule

Add one source class at a time, disabled by default, with source registry policy, offline fixtures, retention policy, and side-by-side report comparison.

## Do not

- scrape browser pages
- build a social firehose
- auto-enable noisy sources
- start Phase G
- expand MCP tools
- build UI/dashboard
