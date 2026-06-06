# E2E Scenario Seed Guide

Load this for §20 (`prompts/10`). The `ux-design` skill generates **E2E scenario
seeds**, not executable tests.

## The pipeline

```text
UX user story
  -> acceptance criteria
  -> E2E scenario seed        (this skill stops here)
  -> implementation-plan test task
  -> executable test
```

A seed is a declarative, Gherkin-style scenario derived from a user story and
its acceptance criteria. Later stages turn a seed into a concrete test for the
chosen surface(s): CLI E2E test, API E2E test, Web E2E test, MCP tool-call test,
or AI-skill invocation test.

## What a seed is

```gherkin
Feature: <capability>

Scenario: <happy / alternative / failure scenario>
  Given <precondition tied to architecture state>
  When <user / agent action>
  Then <observable outcome>
  And an audit record is created
```

## What a seed is NOT

```text
step definitions / glue code
assertions written in a programming language
fixtures, mocks, or test data files
test-runner or CI configuration
exact selectors, endpoints, or CLI flags
```

## Coverage rules

- Always seed the **primary happy path**.
- Seed the **main failure / recovery paths** (provider unavailable, validation
  error, degraded output, permission denied — see §15).
- Seed each **user-visible state that must be testable**.
- Seed the **human-review path** if the architecture has a review flow.
- Tie each seed to the **surfaces** it applies to (CLI / Web / API / MCP / AI
  skill).
- Keep `Then` clauses to **observable** outcomes (what the user/agent sees or
  what is recorded) — never internal implementation details.

## Quality check

A seed is good if a later stage can mechanically turn it into a test for a chosen
surface without re-deriving the intent. If a seed is not testable (vague `Then`,
no observable outcome), it is a WARNING in the quality gate.
