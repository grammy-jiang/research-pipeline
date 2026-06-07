# Open Question Ledger Guide

Load this when producing `<topic-slug>-open-question-ledger.md` in `materialize`
mode. The open-question ledger centralizes all open questions from the pipeline
so implementation-plan has a single, prioritized list with owners.

## Purpose

Open questions accumulate across multiple pipeline artifacts:

```text
blueprint §18 Open Questions
architecture §25 Open Questions
architecture update notes §10/§13 Remaining Open Questions
ux-design §20 Open Questions
security-review findings
```

Without centralization, implementation agents must scan every document for open
questions and may miss them.

## When to create / update

- Created by `architecture --mode materialize`.
- Updated whenever `materialize` is re-run.
- Implementation-plan may also add new questions discovered during task breakdown.

## Rules

```text
No open question should remain ownerless.
Every open question must have:
  - ID (unique, prefixed by source category)
  - Source artifact (where first raised)
  - Owner stage (who resolves it)
  - Blocking status (what it blocks)
  - Required resolution action
```

## ID Prefix Convention

| Prefix | Source category |
|---|---|
| `OQ-N` | Architecture open questions (numbered) |
| `OQ-WN` | Security/warning-origin questions |
| `OQ-UXN` | UX-design-origin questions |
| `OQ-BPN` | Blueprint-origin questions |
| `OQ-SRN` | Security-review-origin questions |

## Question Status Values

| Status | Meaning |
|---|---|
| `OPEN` | Unresolved; needs action |
| `RESOLVED` | Resolution documented |
| `DEFERRED` | Explicitly deferred to a later stage |

## Owner Stage Values

Use only known pipeline stages:

```text
implementation-plan
security-review
test-design
ux-design
architecture
blueprint
```

## Blocking Status

Each open question should state what it blocks:

```text
MVP-0             — blocks the first minimum viable product milestone
MVP-1             — blocks the first feature-complete milestone
security-release  — blocks any release requiring security sign-off
implementation    — blocks implementation start
none              — informational; does not block a milestone
```

## Template

See `templates/open_question_ledger_template.md` for the full template.

## Example Entry

```markdown
| OQ-1 | Default backbone LLM model for academic plugin | architecture §25 | implementation-plan | MVP-0 config | OPEN | benchmark candidate models; pick top 3 with latency + cost profile |
| OQ-W2 | Glossary/terminology sanitization spec | security-review/update-2 | implementation-plan | security-release | OPEN | define escaping rules, max length, null stripping, injection seed tests |
| OQ-UX3 | Plugin progress bar granularity | ux-design §20 | implementation-plan | MVP-1 | OPEN | agree on stage-level vs step-level events |
```
