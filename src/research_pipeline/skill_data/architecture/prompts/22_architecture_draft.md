# Prompt 22 — Architecture Draft

You are assembling the full architecture document draft from the intermediate
artifacts.

## Inputs

- All `intermediate/*` artifacts (strategy, goals, tech stack, matrix, C4,
  contracts, data, security, observability, failure, testing, ADRs, rule-pack
  review).
- `intermediate/existing_architecture_status.json` (update mode).
- `templates/architecture_design_template.md` (the 25-section skeleton).

## Instructions

1. Compose all 25 sections in order, starting with `## Contents` and
   `## Update History` near the top.
2. Fill §1 generation metadata by copying known values; use `unknown` where a
   value is unavailable. Never invent metadata.
3. Respect the update mode:
   - `regenerate` → rebuild the whole document; append an Update History row.
   - `patch` → change only the affected sections; append an Update History row
     listing them; keep other sections intact.
   - `adr-only` → add/supersede ADRs and update §21 + Update History only.
   - `compare` → emit a diff-style review, do not rewrite the document.
   - `resume` → continue from previously unresolved questions.
4. Include the required Mermaid C4 views (context, container, dynamic) and the
   Traditional-vs-AI matrix and tech-stack table.

## Output

`intermediate/architecture_draft.md`.

## Validation / failure policy

- Gate: the draft respects the update mode and contains all required sections.
- Failure policy: `revise`.
