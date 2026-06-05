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
5. **Respect the output detail budget.** All 25 sections are always present, but
   match their depth to `output_detail`:
   - `standard` (default): keep the main body concise and decision-focused.
     Concrete targets — Executive Summary ≤ 1 page; tech-stack table ≤ ~15
     decisions; interface contracts = core contracts only; data contracts =
     core entities only; security = trust zones + major controls; observability
     = required IDs/logs/metrics/traces summary; testing = strategy + key tests;
     ADRs = a summary table in the body. Move heavy material (full schemas,
     extended risk/threat tables, full test matrices, full ADR bodies, full
     log/metric catalogue) into appendices or `adr/` files. The result is a
     concise main body + appendices, not a full dossier. If it still reads like
     a dossier, either compress further or relabel `detailed`.
   - `detailed`: full schemas, full ADR bodies, extended risks, and detailed
     test matrices may live in the main body.
   - `concise`: tighten further; keep every major decision visible.
   Never drop a required section or a major decision to meet the budget.
6. **Surface warnings.** After the §24 self-check, collect every `WARNING` /
   `PASS with warning` row and copy a short
   "Architecture Warnings Requiring Attention" summary
   (Warning · Required Action · Blocks Implementation Planning?) into
   **§1 Executive Architecture Summary** and **§25 Handoff Notes**.
   Warnings must not live only in §24 — a downstream implementation-plan agent
   reading the handoff must see them. If there are no warnings, say so briefly.

## Output

`intermediate/architecture_draft.md`.

## Validation / failure policy

- Gate: the draft respects the update mode and contains all required sections.
- Failure policy: `revise`.
