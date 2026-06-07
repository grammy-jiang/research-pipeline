# Prompt 22 — Architecture Draft

You are assembling the full architecture document draft from the intermediate
artifacts.

## Inputs

- All `intermediate/*` artifacts (strategy, goals, tech stack, matrix, C4,
  contracts, data, security, observability, failure, testing, ADRs, rule-pack
  review).
- `intermediate/blueprint_parse.json` (the `product_experience` §9 facts and the
  `recommended_next_stages` §19 routing).
- `intermediate/existing_architecture_status.json` (update mode).
- `templates/architecture_design_template.md` (the 27-section skeleton),
  `references/experience-architecture-guide.md`,
  `references/next-stages-and-handoffs-guide.md`.

## Instructions

1. Compose all 27 sections in order, starting with `## Contents` and
   `## Update History` near the top.
2. Fill §1 generation metadata by copying known values; use `unknown` where a
   value is unavailable. Never invent metadata.
2a. **Author §23 Experience Architecture** from `product_experience` (blueprint
   §9): UX direction inherited, interaction surface matrix, user-visible state
   model (mapped onto the §14 canonical states), feedback/progress model, error/
   recovery model (aligned with §18), human-review technical flow (producing §16
   audit events), trust/transparency support, interaction observability (routed
   to §16), and the UX handoff. Architecture-level UX support only — no screen
   layouts, wireframes, exact CLI syntax, or copy. Depth follows the §19
   ux-design routing (`references/experience-architecture-guide.md`).
2b. **Author §24 Recommended Next Stages and Downstream Handoffs** from
   `recommended_next_stages` (blueprint §19): reflect each stage's
   RUN/SKIP/DEFER/ASK_USER decision; pointer to the §7.2 Tech-Stack Selection
   Handoff; UX / security-review / test-design handoffs per their routing; and
   update/reconciliation triggers. Do not invent stages the blueprint did not
   route (`references/next-stages-and-handoffs-guide.md`).
2c. **Cross-Section Consistency Pass.** After drafting §23 and §24, verify:
   - Every user-facing MVP operation in §23 exists in §12 Interface Contracts.
   - Every user-visible state in §23.3/§23.4 maps onto a §14 lifecycle state,
     condition flag, or audit event.
   - Every human-review action in §23.6 has a §12 contract, §14 state
     transition, §16 audit event, and §18 failure behaviour.
   - Every progress item in §23.4 has an observability event in §16.
   - §24 handoff tables mention only operations already formalized in the
     architecture body or explicitly marked deferred/future.
   Log any gap found as an open question in §25 and as a warning in §27, so
   downstream skills (ux-design, implementation-plan) find the gap in one place
   rather than discovering it too late.
3. Respect the update mode:
   - `regenerate` → rebuild the whole document; append an Update History row.
   - `patch` → change only the affected sections; append an Update History row
     listing them; keep other sections intact.
   - `adr-only` → add/supersede ADRs and update §21 + Update History only.
   - `compare` → emit a diff-style review, do not rewrite the document.
   - `resume` → continue from previously unresolved questions.
4. Include the required Mermaid C4 views (context, container, dynamic) and the
   Traditional-vs-AI matrix and tech-stack table.
5. **Respect the output detail budget.** All 27 sections are always present, but
   match their depth to `output_detail`:
   - `standard` (default): keep the main body concise and decision-focused.
     Concrete targets — Executive Summary ≤ 1 page; tech-stack table ≤ ~15
     decisions; interface contracts = core contracts only; data contracts =
     core entities only; security = trust zones + major controls; observability
     = required IDs/logs/metrics/traces summary; testing = strategy + key tests;
     §23 Experience Architecture and §24 Recommended Next Stages = compact tables
     (≤ ~1 page each); ADRs = a summary table in the body. Move heavy material
     (full schemas,
     extended risk/threat tables, full test matrices, full ADR bodies, full
     log/metric catalogue) into appendices or `adr/` files. The result is a
     concise main body + appendices, not a full dossier. If it still reads like
     a dossier, either compress further or relabel `detailed`.
   - `detailed`: full schemas, full ADR bodies, extended risks, and detailed
     test matrices may live in the main body.
   - `concise`: tighten further; keep every major decision visible.
   Never drop a required section or a major decision to meet the budget.
6. **Surface warnings.** After the §26 self-check, collect every `WARNING` /
   `PASS with warning` row and copy a short
   "Architecture Warnings Requiring Attention" summary
   (Warning · Required Action · Blocks Implementation Planning?) into
   **§1 Executive Architecture Summary** and **§27 Handoff Notes**.
   Warnings must not live only in §26 — a downstream implementation-plan agent
   reading the handoff must see them. If there are no warnings, say so briefly.

## Output

`intermediate/architecture_draft.md`.

## Validation / failure policy

- Gate: the draft respects the update mode and contains all required sections.
- Failure policy: `revise`.

## Cross-Skill Artifact Contract Compliance

Comply with the Cross-Skill Artifact Contract (`references/artifact-contract.md`).
The output document must expose the contract fields using the controlled
vocabulary:

- **Generation Metadata** including `Artifact Type` (a registry value) and a
  stable `Topic Slug` (carried unchanged across the pipeline).
- **Source Artifacts Consumed** (what was read and how it was used).
- **Resolved Input Artifacts** when inputs were auto-discovered (else
  `NOT_APPLICABLE — all input artifacts were explicitly supplied by the user`).
- A **decision register** (controlled status values), **assumptions** kept
  separate from decisions, **open questions** assigned to a next stage, and a
  **Recommended Next Stage** (RUN / SKIP / DEFER / ASK_USER).
- A **Quality-Gate Self-Check** that includes the **Cross-Skill Artifact
  Contract Gate**.

If a section already exists under this skill's own heading, align it to the
contract (a Contract Field Map is fine) rather than duplicating. Mark any
not-applicable field `NOT_APPLICABLE — <reason>`; never omit it.
