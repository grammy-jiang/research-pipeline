# Prompt 04 — UX Clarification

You are resolving the **high-impact** UX decisions before writing the document.
Use a `grill-me` style: one question at a time, each with a recommended answer,
alternatives, and the default assumption if unanswered.

## Inputs

- `intermediate/architecture_parse.json`, `intermediate/blueprint_parse.json`.
- `references/ux-question-bank.md`.

## Instructions

1. Pick the operating mode: **interactive** (ask high-impact questions),
   **automatic** (infer + record assumptions), or **hybrid (default)** (ask only
   high-impact questions, infer low-risk details).
2. **Ask only if the answer affects** MVP scope, architecture constraints,
   security/privacy, the state model, human review, failure/recovery, E2E
   scenarios, or implementation-plan readiness. Draw from the eight question
   categories in `references/ux-question-bank.md` (Users & Roles; Workflow;
   Interaction Surface; Trust/Control/Transparency; Failure & Recovery; Human
   Review; AI Skill & MCP; E2E Test Orientation).
3. **Do not ask** low-value visual/copy questions (button colour, card-vs-table,
   exact warning wording, exact CLI flags) — those belong to implementation or
   visual design.
4. Use this question format (one at a time):

   ```markdown
   ### UX Clarification Question <N>
   **Question:** ...
   **Why this matters:** ...
   **Recommended answer:** ...
   **Alternatives:** - ...
   **If no answer is provided:** The UX design will assume ...
   ```
5. Record every answer/assumption. **Flag high-risk assumptions** (e.g.
   data-egress shown only on request; human review is rare; CLI users are
   technical enough for JSON; MCP users are trusted agents; review-packet export
   is sufficient for MVP) for review.
6. Do not block generation in hybrid/automatic mode unless a missing answer would
   break the UX design.

## Output

`intermediate/clarifications.md` — the asked questions, the answers/assumptions,
and a UX Assumptions table (Assumption · Source · Confidence · Reversible? ·
Review Trigger) feeding §9.

## Validation / failure policy

- Gate: high-impact UX decisions are resolved or recorded as assumptions.
- Failure policy: `record_assumptions`.
