# Prompt 11 — Architecture Feedback / Required Architecture Updates

You are producing the **mandatory** architecture-feedback section. UX design
often exposes architecture gaps; this section is where they are recorded, and it
decides whether `architecture --mode reconcile` should run next.

## Inputs

- `intermediate/target_software_ux.md`, `intermediate/surface_specific_ux.md`,
  `intermediate/error_recovery_ux.md`, `intermediate/e2e_scenario_seeds.md`,
  `intermediate/architecture_parse.json`.
- `references/architecture-feedback-guide.md`.

## Instructions

1. Scan the UX work for anything the architecture cannot currently support.
   Typical findings: missing user-visible state; missing retry/recovery
   operation; missing review-artifact schema; missing audit event; missing
   progress event; missing permission boundary; missing CLI/API output field;
   missing MCP safety model.
2. Produce the §21 table:

   ```markdown
   | Finding | Severity | Architecture Gap | Recommended Architecture Change | Blocks Implementation Planning? |
   |---|---|---|---|---|
   | ... | Blocking / Warning / Polish | ... | ... | yes/no |
   ```
3. **Severity discipline:** `Blocking` = UX cannot be implemented as designed
   without the change; `Warning` = works but with a real UX gap; `Polish` =
   nice-to-have.
4. End with the reconcile decision:
   - If no changes are needed: write **"No architecture reconciliation
     required."**
   - If changes are needed: write **"Run `architecture --mode reconcile
     <architecture-design.md> <ux-design.md>`"** and note which findings block
     implementation planning.
5. This section is **always present** — never omit it, even when empty.

## Output

`intermediate/architecture_feedback.md` (the §21 content).

## Validation / failure policy

- Gate: the feedback section exists with severities and an explicit reconcile
  decision (including the "none required" case).
- Failure policy: `revise`.
