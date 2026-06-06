# Architecture Feedback Guide

Load this for §21 (`prompts/11`). The architecture-feedback section is
**mandatory** — it is the UX skill's structured channel back to the architecture
and the trigger for `architecture --mode reconcile`.

## Why this exists

UX design frequently discovers that the product experience needs something the
architecture has not yet defined. The UX skill must **not** silently invent it
(that would make the UX design inconsistent with the architecture). Instead it
records the gap here, so the architecture can be reconciled.

## What to look for

```text
missing user-visible state            (a story needs a state the model lacks)
missing retry / recovery operation    (the recovery UX has no architecture path)
missing review-artifact schema        (human review needs a packet not defined)
missing audit event                   (a UX action should be audited but isn't)
missing progress event                (progress UX has no observability event)
missing permission boundary           (a surface needs an undefined permission)
missing CLI / API output field        (a surface needs data not in a contract)
missing MCP safety model              (agent UX needs undefined tool guardrails)
```

## The §21 table

```markdown
| Finding | Severity | Architecture Gap | Recommended Architecture Change | Blocks Implementation Planning? |
|---|---|---|---|---|
| ... | Blocking / Warning / Polish | ... | ... | yes/no |
```

## Severity

- **Blocking** — the UX cannot be implemented as designed without the
  architecture change. Blocks implementation planning until reconciled.
- **Warning** — the UX works but has a real gap (degraded experience, missing
  audit/observability). Should be reconciled but does not block.
- **Polish** — a nice-to-have improvement.

## Reconcile decision (always state one)

- No findings (or only Polish that the user accepts): write **"No architecture
  reconciliation required."**
- Any Blocking/Warning finding: write **"Run `architecture --mode reconcile
  <architecture-design.md> <ux-design.md>`"** and list which findings block
  implementation planning.

## Discipline

- The section is **always present**, even when empty.
- Recommend *minimal* architecture changes — describe the gap and the smallest
  change that closes it; do **not** redesign the architecture here (that is the
  architecture skill's job during reconcile).
- Keep findings traceable to a specific user story, surface, or failure state.
