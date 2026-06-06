# Prompt update_02 — Apply Accepted Decisions

You are applying **already-accepted** decisions into the architecture. `update`
mode is not review and not conflict analysis — only accepted decisions are
applied, and the architecture design document is not overwritten by default.

## Inputs

- `intermediate/resolved_artifacts.json` (architecture design + the accepted
  update source).
- The architecture design and the update source (tech-stack / reconciliation /
  security-review / newer blueprint / explicit user decision).
- `references/architecture-update-guide.md`,
  `templates/architecture_update_template.md`.

## Instructions

1. **Confirm the update source is accepted**, in priority order: a tech-stack
   with `Architecture Update Required? = Yes`; an accepted reconciliation; a
   security-review with architecture-impacting findings; a newer blueprint;
   explicit user decision text. **Reject ux-design as a direct source** — if the
   only source is ux-design, STOP and recommend `architecture --mode reconcile`
   first.
2. **§4 Accepted Decisions Applied** — one row per accepted decision: Decision ·
   Source · Evidence · Affected Architecture Sections · Applied?. Apply only
   accepted decisions; never speculative comments.
3. **§5 Sections Requiring Update** — list the affected architecture sections with
   the change required, reason, and priority.
4. **§6 Architecture Patch Summary** — per area: Old Assumption → New Decision →
   Patch Summary. Describe the patch; do **not** rewrite the architecture in
   place.
5. **§7 Updated ADRs / Decision Register** — promote accepted ADR candidates;
   supersede prior ADRs (never silently overwrite).
6. **§8 Updated Handoffs** — which §24 / §27 handoffs change.
7. **§9 Compatibility Check** — verify the invariants are preserved (blueprint
   thesis, Product Experience Direction, state model, interface contracts,
   security boundaries, observability, Recommended Next Stages). Flag any that the
   accepted decision would weaken.

## Output

`intermediate/update_applied.md` (the §4–§10 content).

## Validation / failure policy

- Gate: only accepted decisions are applied; changed sections are listed;
  unaffected sections are preserved; invariants are checked; ux-design is not used
  as a direct source.
- Failure policy: `revise` (or `stop` if the only source is unaccepted/speculative).
