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
3. **§5 Patch Manifest** — produce a machine-readable YAML Patch Manifest using
   `references/patch-manifest-guide.md`. For every accepted decision that changes
   an architecture section, emit a patch entry with: `id`, `source`, `target_section`,
   `operation`, `patch_type` (from the taxonomy below), `blocks_implementation`,
   and `description`. This is consumed by `architecture --mode materialize`. If a
   patch entry cannot be determined, emit a WARNING and explain why.
4. **Patch-Type Taxonomy** — classify the overall update with one or more patch
   types: `NONE`, `NOTE_ONLY`, `ADR_ONLY`, `CONTRACT_PATCH`, `SECURITY_PATCH`,
   `OBSERVABILITY_PATCH`, `STRUCTURAL_PATCH`, `BREAKING_CHANGE`. Record the type
   in Generation Metadata §1. Multi-type is allowed (e.g. `CONTRACT_PATCH +
   OBSERVABILITY_PATCH`).
5. **§6 Sections Requiring Update** — list the affected architecture sections with
   the change required, reason, and priority.
6. **§7 Architecture Patch Summary** — per area: Old Assumption → New Decision →
   Patch Summary. Describe the patch; do **not** rewrite the architecture in
   place.
7. **§8 Updated ADRs / Decision Register** — promote accepted ADR candidates;
   supersede prior ADRs (never silently overwrite).
8. **§9 Updated Handoffs** — which §24 / §27 handoffs change.
9. **§10 Compatibility Check** — verify the invariants are preserved (blueprint
   thesis, Product Experience Direction, state model, interface contracts,
   security boundaries, observability, Recommended Next Stages). Flag any that the
   accepted decision would weaken.
10. **§12 Feedback Closure Matrix** — if this update applies feedback from a
    downstream artifact (ux-design Architecture Feedback, security-review findings,
    reconciliation), produce a Feedback Closure Matrix: Feedback Item · Source ·
    Closed By (patch ID) · Status (RESOLVED / PARTIAL / OPEN). If not applicable,
    mark `NOT_APPLICABLE`.

## Output

`intermediate/update_applied.md` (the §4–§12 content, using new section numbers).

## Validation / failure policy

- Gate: only accepted decisions are applied; Patch Manifest produced (or WARNING
  if cannot be determined); patch types classified; changed sections are listed;
  unaffected sections are preserved; invariants are checked; ux-design is not used
  as a direct source; Feedback Closure Matrix present when applicable.
- Failure policy: `revise` (or `stop` if the only source is unaccepted/speculative).
