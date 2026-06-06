# Prompt review_02 — Assess and Score

You are evaluating an existing architecture's quality. `review` mode is
**non-mutating** — it never edits the architecture.

## Inputs

- `intermediate/resolved_artifacts.json` (architecture design + optional
  siblings).
- The architecture design document and any present optional artifacts
  (blueprint, tech-stack, ux-design, security-review, test-design).
- `references/architecture-review-guide.md`,
  `templates/architecture_review_template.md`.

## Instructions

1. **Read only what was resolved.** Score only content actually present; do not
   claim to have verified a missing artifact. Missing optional docs scope the
   score (note them) but never fail review.
2. **Score the 15 dimensions** (§4) on a 10-point scale, each with a one-line
   justification — a score without a reason is a gate failure. Compute the
   overall score.
3. **Write the per-area assessments** (§5–§14): blueprint fidelity, Product
   Experience Direction preservation, Recommended Next Stages consumption, system
   quality, state/contract/data model, security/egress, observability/audit, tech
   stack consistency, UX-design readiness, implementation-plan readiness. Mark
   sections n/a where the relevant artifact is absent (e.g. no tech-stack → §12
   n/a).
4. **Classify every finding** into exactly one bucket — never mix blocking with
   polish:
   - §15 **Blocking** (missing state model / contracts / security boundary /
     data-egress policy; contradicts blueprint; ignores Product Experience
     Direction);
   - §16 **Warning** (provisional stack; incomplete UX handoff; underspecified
     failure paths; generic observability);
   - §17 **Polish** (section order, wording, formatting, minor traceability).
5. **Recommend next actions** (§18): an ordered list tied to the findings —
   which of `update` / `reconcile` / `stack` / `ux-design` /
   `implementation-plan` should run next and why.

## Output

`intermediate/review_assessment.md` (the §3–§18 content).

## Validation / failure policy

- Gate: every score is justified; findings are classified into
  blocking/warning/polish; next actions are clear; nothing claims to verify an
  unread artifact.
- Failure policy: `revise`.
