# Architecture Review Guide (`review` mode)

Load this in `review` mode. Review **evaluates architecture quality without
changing it**. It answers: is this architecture good enough? what is missing?
what blocks implementation planning? what should be improved? what should run
next (update / reconcile / stack / ux-design)?

## Core discipline

- **Non-mutating.** Review never rewrites, patches, or overwrites the
  architecture. It produces a separate `<topic-slug>-architecture-review.md`.
- **Read what you score.** Only credit or fault content you actually read; never
  claim to have verified an artifact that was missing.
- **Default to review.** When the user invokes bare `architecture` and an
  architecture already exists, `review` is the safe default (not `update`).

## Default input behaviour

- **Required:** the latest `<topic-slug>-architecture-design.md`.
- **Optional (use only if present; missing never fails review):** matching
  blueprint, tech-stack, ux-design, architecture-update, architecture-
  reconciliation, security-review, test-design.

## Score breakdown (10-point scale)

Score these dimensions, each with a one-line justification (a score with no
reason is a gate failure):

```text
Blueprint fidelity
Product Experience Direction preservation
Recommended Next Stages consumption
Experience Architecture quality
System decomposition
State model
Interface contracts
Data contracts
Security / trust boundaries
Data egress / privacy
Observability / audit
Failure / recovery
Tech stack separation
UX-design readiness
Implementation-plan readiness
```

Table form: `| Area | Score | Comment |` (e.g. `Blueprint fidelity | 9.2 |
Preserves product thesis and MVP split`).

## Issue classification

Separate every finding into exactly one of three buckets — never mix blocking
issues with polish:

- **Blocking** — missing state model; missing interface contracts; no security
  boundary; no data-egress policy for external LLM use; architecture contradicts
  the blueprint; architecture ignores the Product Experience Direction.
- **Warning** — tech stack still provisional; UX handoff incomplete; some failure
  paths underspecified; observability event list too generic.
- **Polish** — section order; wording; table formatting; minor traceability
  improvements.

## Recommended next actions

End with a clear, ordered list: which of `update` / `reconcile` / `stack` /
`ux-design` / `implementation-plan` should run next and why, tied to the
blocking/warning findings.

## Review quality gate

```markdown
| Gate | Status | Finding | Required Action |
|---|---|---|---|
| Review is non-mutating | PASS / FAIL | ... | ... |
| Source architecture found | PASS / FAIL | ... | ... |
| Optional artifacts handled correctly | PASS / WARNING / FAIL | ... | ... |
| Scores are justified | PASS / WARNING / FAIL | ... | ... |
| Issues are classified | PASS / WARNING / FAIL | ... | ... |
| Recommended next actions are clear | PASS / WARNING / FAIL | ... | ... |
```

Fail if: no architecture design document is found; the review claims to verify
artifacts it did not read; the review silently rewrites architecture; scores are
given without reasons; or blocking issues are mixed with polish comments.
