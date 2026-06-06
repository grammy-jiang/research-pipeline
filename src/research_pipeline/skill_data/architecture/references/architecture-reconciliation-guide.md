# Architecture Reconciliation Guide (`reconcile` mode)

Load this in `reconcile` mode. Reconcile **compares the architecture with
downstream artifacts and detects gaps, conflicts, or missing architecture
support**. It is conflict-driven and runs after a downstream artifact exists
(ux-design, test-design, security-review, implementation-plan feedback).

## Core discipline (no default patch)

- **Default output is a reconciliation note**,
  `<topic-slug>-architecture-reconciliation.md`. Reconcile **does not patch the
  architecture by default** — it produces findings and recommended changes, then
  hands off to `update` mode for the *accepted* ones.
- **Do not blindly accept the downstream artifact.** A downstream finding may
  conflict with the blueprint; if so, flag it as a conflict to resolve, not an
  automatic architecture change.
- **Recommend minimal changes** — the smallest architecture change that closes
  each gap.

## Default feedback-source priority

```text
1. UX design with an "Architecture Feedback / Required Architecture Updates" section.
2. Security review with architecture-impacting findings.
3. Test design with impossible or uncovered scenarios.
4. Implementation-plan with architecture blockers.
```

If a ux-design exists, make it the **primary** source. If multiple same-topic
feedback docs exist, include all relevant ones.

## What reconcile should detect

```text
missing user-visible state / internal state / transition
missing retry / cancellation operation
missing progress event
missing review-artifact schema
missing audit event
missing permission boundary
missing data-egress confirmation
missing API / CLI / MCP output field
UX flow incompatible with the state machine
E2E scenario impossible with current contracts
security control impossible with current data flow
implementation task blocked by a missing architecture decision
```

## Findings table

```markdown
| Finding | Source Artifact | Severity | Architecture Gap | Recommended Change | Requires Update Mode? |
|---|---|---|---|---|---|
| ... | ux-design | Blocking / Warning / Polish | ... | ... | yes/no |
```

Severity: **Blocking** prevents implementation planning or safe release;
**Warning** should be fixed but does not block; **Polish** improves clarity /
maintainability. **Separate conflicts from enhancements.**

## Missing Architecture Support

```markdown
| UX / Test / Security Need | Missing Architecture Support | Affected Section | Suggested Fix |
|---|---|---|---|
| User needs retry after provider failure | No retry state/operation defined | State Model, Interface Contracts | Add retry_allowed state + retry operation |
```

## Architecture Update Required?

End with an explicit verdict and the handoff:

```markdown
| Update Required | Reason | Recommended Next Command |
|---|---|---|
| Yes / No | ... | architecture --mode update |
```

If Yes, `architecture --mode update` should consume this reconciliation document
(its *accepted* recommendations become an update source).

## Reconciliation quality gate

```markdown
| Gate | Status | Finding | Required Action |
|---|---|---|---|
| Source architecture found | PASS / FAIL | ... | ... |
| Feedback document found | PASS / FAIL | ... | ... |
| Findings traceable to downstream artifacts | PASS / WARNING / FAIL | ... | ... |
| Conflicts separated from enhancements | PASS / WARNING / FAIL | ... | ... |
| Recommended changes are minimal | PASS / WARNING / FAIL | ... | ... |
| Architecture update requirement explicit | PASS / WARNING / FAIL | ... | ... |
| Downstream artifact not blindly accepted | PASS / WARNING / FAIL | ... | ... |
```

Fail if: no architecture document exists; no feedback artifact exists; a finding
cannot be mapped to an architecture section; the mode tries to silently rewrite
the architecture; or a downstream artifact contradicts the blueprint and is
accepted without warning.
