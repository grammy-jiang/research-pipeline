# Prompt 23 — Quality-Gate Self-Check

You are auditing the draft against the architecture quality gates and emitting
the actionable §24 self-check.

## Inputs

- `intermediate/architecture_draft.md`, `intermediate/rule_pack_review.md`.
- `tests/expected_sections_checklist.md`, the other `tests/*_checklist.md`.

## Gates — the output FAILS if any hold

```text
[ ] No SKILL.md trigger/use contract / manifest does not cover major passes.
[ ] No blueprint-to-architecture traceability map.
[ ] No generation metadata. / No Contents section. / No Update History.
[ ] No tech-stack rationale. / No architecture goals and constraints.
[ ] No traditional-vs-AI responsibility matrix.
[ ] No C4 System Context / Container / main-workflow Dynamic view.
[ ] No module/container ownership.
[ ] No interface contracts. / No data contracts.
[ ] No security/trust boundary model. / No observability/audit plan.
[ ] No failure handling. / No ADRs for major decisions.
[ ] MCP introduced without justification.
[ ] AI components can mutate durable state without deterministic validation.
[ ] MVP-0 / MVP-1 from blueprint ignored.
[ ] Every conceptual component converted to a service without rationale.
[ ] Existing-architecture update behavior undefined when a file already exists.
[ ] Handoff notes for implementation planning are missing.
```

## Instructions

1. Evaluate each gate against the draft; record PASS / WARN / FAIL with a
   finding, a required action, and a blocks-implementation verdict.
2. Emit the §24 table (Gate · Status · Finding · Required Action · Blocks
   Implementation?).
3. If any gate FAILs, return the specific failing gates so prompt 24 can revise.

## Output

`intermediate/quality_gate_self_check.md` → populates §24.

## Validation / failure policy

- Gate: no failed quality gates.
- Failure policy: `revise_max_3_then_stop`.
