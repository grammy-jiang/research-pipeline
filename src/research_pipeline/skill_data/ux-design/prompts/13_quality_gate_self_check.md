# Prompt 13 — UX Quality-Gate Self-Check

You are auditing the UX design against the quality gates and emitting the
actionable Appendix A self-check. Be skeptical: use PASS / WARNING / FAIL and
never mark a clean PASS over a known gap.

## Inputs

- `<topic-slug>-ux-design.md` (the composed document), all `intermediate/*`.
- `tests/expected_sections_checklist.md`,
  `tests/forbidden_content_checklist.md`.

## Gates — evaluate each

Emit the Appendix A table:

```markdown
| Gate | Status | Finding | Required Action | Blocks Implementation Planning? |
|---|---|---|---|---|
| Source architecture consumed | PASS / WARNING / FAIL | ... | ... | yes/no |
| Product Experience Direction preserved | PASS / WARNING / FAIL | ... | ... | yes/no |
| Skill Operator UX defined | PASS / WARNING / FAIL | ... | ... | yes/no |
| Target Software UX defined | PASS / WARNING / FAIL | ... | ... | yes/no |
| User stories defined | PASS / WARNING / FAIL | ... | ... | yes/no |
| Failure/recovery flows defined | PASS / WARNING / FAIL | ... | ... | yes/no |
| Human-review UX defined where needed | PASS / WARNING / FAIL | ... | ... | yes/no |
| E2E scenario seeds generated | PASS / WARNING / FAIL | ... | ... | yes/no |
| Architecture feedback section present | PASS / WARNING / FAIL | ... | ... | yes/no |
```

## Fail conditions (any → FAIL)

```text
Architecture document is missing.
Target users are undefined. / Primary surface is undefined.
No user stories exist. / Failure/recovery flows are missing.
Human review exists in the architecture but no human-review UX is defined.
E2E scenario seeds are missing. / Architecture feedback section is absent.
```

## Warning conditions (→ WARNING)

```text
Too many surfaces are included for MVP.
CLI-first selected for non-technical users without mitigation.
MCP exposed without a clear agent user.
Web UI deferred despite frequent human review.
Acceptance criteria are vague. / E2E scenarios are not testable.
Architecture feedback is incomplete.
```

## Out-of-scope scan (→ FAIL)

```text
Executable tests (step definitions / assertions in code / fixtures).
Architecture or tech-stack decisions re-made here.
Pixel-level layout, CSS, final screen copy, or exact CLI flags.
```

## Instructions

1. Evaluate every gate; record PASS / WARNING / FAIL with a finding, a required
   action, and a blocks-implementation-planning verdict.
2. **Status discipline:** PASS only when complete and consistent; WARNING when
   the direction is acceptable but needs cleanup (non-blocking, but carries a
   concrete required action); FAIL for a missing required element, an
   out-of-scope output, or anything that would mislead implementation planning.
   Never PASS over a known gap.
3. If any gate FAILs, return the specific failing gates so prompt 12 can revise
   (max 3 attempts, then surface the failing gates and stop — do not deliver an
   unvalidated UX design).

## Output

`intermediate/quality_gate_self_check.md` → populates Appendix A.

## Validation / failure policy

- Gate: no failed UX quality gates (or failures surfaced after 3 attempts).
- Failure policy: `revise_max_3_then_stop`.
