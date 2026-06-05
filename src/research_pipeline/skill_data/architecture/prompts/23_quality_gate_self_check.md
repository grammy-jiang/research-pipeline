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

## v0.2.0 quality-control gates (also evaluate these)

1. **Metadata consistency** — FAIL if the §1 generation-metadata counts do not
   match the document: `Clarification count` must equal the number of §3
   rows; `Assumptions made` must equal the number of §4.9 assumption rows;
   every `A-N` reference must resolve to an assumption row (no missing/duplicate
   IDs); every ADR reference must resolve to a §21 entry; every Contents link
   and section reference must resolve; no metadata field is invented (use
   `unknown`).
2. **Hybrid decision review** — when operating mode is hybrid, FAIL if any major
   §3 decision lacks a Source and a Review Requirement, or if a high-impact
   inferred/assumed decision (external LLM use, data privacy, deployment,
   storage, auth, retention, cost routing, MCP, human approval) is not
   review-flagged. WARN if high-impact decisions are reviewable but proceeding.
3. **Technology-specific validity** — FAIL if the architecture credits a chosen
   technology with enforcement/security properties it does not provide; absolute
   wording must be downgraded to application-enforced / tamper-evident /
   best-effort / requires-operational-control, with a risk or ADR note when
   enforcement depends on implementation discipline.
4. **Probe / evaluator availability** — FAIL if a model-backed evaluator or
   gating probe lacks an availability policy (required level, behavior if
   unavailable, whether auto-accept/gate-pass is still allowed, audit event);
   required/release-gate probes must disable auto-accept when unavailable.
   PASS as n/a if the system has no model-backed evaluators.
5. **Architecture-vs-implementation boundary** — FAIL if the document contains
   task tickets, code patches, migration scripts, or file-by-file
   implementation steps. WARN if proposed package/module names are presented as
   final file paths rather than labelled "proposed module namespaces".

## Instructions

1. Evaluate each gate (the FAIL list above and the five v0.2.0 gates) against the
   draft; record PASS / WARNING / FAIL with a finding, a required action, and a
   blocks-implementation verdict.
2. Emit the §24 table (Gate · Status · Finding · Required Action · Blocks
   Implementation?), including rows for: Metadata consistency, Hybrid decision
   review, Technology-specific validity, Probe/evaluator availability, and
   Architecture-vs-implementation boundary.
3. If any gate FAILs, return the specific failing gates so prompt 24 can revise.
   Every WARNING must carry a concrete required action, not a passive note.

## Output

`intermediate/quality_gate_self_check.md` → populates §24.

## Validation / failure policy

- Gate: no failed quality gates.
- Failure policy: `revise_max_3_then_stop`.
