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
   implementation steps. WARNING if proposed package/module names are presented
   as final file paths rather than labelled "proposed module namespaces".

## v0.3.0 hardening gates (also evaluate these)

6. **Residual invalid-claim scan** — scan the *whole* document (not just the
   tech-stack section): §17 security gates, §13/§14 data, §24 checklist rows,
   ADRs. FAIL/WARNING on any technology-inconsistent claim — e.g. borrowing one
   technology's permission/enforcement/immutability/provider-neutrality wording
   for a different chosen technology (an embedded file-based store described
   with role/grant "DB user" wording; local filesystem called "immutable"; a
   provider wrapper called "provider-independent"). Rewrite to the honest
   mechanism (application-enforced / tamper-evident / content-addressed +
   integrity-checked / provider-abstraction).
7. **Data egress / external model use** — FAIL if the architecture uses any
   external model/provider but has no separate §3 data-egress decision
   (external_allowed / …_with_redaction / local_only / hybrid_by_domain /
   unknown_requires_user_review). It must not be merged into the
   provider-abstraction decision; if unanswered it is "review before
   implementation planning". PASS as n/a only if nothing leaves the local
   boundary.
8. **State-semantics consistency** — FAIL if a status/state/condition term is
   used (API, failure handling, observability, human-review, probe policy) but
   does not resolve to the canonical §14 state model; lifecycle states,
   operational condition flags, and audit events must stay distinct (e.g. do not
   use "degraded" as a lifecycle state unless it is in the lifecycle list).
9. **Standard-vs-detailed budget** — WARNING if `output_detail` is `standard`
   but the main body is dossier-sized (heavy schemas/ADR bodies/long matrices
   inline instead of in appendices). Required action: move heavy material to
   appendices or relabel `detailed`. Non-blocking.
10. **Security gate verification format** — FAIL if §17 security quality gates
    are written as ambiguous unchecked `- [ ]` checkboxes instead of a
    verification table (Security Gate · Required Implementation Evidence ·
    Verification Method · Blocks Release?). Checkbox gates are also the place
    residual invalid claims hide (gate 6) — convert them and rewrite any
    technology-inconsistent wording to the honest mechanism while doing so.
11. **Data-egress table** — when external models are used, FAIL if §17.9 lacks
    the dedicated Data Egress / External Model Use table (content-leaves-boundary,
    which providers, redaction, logs-may-contain-source, domain-plugin-override);
    `unknown_requires_user_review` blocks implementation planning.

When running gate 6 (residual invalid-claim scan), scan the §17 security gates,
§24 checklist rows, and ADRs character-by-character for borrowed-technology
wording — not just the §7 prose. If a gate is currently a checkbox, rewriting it
as a verification table (gate 10) is the moment to fix the wording.

## Self-check skepticism (status discipline)

Status values are **PASS / WARNING / FAIL** (WARNING ≡ "PASS with warning").

- Use **PASS** only when the section is complete *and* internally consistent.
- Use **WARNING** when the direction is acceptable but wording, assumptions,
  review flags, consistency, or budget need cleanup. A WARNING is not blocking
  but must carry a concrete required action and a blocks-implementation verdict.
- Use **FAIL** for a missing required section, an unsupported major decision, a
  false technology claim, a missing required security/observability control, or
  anything that would mislead a downstream implementation agent.
- **Do not mark a section PASS when a known contradiction or residual invalid
  claim exists** — downgrade it to WARNING (or FAIL) with the required action.
  Be skeptical of your own draft; the residual-claim scan exists precisely
  because earlier passes rubber-stamped PASS.

## Instructions

1. Evaluate every gate (the FAIL list, the five v0.2.0 gates, and the v0.3.0 /
   v0.4.0 gates) against the draft; record PASS / WARNING / FAIL with a finding,
   a required action, and a blocks-implementation verdict.
2. Emit the §24 table (Gate · Status · Finding · Required Action · Blocks
   Implementation?), including rows for: Metadata consistency, Hybrid decision
   review, Technology-specific validity, Probe/evaluator availability,
   Architecture-vs-implementation boundary, Residual invalid-claim scan, Data
   egress / external model use, State-semantics consistency,
   Standard-vs-detailed budget, and Security gate verification format.
3. If any gate FAILs, return the specific failing gates so prompt 24 can revise.
   Every WARNING must carry a concrete required action, not a passive note.

## Output

`intermediate/quality_gate_self_check.md` → populates §24.

## Validation / failure policy

- Gate: no failed quality gates.
- Failure policy: `revise_max_3_then_stop`.
