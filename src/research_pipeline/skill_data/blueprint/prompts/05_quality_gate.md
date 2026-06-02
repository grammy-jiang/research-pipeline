# Prompt 05 — Quality Gate

Review the generated product blueprint against the seven gates below.
Gates are checked in order; earlier gates may be preconditions for later
ones. List only genuine failures — do not flag style issues.

**Maximum revision attempts: 3.** After 3 failed revisions, surface all
failing gates to the user and stop. Do not deliver an unvalidated
blueprint.

## Gate 1 — Input understanding

- The blueprint names the source report's main research question.
- It acknowledges the input quality level and any missing sections.
- If multi-domain, the targeted domain is documented.

## Gate 2 — Research-to-product traceability

- Every major capability traces to a research citation (`[arxiv_id]` or
  `[Author, Year]`) or a constrained explicit design decision with
  rationale. Untraceable capabilities must be marked "Design hypothesis —
  requires validation."

## Gate 3 — Implementation neutrality

Fail if the blueprint includes any of: programming-language choice,
framework choice, database choice (including specific DB products), cloud
provider, vendor-specific service, package/module structure, deployment
commands, code, or implementation tickets. Use
`references/borderline-cases.md` when uncertain.

## Gate 4 — Workflow completeness

Each major workflow must include all of: trigger, inputs, decision gates,
steps (or a Mermaid flow), outputs, failure modes, and success criteria.

## Gate 5 — MVP discipline

- MVP is small enough to build and contains the core value path.
- MVP excludes advanced or speculative research extensions.
- Safety-critical baseline controls are included.
- MVP success is explicitly defined.
- `ACADEMIC`-gap items are not in MVP unless the product validates that gap.
- Flag (warning, not failure) an MVP with more than 6 capabilities lacking
  justification. Fail only if the MVP no longer represents a small,
  testable core value path.

## Gate 6 — Risk honesty

- HIGH-impact risks are explicit.
- Mitigations are realistic (never "prompt the model better").
- Open risks are not hidden.
- Safety-critical deferred items are flagged as release gates.
- Risks from unvalidated `ACADEMIC` items are flagged accordingly.

## Gate 7 — Downstream usefulness

- A later technical-design agent can choose a tech stack and produce an
  implementation plan without re-reading the original papers.
- The `## Contents` section exists and all section links are valid.
- At least one Mermaid diagram exists for the main end-to-end workflow.
- At least one Mermaid diagram exists for the logical architecture.

## Immediate-fail conditions

- Any tech-stack choice, code, or implementation ticket.
- Either required Mermaid diagram missing.
- The Contents section absent.
- Open research gaps silently treated as solved.
- Risks omitted.
- Logical architecture replaced with technical architecture.
- Handoff to technical design missing.
- `ACADEMIC`-gap items appearing as MVP requirements without explicit
  justification.

## For each failure, state

- Gate name and number.
- Specific location in the document (section and paragraph).
- The required fix.

Then revise the failing sections and re-run the gates (bounded to 3
attempts total).
