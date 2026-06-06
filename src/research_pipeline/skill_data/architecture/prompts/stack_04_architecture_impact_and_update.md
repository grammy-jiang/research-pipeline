# Prompt stack_04 — Architecture Impact and Update Verdict

You are stating how the selected stack affects the architecture and whether an
architecture update is required. Stack mode **declares** impact; it never
rewrites the architecture.

## Inputs

- `intermediate/stack_selection.md`, `intermediate/stack_decision_drivers.md`.
- The architecture design document.
- `references/tech-stack-selection-guide.md`.

## Instructions

1. Write §16 ADR Candidates — propose ADRs for high-impact stack decisions
   (status `Proposed`); note that the design `update` mode promotes them into the
   architecture ADR register and handles supersession. Do not finalize ADRs here.
2. Write §17 Architecture Impact Notes — per selected technology, the
   architecture sections it touches and how. Keep these as notes, not a rewrite.
3. Write §18 **Architecture Update Required?** — the explicit verdict table:
   Update Needed? · Affected Architecture Sections · Reason · Priority. Use the
   trigger examples in the guide (embedded store → audit/concurrency/deployment;
   server DB → deployment/backup/permissions/ops; LLM gateway → provider
   abstraction/observability/logging/dependency-risk; MCP SDK → integration
   surface/security boundary).
4. If a chosen technology *conflicts* with an architecture constraint, set
   Update Needed = Yes with the reason — do **not** silently edit the
   architecture. If nothing changes, set Update Needed = No and say why.

## Output

`intermediate/stack_architecture_impact.md` (the §16–§18 content).

## Validation / failure policy

- Gate: architecture impact notes exist and the Architecture Update Required?
  verdict is explicit (Yes/No with affected sections + reason).
- Failure policy: `revise`.
