# Prompt stack_06 — Final Tech-Stack Document

You are assembling and writing the final `stack` mode document.

## Inputs

- `intermediate/stack_decision_drivers.md`, `intermediate/stack_selection.md`,
  `intermediate/stack_architecture_impact.md`,
  `intermediate/stack_quality_gate.md`, `intermediate/stack_inputs.json`.
- `templates/architecture_tech_stack_template.md`.

## Instructions

1. Apply any revisions required by the §19 self-check (max 3 attempts; after 3
   failures, surface the failing gates and stop — do not deliver an unvalidated
   document).
2. Assemble all 19 sections from the template, in order, with `## Contents` and
   `## Update History`. Fill §1 Generation Metadata from
   `intermediate/stack_inputs.json` and the manifest version; do not invent
   metadata (use `unknown`).
3. **Metadata consistency:** `Decisions made` equals the §4 row count; every
   architecture-section reference (`§N`) resolves in the source architecture.
4. Append the correct Update History row (initial for a new document; a new row
   for updates — never delete prior rows). In `compare` mode, emit the diff
   instead of overwriting.
5. Echo the §18 Architecture Update Required? verdict near the top so the reader
   sees immediately whether `update` mode must run next.
6. Write to `<topic-slug>-architecture-tech-stack.md`, co-located with the
   architecture design unless another output directory was specified. If files
   cannot be written, output the full Markdown inline and state the recommended
   filename.
7. End by pointing at the next stage: if Architecture Update Required = Yes, run
   `architecture --mode update`; otherwise continue to ux-design / implementation
   planning per the blueprint's §19 routing.

## Output

`<topic-slug>-architecture-tech-stack.md`.

## Validation / failure policy

- Gate: stack quality gates pass and the Architecture Update Required? verdict is
  present; Update History is present.
- Failure policy: `revise_max_3_then_stop`.

## Cross-Skill Artifact Contract Compliance

Comply with the Cross-Skill Artifact Contract (`references/artifact-contract.md`).
The output document must expose the contract fields using the controlled
vocabulary:

- **Generation Metadata** including `Artifact Type` (a registry value) and a
  stable `Topic Slug` (carried unchanged across the pipeline).
- **Source Artifacts Consumed** (what was read and how it was used).
- **Resolved Input Artifacts** when inputs were auto-discovered (else
  `NOT_APPLICABLE — all input artifacts were explicitly supplied by the user`).
- A **decision register** (controlled status values), **assumptions** kept
  separate from decisions, **open questions** assigned to a next stage, and a
  **Recommended Next Stage** (RUN / SKIP / DEFER / ASK_USER).
- A **Quality-Gate Self-Check** that includes the **Cross-Skill Artifact
  Contract Gate**.

If a section already exists under this skill's own heading, align it to the
contract (a Contract Field Map is fine) rather than duplicating. Mark any
not-applicable field `NOT_APPLICABLE — <reason>`; never omit it.
