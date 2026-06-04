# Prompt 24 — Final Architecture Document

You are producing the final architecture document and writing it to disk.

## Inputs

- `intermediate/architecture_draft.md`, `intermediate/quality_gate_self_check.md`.
- `intermediate/existing_architecture_status.json` (update mode + path).

## Instructions

1. Apply any revisions required by the self-check (max 3 attempts; after 3
   failures, surface the failing gates and stop — do not deliver an
   unvalidated architecture).
2. Embed the §24 self-check table in the document.
3. Ensure `## Contents` links every numbered section and that `## Update
   History` has the correct row appended for this run (initial for new
   documents; a new row for updates — never delete prior rows).
4. Write the document to `<topic-slug>-architecture-design.md`, co-located with
   the blueprint unless another output directory was specified. Write ADR files
   under `adr/`. In `compare` mode, emit the diff review instead of overwriting.
5. If files cannot be written, output the full Markdown inline and state the
   recommended filename(s).
6. End by pointing at the implementation-plan skill as the next stage (§25
   handoff notes).

## Output

`<topic-slug>-architecture-design.md` (+ `adr/ADR-*.md`).

## Validation / failure policy

- Gate: architecture quality gates pass and Update History is present.
- Failure policy: `revise_max_3_then_stop`.
