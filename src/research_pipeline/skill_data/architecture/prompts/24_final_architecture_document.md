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
4. **Enforce metadata consistency before writing:** the §1 metadata
   `Clarification count` equals the §3 row count; `Assumptions made` equals the
   §4.9 row count; every `A-N`, ADR, Contents, and section reference resolves;
   no metadata is invented. Fix any mismatch (do not deliver inconsistent
   metadata).
5. **Label proposed namespaces, not file paths:** if §25 lists package/module
   names, introduce them as "proposed module namespaces for implementation
   planning, not mandatory file-by-file implementation tasks." Do not emit task
   tickets, code patches, migration scripts, or file-by-file steps — those
   belong to the implementation-plan skill.
5a. **Cap build sequencing:** §25 may include at most **five** high-level
   sequencing constraints (e.g. "build the deterministic spine before AI
   adapters", "freeze core contracts first", "implement audit before state
   transitions"). A detailed build plan, file-by-file order, PR sequence, or
   class/migration ordering must be removed or deferred to the implementation-plan
   skill.
5b. **Surface warnings:** ensure every §24 WARNING / PASS-with-warning row is
   echoed in §1 and §25 with its required action and blocking status.
6. Write the document to `<topic-slug>-architecture-design.md`, co-located with
   the blueprint unless another output directory was specified. Write ADR files
   under `adr/`. In `compare` mode, emit the diff review instead of overwriting.
7. If files cannot be written, output the full Markdown inline and state the
   recommended filename(s).
8. End by pointing at the implementation-plan skill as the next stage (§25
   handoff notes).

## Output

`<topic-slug>-architecture-design.md` (+ `adr/ADR-*.md`).

## Validation / failure policy

- Gate: architecture quality gates pass and Update History is present.
- Failure policy: `revise_max_3_then_stop`.
