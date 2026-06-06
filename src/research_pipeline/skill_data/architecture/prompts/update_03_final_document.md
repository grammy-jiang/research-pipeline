# Prompt update_03 — Final Update Document

You are assembling and writing the final `update` mode document. By default,
write only the **update note** — do not overwrite the architecture design.

## Inputs

- `intermediate/update_applied.md`, `intermediate/resolved_artifacts.json`.
- `templates/architecture_update_template.md`,
  `references/architecture-update-guide.md`.

## Instructions

1. Compose all 11 sections in order, starting with `## Contents`. Fill §1
   Generation Metadata (source architecture, update sources, skill version, and
   the Update History row to append to the architecture); do not invent metadata.
2. Embed the **Resolved Input Artifacts** table in §3.
3. **Run the §11 Update Quality-Gate Self-Check** and embed it. Gates:
   source-architecture-found, update-source-found, only-accepted-decisions-applied,
   changed-sections-listed, unaffected-sections-preserved, ADRs/decision-register-
   updated, update-history-updated, downstream-handoffs-still-valid. PASS /
   WARNING / FAIL; never PASS over a known gap.
4. **Fail conditions** (→ revise, max 3 then surface and stop): no architecture
   document; no update source; speculative-only source; ux findings applied
   directly without reconciliation; an overwrite of major decisions without
   evidence; no changed sections listed.
5. **Output discipline (no silent mutation):** write
   `<topic-slug>-architecture-update.md` by default. Optionally also write a
   proposed `<topic-slug>-architecture-design.updated.md`. **Do not overwrite**
   `<topic-slug>-architecture-design.md` unless the user explicitly asked, the
   change set is listed, the architecture's Update History row is appended, and
   the previous architecture is recoverable. If files cannot be written, output
   the Markdown inline and state the recommended filename(s).
6. End by stating whether anything else should run (e.g. re-`review` after the
   update, or `implementation-plan`).

## Output

`<topic-slug>-architecture-update.md` (+ optional `…-architecture-design.updated.md`).

## Validation / failure policy

- Gate: all 11 sections present with Contents; the §11 self-check passes; the
  architecture design is not overwritten by default.
- Failure policy: `revise_max_3_then_stop`.
