# Prompt reconcile_03 — Final Reconciliation Document

You are assembling and writing the final `reconcile` mode document. Reconcile
does **not** patch the architecture — write only the reconciliation note and hand
off accepted changes to `update` mode.

## Inputs

- `intermediate/reconcile_findings.md`, `intermediate/resolved_artifacts.json`.
- `templates/architecture_reconciliation_template.md`,
  `references/architecture-reconciliation-guide.md`.

## Instructions

1. Compose all 11 sections in order, starting with `## Contents`. Fill §1
   Generation Metadata (source architecture, feedback sources, skill version, the
   Architecture Update Required? verdict); do not invent metadata.
2. Embed the **Resolved Input Artifacts** table in §3.
3. **§9 Architecture Update Required?** — the explicit verdict table (Update
   Required · Reason · `architecture --mode update`). **§10 Handoff** — if Yes,
   state that `update` mode should consume this document and which recommendations
   are accepted (an update source) vs open (need a user decision first).
4. **Run the §11 Reconciliation Quality-Gate Self-Check** and embed it. Gates:
   source-architecture-found, feedback-document-found, findings-traceable,
   conflicts-separated-from-enhancements, recommended-changes-minimal,
   update-requirement-explicit, downstream-not-blindly-accepted. PASS / WARNING /
   FAIL; never PASS over a known gap.
5. **Fail conditions** (→ revise, max 3 then surface and stop): no architecture
   document; no feedback artifact; a finding that cannot be mapped to an
   architecture section; any attempt to silently rewrite the architecture; a
   downstream artifact that contradicts the blueprint accepted without a warning.
6. **Output discipline (no default patch):** write
   `<topic-slug>-architecture-reconciliation.md` only. Do **not** patch or
   overwrite the architecture design. If files cannot be written, output the
   Markdown inline and state the recommended filename.
7. End by pointing at `architecture --mode update` (if Update Required = Yes) or
   stating no architecture change is needed.

## Output

`<topic-slug>-architecture-reconciliation.md`.

## Validation / failure policy

- Gate: all 11 sections present with Contents; the §11 self-check passes; the
  architecture design is not patched.
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
