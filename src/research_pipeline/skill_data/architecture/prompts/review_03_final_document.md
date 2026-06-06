# Prompt review_03 — Final Review Document

You are assembling and writing the final `review` mode document. Review is
**non-mutating**: write only `<topic-slug>-architecture-review.md` — never touch
the architecture design document.

## Inputs

- `intermediate/review_assessment.md`, `intermediate/resolved_artifacts.json`.
- `templates/architecture_review_template.md`,
  `references/architecture-review-guide.md`.

## Instructions

1. Compose all 19 sections in order, starting with `## Contents`. Fill §1 Review
   Metadata (reviewed architecture, version, skill version, overall score,
   implementation-plan-ready verdict); do not invent metadata (use `unknown`).
2. Embed the **Resolved Input Artifacts** table in §2 Documents Reviewed so the
   reader sees exactly what was read.
3. **Run the §19 Review Quality-Gate Self-Check** and embed it. Gates:
   review-is-non-mutating, source-architecture-found,
   optional-artifacts-handled-correctly, scores-justified, issues-classified,
   recommended-next-actions-clear. Be skeptical: PASS / WARNING / FAIL, never a
   clean PASS over a known gap.
4. **Fail conditions** (→ revise, max 3 then surface and stop): no architecture
   design found; the review claims to verify artifacts it did not read; the
   review rewrites architecture; scores without reasons; blocking issues mixed
   with polish.
5. Write to `<topic-slug>-architecture-review.md`, co-located with the
   architecture unless told otherwise. If files cannot be written, output the
   full Markdown inline and state the recommended filename. **Do not edit, patch,
   or overwrite** the architecture design document.
6. End by pointing at the §18 recommended next actions (e.g. `architecture
   --mode update` / `--mode reconcile`, or `implementation-plan`).

## Output

`<topic-slug>-architecture-review.md`.

## Validation / failure policy

- Gate: all 19 sections present with Contents; the §19 self-check passes; the
  architecture design document is unmodified.
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
