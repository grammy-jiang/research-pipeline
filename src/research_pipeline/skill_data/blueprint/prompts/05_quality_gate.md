# Prompt 05 — Quality Gate & Post-Generation Review

Review the generated product blueprint against the gates below, then emit
the self-check appendix (last section). Gates are checked in order; earlier
gates may be preconditions for later ones. List only genuine failures — do
not flag style issues.

A gate is `FAIL` (must fix before delivery), `WARNING` (deliver but surface
it in the self-check), or `PASS`.

**Maximum revision attempts: 3.** After 3 failed revisions, surface all
failing gates to the user and stop. Do not deliver an unvalidated
blueprint.

## Gate 1 — Input understanding & metadata integrity

- The blueprint names the source report's main research question.
- It acknowledges the input quality level and any missing sections.
- If multi-domain, the targeted domain is documented.
- **Metadata integrity:** no metadata value is invented, normalised, or
  upgraded. The blueprint skill version is copied from `manifest.json` or
  is `unknown` (never a fabricated number). **Pipeline runs integrated**
  and **gap-closure rounds** are separate fields — flag a `WARNING` if a
  pipeline-run count has been relabelled as a round count. Unavailable
  fields read `unknown`.

## Gate 2 — Research-to-product traceability & source fidelity

- Every major capability traces to a research citation (`[arxiv_id]` or
  `[Author, Year]`) or a constrained explicit design decision with
  rationale.
- Classify each major product claim as: **research-backed** (cites the
  report), **engineering extrapolation** (cites an engineering gap or
  recommendation), **product design decision** (connects / operationalizes
  / governs research-backed capabilities, with rationale), **speculative**,
  or **unsupported**.
  - Speculative claims must move to Open Questions (§16) or Future
    Extensions (§15).
  - Unsupported claims must be removed or marked "Design hypothesis —
    requires validation."
  - A product decision must not replace research evidence for a core claim.

## Gate 3 — Implementation neutrality (with warning tier)

Classify implementation-leaning wording using `references/borderline-cases.md`:

- **Forbidden** → `FAIL`: programming language, framework, database
  (incl. specific products), cloud provider, vendor service, package/module
  structure, deployment commands, code, or implementation tickets.
- **Warning** → `WARNING`: runtime/architecture-leaning phrasing
  (e.g. "service-deployable", "deployed as microservices", a specific
  context-window assumption). Rephrase to its purpose or defer to §17,
  unless it is a cited research-derived evaluation anchor.
- **Research-derived exception** → keep only if cited and tied to
  evaluation (e.g. a named decoding/caching technique from a paper).

## Gate 4 — Workflow completeness

Each major workflow must include all of: trigger, inputs, decision gates,
steps (or a Mermaid flow), outputs, failure modes, and success criteria.

## Gate 5 — Scope control & MVP discipline

- **Scope:** every primary actor and MVP domain is named (or clearly
  implied) by the product thesis. High-stakes/adjacent domains that appear
  only as research evidence must be Secondary/Future, not primary — flag a
  `WARNING` if such a domain appears as a primary actor or MVP requirement.
- **MVP structure:** §14 separates Core Value Path from Safety Baseline and
  Evaluation Baseline, and has an explicit pass/fail success definition.
- The **core value path** is minimal and proves the thesis; safety and
  evaluation baselines are justified separately (they do not count against
  the core-path size).
- `ACADEMIC`-gap items are not in MVP unless the product validates that gap.
- Flag a `WARNING` (not `FAIL`) when the core value path carries more than
  6 capabilities without justification, or when `standard` output exceeds
  the §04 length budgets. `FAIL` only if the MVP no longer represents a
  small, testable core value path.

## Gate 6 — Risk honesty

- HIGH-impact risks are explicit; mitigations are realistic (never "prompt
  the model better"). Open risks are not hidden. Safety-critical deferred
  items are release gates. Risks from unvalidated `ACADEMIC` items are
  flagged.

## Gate 7 — Downstream usefulness

- A technical-design agent can choose a tech stack and plan an
  implementation without re-reading the papers.
- The `## Contents` section exists and all section links are valid.
- A Mermaid diagram exists for both the main end-to-end workflow and the
  logical architecture.

## Immediate-fail conditions

- Any tech-stack choice, code, or implementation ticket.
- Either required Mermaid diagram missing.
- The Contents section absent.
- Open research gaps silently treated as solved.
- Risks omitted.
- Logical architecture replaced with technical architecture.
- Handoff to technical design missing.
- Invented metadata (e.g. a fabricated skill version).
- `ACADEMIC`-gap items appearing as MVP requirements without explicit
  justification.

## For each failure, state

- Gate name and number.
- Specific location in the document (section and paragraph).
- The required fix.

Then revise the failing sections and re-run the gates (bounded to 3
attempts total).

## Self-check output

After the gates pass (no `FAIL` remaining), emit `## Appendix A: Blueprint
Quality-Gate Self-Check` in the blueprint: a compact table marking each
gate `PASS` / `WARNING` / `FAIL` with a one-line note. Every `WARNING`
raised above must appear there — do not hide it. Example notes:

```text
MVP discipline: WARNING — core path has 7 capabilities; safety/eval
  baselines listed separately and justified.
Metadata integrity: WARNING — report stated "15 pipeline runs"; rounds
  recorded as unknown.
Implementation neutrality: WARNING — "service-deployable" deferred to §17.
```
