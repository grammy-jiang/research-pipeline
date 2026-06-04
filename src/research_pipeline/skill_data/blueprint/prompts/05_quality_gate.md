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
- **Thesis emphasis:** the thesis leads with the primary research-backed
  architecture; a conditional, bounded, escalation-only, or secondary
  mechanism is not promoted to product identity. Flag a `WARNING` and
  rewrite if it is (e.g. leading with "multi-agent" when the evidence says
  "backbone-first with conditional review").

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
- **No blank citations:** paper-derived rows cite `[arxiv_id]` /
  `[Author, Year]`; gap-derived rows cite
  `[Source Report: Research Gaps — <gap name>]`. A cell is blank only when
  the row is explicitly an internal design hypothesis.

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
- **MVP structure:** §14 splits the core path into **MVP-0** (smallest
  demonstrable end-to-end slice) and **MVP-1** (first usable version),
  separates Safety and Evaluation baselines, and has an explicit pass/fail
  success definition.
- **MVP-0** is minimal and proves the thesis; safety and evaluation
  baselines are justified separately (they do not count against MVP-0
  size). Flag a `WARNING` if a large Phase-1 system is labelled MVP-0, or
  if MVP-0/MVP-1 are split artificially for a trivial product.
- `ACADEMIC`-gap items are not in MVP unless the product validates that gap.
- Flag a `WARNING` (not `FAIL`) when MVP-0 carries more than 6 capabilities
  without justification, or when `standard` output exceeds the §04 length
  budgets. `FAIL` only if the MVP no longer represents a small, testable
  core value path.

## Gate 6 — Risk honesty

- HIGH-impact risks are explicit; mitigations are realistic (never "prompt
  the model better"). Open risks are not hidden. Safety-critical deferred
  items are release gates. Risks from unvalidated `ACADEMIC` items are
  flagged.
- **Release-gate confidence consistency:** a release gate derived from a
  MEDIUM/LOW-confidence mechanism is justified only if risk impact is HIGH,
  no cheaper baseline control exists, and the blueprint says why it is
  required now. Otherwise flag a `WARNING` and downgrade it to a warning,
  an evaluation/monitoring requirement, or a Phase 2 gate. Each release
  gate states its confidence and risk impact.

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
Quality-Gate Self-Check`: a compact table with columns **Gate · Status ·
Finding · Required Action · Blocks Technical Design?**. Mark each gate
`PASS` / `WARNING` / `FAIL`. Every `WARNING` raised above must appear with
a concrete required action and a yes/no blocks-technical-design verdict —
never a passive note. Example rows:

| Gate | Status | Finding | Required Action | Blocks TD? |
|---|---|---|---|---|
| Thesis emphasis | WARNING | Thesis leads with a conditional mechanism | Rewrite to lead with the primary architecture | No |
| MVP discipline | WARNING | MVP-0 still has 6 capabilities | Move routing + plugin architecture to MVP-1 | No — resolve before implementation planning |
| Release-gate confidence | WARNING | MEDIUM-confidence control set as default gate | Downgrade to monitoring at MVP-0; gate in Phase 2 | No |
