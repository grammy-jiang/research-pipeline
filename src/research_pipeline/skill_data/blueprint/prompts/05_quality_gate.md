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
  - Speculative claims must move to Open Questions (§17) or Future
    Extensions (§16).
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
  context-window assumption). Rephrase to its purpose or defer to §18,
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
- **MVP structure:** §15 splits the core path into **MVP-0** (smallest
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
- The `## Contents` section exists, all links are valid, and it lists every
  numbered section **and every appendix present** (Appendix A always;
  Appendix B if included). Flag a `WARNING` if an appendix in the body is
  missing from Contents.
- A Mermaid diagram exists for both the main end-to-end workflow and the
  logical architecture.

## Gate 8 — Product experience direction (§9)

Check the Product Experience Direction captures UX **intent** without drifting
into UX design. Full fail/warning conditions live in
`references/product-experience-direction.md`; emit the eight Product Experience
Gate rows in the self-check.

- `FAIL` if any of: no primary user defined; no job-to-be-done defined; the
  product needs user interaction but no interaction mode is selected; human
  review is required but no human-review experience is defined; AI uncertainty
  exists but no uncertainty/review behaviour is defined;
  trust/control/transparency requirements are absent for an AI-heavy system; or
  UX assumptions are not handed off to architecture.
- `WARNING` if any of: multiple interaction modes are plausible but none is
  primary; an interaction-mode label is ambiguous or unclassified (a bare "AI
  Skill" not tagged wrapper/integration vs primary surface, or conflated with
  MCP) when multiple modes are listed; MCP is selected without a clear external
  AI-client need; Web UI is selected but users are mostly technical and CLI/API
  may be cheaper; CLI is selected but first users are non-technical; human review
  is deferred despite high quality risk; data egress is possible but user
  visibility is undefined; or auditability is required but user-facing audit
  access is undefined.
- `FAIL` (UX over-reach) if §9 contains screen layout, wireframes, CSS/visual
  design, exact CLI syntax/flags, exact MCP/API schemas, detailed copywriting,
  or implementation tasks — relocate those to a later UX-design/implementation
  stage. `standard` §9 over 1–2 pages is a `WARNING` (compress).

## Gate 9 — Adaptive stage-gate recommendation (§19)

Check that §19 Recommended Next Stages routes the workflow without silently
expanding it. Full fail/warning conditions live in
`references/adaptive-stage-gate-routing.md`; emit the seven Adaptive Stage-Gate
Recommendation Gate rows in the self-check.

- `FAIL` if any of: the Recommended Next Stages section is missing; a stage
  decision uses uncontrolled wording instead of RUN / SKIP / DEFER / ASK_USER; a
  RUN lacks evidence; an ASK_USER does not identify the missing information;
  `architecture-design` is skipped without strong justification; a high-risk
  project (complexity ≥ 13) recommends no optional gates; or ASK_USER is absent
  despite an unresolved high-impact unknown with no downstream owner.
- `WARNING` if any of: `ux-design` is skipped despite human review or multiple
  user roles; `security-review` is skipped despite external data egress;
  `test-design` is skipped despite E2E-critical workflows;
  `tech-stack-selection` is skipped despite multiple serious technology choices;
  a DEFER has no revisit trigger; `architecture-update` /
  `architecture-reconciliation` is not DEFER at blueprint stage; the Stage
  Recommendations table omits the **`Depends On`** column; the recommended
  pipeline is a single flat list with no **Recommended Linear Path** /
  **Conditional Follow-up Gates** split (or a deferred conditional gate is
  dropped from it); the **ASK_USER Decision Rationale** is missing when no stage
  is ASK_USER; or the complexity score is presented as a formal estimate rather
  than a labelled **routing heuristic**.
- Confirm §9 Product Experience Direction signals are reflected in the
  UX/security/test decisions.

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

## Self-repair pass (before delivery)

Behave like a final editor: **detect → repair → re-check → deliver**. For
each `WARNING`:

- If it is a safe wording rewrite (e.g. a runtime-leaning phrase with a
  clear product-level equivalent per `references/borderline-cases.md`, or a
  missing appendix link), **apply the fix now** rather than only reporting it.
- If it is a structural issue, revise the relevant section.
- Keep a `WARNING` only when it genuinely needs human or downstream
  technical judgement (then give it a required action).

Re-run the self-check after repairs so the appendix reflects the
**post-repair** document, not the pre-repair draft.

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

Then add the **Product Experience Gate** rows (§9): Primary user identified ·
Primary job-to-be-done defined · Primary experience thesis defined · Primary
interaction mode selected · Interaction modes classified · Trust / control /
transparency needs defined · Human-in-the-loop experience defined where needed ·
Failure / recovery expectations defined · UX assumptions handed off to
architecture. Each carries a status, a concrete required action for any
WARNING/FAIL, and a blocks-technical-design verdict (here "Blocks Architecture?"
≡ "Blocks TD?").

Then add the **Adaptive Stage-Gate Recommendation Gate** rows (§19): Recommended
Next Stages section exists · Controlled decision values used · RUN decisions
have evidence · SKIP decisions have reason · DEFER decisions have revisit
trigger · ASK_USER decisions identify missing info · Product Experience
Direction informs recommendations · Stage table has Depends On · Linear-path vs
conditional-gates split · ASK_USER absence explained · Complexity score labelled
heuristic. Each carries a status, a required action for any WARNING/FAIL, and a
blocks-next-stage verdict.
