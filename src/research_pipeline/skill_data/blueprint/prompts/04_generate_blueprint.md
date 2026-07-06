# Prompt 04 — Generate Blueprint

You are composing an implementation-neutral product blueprint.

Use `templates/product_blueprint_template.md` as the skeleton. The
document must start with a `## Contents` section containing internal
Markdown links to all 20 sections.

## Required sections (in order)

1. Executive Product Thesis
2. Source Research Interpretation
3. Target Users and System Actors
4. Product Goals and Non-Goals
5. Research-to-Product Translation Map
6. Adopt / Adapt / Merge / Defer / Reject Decisions
7. Core Product Capabilities
8. Workflow Model
9. Product Experience Direction
10. Logical Architecture
11. Conceptual Information Model
12. Decision Policies
13. Risk, Governance, and Safety Model
14. Evaluation Strategy
15. MVP Scope
16. Roadmap and Future Extensions
17. Open Questions and Validation Plan
18. Handoff Notes for Technical Design
19. Recommended Next Stages
20. Traceability Appendix

## Thesis

Generate a specific, short, product-oriented thesis using one of the
templates in the output template (single-domain / multi-domain /
research-validation). It must not be a paper summary, a list of
technologies, or a vague research ambition.

**Emphasis control.** Lead with the *primary* research-backed architecture,
not a conditional or secondary mechanism. If the report frames a mechanism
as conditional, bounded, escalation-only, or secondary, describe it as a
supporting mechanism — do not make it the product identity. (E.g. if the
evidence says "backbone-first generation with conditional review", lead
with the backbone-first engine, not with "multi-agent".)

## Workflows (§8)

Every major workflow needs: purpose, trigger, actors, inputs,
preconditions, decision gates, steps, outputs, failure modes, success
criteria, related capabilities, and traceability. See
`templates/workflow_template.md`.

## Product experience direction (§9)

Capture **UX intent**, not UX design. This section exists so the later
architecture stage does not invent UX assumptions. Blueprint defines the
experience direction; architecture defines the UX-enabling technical structure;
a later UX-design skill defines the detailed experience. Include, compactly:
Primary Experience Thesis · Primary User / Operator · Primary Job-to-Be-Done ·
Primary Interaction Mode (with a Classification, MVP stage + rationale) ·
Secondary / Future Interaction Modes · Critical Trust, Control, and Transparency
Requirements · Human-in-the-Loop Experience · Failure and Recovery Expectations ·
UX Assumptions for Architecture · Product Experience Handoff to Architecture.

- **Classify every interaction mode** (§9.4 and §9.5) with a controlled value:
  *primary surface* · *secondary surface* · *wrapper / integration surface* ·
  *future surface*. Disambiguate "AI Skill" (usually a wrapper / integration
  surface around the CLI/core, not a separate runtime) and never conflate it
  with MCP (a tool surface for external AI agents). See
  `references/product-experience-direction.md`.

- Keep it to **1–2 pages** in `standard` output; use tables, not long prose;
  include only UX detail that affects architecture or MVP scope.
- Do **not** include screen layout, button placement, CSS/styling, visual
  hierarchy, wireframes, full user journeys, exact CLI command syntax/flags,
  exact MCP tool schemas, exact API routes, detailed copywriting, mobile
  navigation, a full accessibility checklist, or implementation tasks — those
  belong to a later UX-design or implementation stage.
- Ask a UX clarification question only when the answer materially affects
  product direction, architecture, risk, or implementation. In automatic mode,
  infer a reasonable default and record it in §9 "UX Assumptions for
  Architecture"; high-impact assumptions must be review-flagged.
- When the primary interaction mode, or a secondary mode promoted into MVP
  scope, is agent-callable / tool-driven, include a blueprint-level READ/ACT
  authorization boundary: who/what can read, who/what can act, where human
  approval is required, and which downstream stage owns detailed schemas. Add a
  matching §13 risk row for agent authority-confusion / prompt injection. Keep
  exact MCP tool schemas, API routes, and permission tables deferred to
  architecture/security review.

See `references/product-experience-direction.md` for the section template,
clarification-question format, the Product Experience Gate, and the boundary
rule.

## Logical architecture (§10)

Describe conceptual responsibilities and boundaries, **not**
implementation components. Conceptual names (e.g. "Admission Controller")
describe responsibility boundaries only — they must not imply classes,
services, packages, processes, or deployable units. See
`templates/logical_architecture_template.md`.

## Formatting requirements

- `## Contents` with valid internal links to every numbered section **and
  every appendix actually present** — scan the final headings and add any
  `## Appendix …` (the Appendix A self-check is always present; the optional
  Appendix B register only when you include it).
- At least one **Mermaid** diagram for the main end-to-end workflow.
- At least one **Mermaid** diagram for the logical architecture.
- Additional Mermaid workflow diagrams only for complex, safety-critical,
  or high-risk workflows.
- Markdown tables for the translation map (§5), decisions (§6), risks
  (§13), evaluations (§14), and policies (§12).
- Cite research evidence as `[arxiv_id]` or `[Author, Year]`, traceable to
  the source report's `## References`.
- Apply the length budget for the active `output_detail` setting. For
  `standard`, prefer: ≤ 8 core capabilities, ≤ 3 workflows, ≤ 10 risks,
  ≤ 8 evaluation scenarios, ≤ 8 decision policies, ≤ 10 open questions,
  and a §9 Product Experience Direction of ≤ 1–2 pages (tables, not prose).
  Keep the main body scannable in one pass — move large tables (the full
  §5 translation map, the §20 traceability appendix) to appendices rather
  than the main flow. If content exceeds a budget, compress or switch to
  `detailed`; do not silently expand `standard` into an architecture dossier.
- End with `## Appendix A: Blueprint Quality-Gate Self-Check` — a compact
  table (Gate · Status · Finding · Required Action · Blocks Technical
  Design?). Every WARNING must carry a concrete required action and a
  blocks-technical-design verdict, not a passive note (see
  `prompts/05_quality_gate.md`).

## Traceability discipline

Every major capability must trace to a research mechanism, recurring
pattern, evidence-backed finding, engineering gap, risk item, assumption,
contradiction resolution, or a constrained explicit design decision. If no
trace exists, mark it **"Design hypothesis — requires validation."** An
explicit design decision may be used only to connect, operationalize, or
govern research-backed capabilities; it needs a rationale and must not
replace research traceability for core product claims.

**Citation fidelity for load-bearing claims.** For thesis emphasis, primary
interaction mode, and primary actor claims, re-read the cited source-report
section before finalizing. If the cited section supports a different concern
(for example build-time developer tooling rather than the product's runtime
interaction mode), reclassify the statement as a product-design decision with
rationale instead of calling it research-derived. Do not upgrade confidence
grades beyond the source report. The deterministic pre-gate checks that paper
citations exist in the source report's `## References` section.

**Gap-citation fallback.** Do not leave a citation cell blank. Paper-derived
items cite `[arxiv_id]` / `[Author, Year]`. An item derived from a
source-report gap with no paper citation cites
`[Source Report: Research Gaps — <gap name>]`. Leave a citation blank only
when the row is explicitly an internal design hypothesis.

## Metadata integrity (§1.5–1.6)

Copy metadata from the source report and skill metadata; never infer,
normalise, invent, or upgrade a value.

- The **blueprint skill version** comes from `manifest.json` (`version`).
  If you cannot read it, write `unknown` — do not invent a number
  (e.g. "1.0").
- Keep **pipeline runs integrated** and **gap-closure rounds** as separate
  fields. A report that "consolidates N pipeline runs" states a run count,
  not a gap-closure round count; if no round count is given, write
  `unknown` for rounds.
- Any field not explicitly available is `unknown` / `not specified`.

## Scope discipline (§3, §4, §15)

Classify every actor, domain, and use case as Primary / Secondary /
Future / Evidence-only / Out-of-scope.

- Only Primary-scope actors and domains — those the **thesis** names —
  appear as first-class users and MVP requirements.
- High-stakes or adjacent domains that appear only as research evidence
  (e.g. legal/medical when the thesis targets technical/academic content)
  are Secondary or Future: put them in §16 roadmap or §4 non-goals, not
  the primary actor table or MVP.

## MVP scope (§15)

Build §15 as MVP-0 / MVP-1 / Safety Baseline / Evaluation Baseline /
Explicitly Deferred / Success Definition.

- **MVP-0** is the smallest end-to-end slice that demonstrates the thesis
  with one realistic input and one realistic output. Keep it minimal — do
  not put every routing, memory, governance, and evaluation mechanism in
  MVP-0 unless the product cannot function or be safely evaluated without
  it. Do not call a large Phase-1 system an MVP-0.
- **MVP-1** is the first usable production version: production hardening,
  richer evaluation, and optional escalation added on top of MVP-0. If the
  product has more than 4 major capabilities the MVP-0/MVP-1 split is
  mandatory; for a genuinely simple product MVP-0 and MVP-1 may coincide —
  say so rather than inventing a split.
- List and justify safety and evaluation baselines **separately** from the
  core path, noting which are required already at MVP-0.
- Do not translate research completeness into MVP inclusion. Anything not
  required moves to a §16 phase with a one-line reason.
- `ACADEMIC`-gap items stay out of MVP unless the product's purpose is to
  validate that gap.

## Release-gate discipline (§13)

A release gate derived from a MEDIUM- or LOW-confidence mechanism is
justified only when the risk impact is HIGH, no cheaper baseline control
exists, and the blueprint states why it is required now. Otherwise
downgrade it to a warning, an evaluation/monitoring requirement, or a
Phase 2 release gate. State each gate's confidence and risk impact.

## Recommended next stages (§19)

After the technical-design handoff, recommend which downstream stages should run
next. The blueprint is the **first adaptive stage-gate router** after research
extraction — it must guide the workflow, not silently expand it. Generate three
compact artifacts, all overrideable defaults (full rules in
`references/adaptive-stage-gate-routing.md`):

- **Pipeline Complexity Assessment** — score seven dimensions 0–3 (user-facing
  complexity, technical ambiguity, security/privacy risk, AI/LLM uncertainty,
  integration complexity, human-review complexity, testing/E2E importance); give
  the total `/ 21` and a workflow class (simple / lightweight / medium / complex).
  Add a one-line note that the score is a **routing heuristic, not a formal
  project estimate**, to be revisited after architecture-design.
- **Stage Recommendations** — one row per stage (architecture-design,
  tech-stack-selection, ux-design, security-review, test-design,
  architecture-update, architecture-reconciliation) with a controlled decision
  (**RUN / SKIP / DEFER / ASK_USER**), a **`Depends On`** prerequisite,
  confidence, reason, blocks-next-step, and revisit trigger. Columns:
  `Stage | Decision | Depends On | Confidence | Reason | Blocks Next Step? | Revisit Trigger`.
- **ASK_USER Decision Rationale** — if any stage is ASK_USER, name the missing
  input; if none is, briefly justify why each high-impact unknown is already
  answered, deferred to architecture-design, or delegated to security-review
  (name the owner). Do not silently assume a high-impact unknown.
- **Recommended Pipeline** — split into a **Recommended Linear Path** (the core
  ordered sequence) and a small **Conditional Follow-up Gates** table
  (`Gate | Run When | Typical Input | Output`) so deferred conditional stages —
  chiefly architecture-update / architecture-reconciliation — are not dropped
  from the picture.
- **Stage-Gate Decision Log** — evidence, risk-if-wrong, and revisit trigger for
  the key decisions.

Rules:

- `architecture-design` is normally **RUN**; SKIP only for a trivial or no-build
  output.
- `architecture-update` and `architecture-reconciliation` default to **DEFER**
  at blueprint stage (no architecture document exists yet) and appear in the
  Conditional Follow-up Gates table with their run condition.
- Every RUN/ASK_USER cites blueprint evidence; every SKIP gives a reason; every
  DEFER names a revisit trigger; every stage names what it `Depends On`. No vague
  wording (`maybe`, `consider`, `nice to have`).
- Let §9 Product Experience Direction drive the UX/security/test decisions
  (human review → ux-design / test-design; external egress → security-review;
  CLI-first → ux-design DEFER; future MCP → tech-stack-selection RUN/DEFER).
- Keep §19 compact: a complexity table + heuristic note, a recommendation table
  with `Depends On`, a few-line ASK_USER rationale, a recommended pipeline
  (linear path + a small conditional-gates table), and a decision log. The
  clarity additions are columns and small tables, not new essays.

## Pre-delivery propagation check

Before delivering, scan whether you are generating from scratch or applying the
amend an existing blueprint guidance from `references/troubleshooting.md`.
For any new or changed load-bearing fact, update its dependent sections and
Appendix A before running the quality gate:

- interaction mode → §3, §8, §9.4/9.5, §10, §16, §18, §19, Appendix A
- MVP roster → §7, §12, §13, §14, §15, Appendix A

If a dependent section is intentionally unchanged, state why in the affected
section or the quality-gate self-check. Do not leave stale references to a prior
interaction mode, actor, capability roster, phase, risk, or downstream-stage
recommendation.

## Optional Appendix B — Design Decision Register

When the blueprint is intended for technical-architecture handoff (and is
detailed enough to warrant it), you may add `## Appendix B: Design Decision
Register` with columns Decision · Type · Rationale · Evidence · Reversible?
· Revisit Trigger. Include only decisions that materially affect downstream
architecture; do **not** duplicate the §6 table — the value is the
reversibility / revisit-trigger view §6 lacks. If you add it, list it in
Contents.

## Hard constraints

- Do **not** select a tech stack (language, framework, database, vector
  database, cloud provider, vendor, UI library, deployment model).
- Do **not** write code, a database schema, or implementation tickets.
- Do **not** treat unresolved research gaps as solved.
- Make the document actionable for a later technical-design skill without
  requiring it to re-read the original papers.

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
