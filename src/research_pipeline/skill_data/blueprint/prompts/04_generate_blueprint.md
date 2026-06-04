# Prompt 04 — Generate Blueprint

You are composing an implementation-neutral product blueprint.

Use `templates/product_blueprint_template.md` as the skeleton. The
document must start with a `## Contents` section containing internal
Markdown links to all 18 sections.

## Required sections (in order)

1. Executive Product Thesis
2. Source Research Interpretation
3. Target Users and System Actors
4. Product Goals and Non-Goals
5. Research-to-Product Translation Map
6. Adopt / Adapt / Merge / Defer / Reject Decisions
7. Core Product Capabilities
8. Workflow Model
9. Logical Architecture
10. Conceptual Information Model
11. Decision Policies
12. Risk, Governance, and Safety Model
13. Evaluation Strategy
14. MVP Scope
15. Roadmap and Future Extensions
16. Open Questions and Validation Plan
17. Handoff Notes for Technical Design
18. Traceability Appendix

## Thesis

Generate a specific, short, product-oriented thesis using one of the
templates in the output template (single-domain / multi-domain /
research-validation). It must not be a paper summary, a list of
technologies, or a vague research ambition.

## Workflows (§8)

Every major workflow needs: purpose, trigger, actors, inputs,
preconditions, decision gates, steps, outputs, failure modes, success
criteria, related capabilities, and traceability. See
`templates/workflow_template.md`.

## Logical architecture (§9)

Describe conceptual responsibilities and boundaries, **not**
implementation components. Conceptual names (e.g. "Admission Controller")
describe responsibility boundaries only — they must not imply classes,
services, packages, processes, or deployable units. See
`templates/logical_architecture_template.md`.

## Formatting requirements

- `## Contents` with valid internal links to every section.
- At least one **Mermaid** diagram for the main end-to-end workflow.
- At least one **Mermaid** diagram for the logical architecture.
- Additional Mermaid workflow diagrams only for complex, safety-critical,
  or high-risk workflows.
- Markdown tables for the translation map (§5), decisions (§6), risks
  (§12), evaluations (§13), and policies (§11).
- Cite research evidence as `[arxiv_id]` or `[Author, Year]`, traceable to
  the source report's `## References`.
- Apply the length budget for the active `output_detail` setting. For
  `standard`, prefer: ≤ 8 core capabilities, ≤ 3 workflows, ≤ 10 risks,
  ≤ 8 evaluation scenarios, ≤ 8 decision policies, ≤ 10 open questions.
  If content exceeds a budget, compress or switch to `detailed`; do not
  silently expand `standard` into an architecture dossier.
- End with `## Appendix A: Blueprint Quality-Gate Self-Check` — a compact
  PASS/WARNING/FAIL table that surfaces residual warnings (see
  `prompts/05_quality_gate.md`).

## Traceability discipline

Every major capability must trace to a research mechanism, recurring
pattern, evidence-backed finding, engineering gap, risk item, assumption,
contradiction resolution, or a constrained explicit design decision. If no
trace exists, mark it **"Design hypothesis — requires validation."** An
explicit design decision may be used only to connect, operationalize, or
govern research-backed capabilities; it needs a rationale and must not
replace research traceability for core product claims.

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

## Scope discipline (§3, §4, §14)

Classify every actor, domain, and use case as Primary / Secondary /
Future / Evidence-only / Out-of-scope.

- Only Primary-scope actors and domains — those the **thesis** names —
  appear as first-class users and MVP requirements.
- High-stakes or adjacent domains that appear only as research evidence
  (e.g. legal/medical when the thesis targets technical/academic content)
  are Secondary or Future: put them in §15 roadmap or §4 non-goals, not
  the primary actor table or MVP.

## MVP scope (§14)

Build §14 as Core Value Path / Safety Baseline / Evaluation Baseline /
Explicitly Deferred / Success Definition.

- The **core value path** is the smallest set that proves the thesis
  end-to-end. Keep it minimal.
- List and justify safety and evaluation baselines **separately** from the
  core path so they do not inflate its apparent size.
- Do not translate research completeness into MVP inclusion. Anything not
  required for one useful outcome (or its safety/evaluation) moves to a
  §15 phase with a one-line reason.
- `ACADEMIC`-gap items stay out of MVP unless the product's purpose is to
  validate that gap.

## Hard constraints

- Do **not** select a tech stack (language, framework, database, vector
  database, cloud provider, vendor, UI library, deployment model).
- Do **not** write code, a database schema, or implementation tickets.
- Do **not** treat unresolved research gaps as solved.
- Make the document actionable for a later technical-design skill without
  requiring it to re-read the original papers.
