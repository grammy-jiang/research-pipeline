---
name: blueprint
description: >
  Convert a research synthesis report into an implementation-neutral product
  blueprint. Transforms confidence-graded findings, gap classifications,
  contradiction maps, and risk items into a product concept, target users,
  workflow model, product experience direction, logical architecture,
  information model, decision policies, MVP boundary, evaluation strategy, and
  technical-design handoff. Use when
  the user has a research report and asks to "create a blueprint", "design
  the product", "turn this research into a product", "what should we build?",
  "generate a product blueprint", "convert the research to an MVP plan", or
  accepts the research-pipeline handover offer. Aliases: "product-blueprint",
  "research-blueprint", "research-to-product". Do NOT use for more literature
  research (use `research-pipeline`), tech-stack selection, detailed UI/UX
  design or wireframes, requirements with user stories (use `req-analysis`),
  or single-paper explanation (use `paper-analyzer`).
license: MIT
---

# Research-to-Product Blueprint

This skill is the second stage of a research-driven development workflow.
It consumes a research synthesis report and produces a product-level
blueprint. Its job is **not** to summarise papers again — it converts
research findings into product-design language.

```text
Research mechanisms / confidence-graded findings / gap classifications
  → product primitives → core capabilities → workflows
  → product experience direction → logical architecture
  → MVP boundary → evaluation strategy → technical-design handoff
```

## When To Trigger

- "Create a blueprint from this research report"
- "Design the product based on this research"
- "Turn this research into a product design" / "What should we build?"
- "Generate a product blueprint" / "Convert the research to an MVP plan"
- "Define the product experience direction" / "Define how users should
  experience the product" / "Define user trust/control/review expectations"
- The aliases `product-blueprint`, `research-blueprint`, `research-to-product`
- Accepting the `research-pipeline` handover offer at the end of its
  iterative gap-closure loop (see that skill's
  `references/iterative-synthesis.md`, "System-Design Handover")
- A `<topic-slug>-research-report.md` exists and the user's intent is
  product design, not more research.

Do **not** trigger for:

- More research (use `research-pipeline`)
- Technical architecture or tech-stack selection (a later tech-arch skill)
- Detailed UI/UX design: drawing wireframes, writing screen copy, defining
  exact CLI flags or MCP tool schemas, building frontend components (a later
  UX-design skill). The blueprint defines UX **intent**, not detailed UX.
- Requirements clarification with user stories (use `req-analysis`)
- Single-paper explanation (use `paper-analyzer`)

## Inputs

**Required:**

- One research synthesis report: `<topic-slug>-research-report.md`
  (produced by `research-pipeline` or a compatible research workflow).

**Optional:**

- `gaps.json` — structured `ACADEMIC` / `ENGINEERING` / `OUT_OF_SCOPE`
  classification with priorities
- `runs/<run_id>/summarize/synthesis_report.json` — machine-readable
  findings and gap objects
- `runs/<run_id>/screen/screened.jsonl` — paper metadata for citations
- `runs/<run_id>/plan/query_plan.json` — original topic framing
- `project_name`, `target_domain`, `target_users`, `product_direction`
- `constraints`, `excluded_scopes`
- `mvp_bias`: small / balanced / ambitious
- `risk_tolerance`: low / medium / high
- `output_language`: english / chinese / bilingual
- `output_detail`: concise / standard / detailed (default: standard)

The Markdown report is **authoritative**. JSON artifacts are
supplementary only. If JSON conflicts with the Markdown report, prefer the
Markdown report and note the conflict in §2 (Source Research
Interpretation). If JSON carries structured detail missing from the
Markdown report, use it but mark it as supplementary.

## Output

One Markdown document written to the current working directory:

```text
<topic-slug>-product-blueprint.md
```

If `<topic-slug>` cannot be inferred, derive it from the source report
filename; if no filename is available, derive a lowercase hyphenated slug
from the product thesis. If files cannot be written, emit the full
Markdown blueprint inline and state the recommended filename.

The blueprint must contain these 19 sections, in order:

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
19. Traceability Appendix

Use `templates/product_blueprint_template.md` as the skeleton.

Formatting requirements:

- A `## Contents` section at the top with internal Markdown links to all
  19 sections.
- At least one **Mermaid** diagram for the main end-to-end workflow.
- At least one **Mermaid** diagram for the logical architecture.
- Additional Mermaid workflow diagrams only for complex, safety-critical,
  or high-risk workflows.
- Markdown tables for translation maps, decisions, risks, evaluations,
  and policies.
- Citations as `[arxiv_id]` (e.g. `[2312.01234]`) or `[Author, Year]`
  (e.g. `[Park et al., 2023]`), traceable to the source report's
  `## References`.

## Non-Goals

Do **not** select: a programming language, framework, database, vector
database, cloud provider, deployment platform, UI library, or concrete
package/module structure.

Do **not** produce: code, a database schema, a deployment plan,
vendor-specific architecture, implementation tickets, or detailed coding
tasks.

Do **not** produce detailed UX design: screen layouts, wireframes, visual
hierarchy, exact CLI command syntax/flags, exact MCP tool or API schemas,
copywriting, or full accessibility checklists. §9 captures **UX intent** only.
The boundary is: **blueprint defines UX intent · architecture defines the
UX-enabling technical structure · a later UX-design skill defines the detailed
experience · implementation-plan turns it into build tasks.** See
`references/product-experience-direction.md`.

Do **not** treat research gaps as solved without explicit validation.

When uncertain whether a statement is implementation-neutral, apply the
decision rule in `references/borderline-cases.md`: if removing the
specific term and replacing it with its purpose still conveys the
constraint, keep the conceptual version; if the statement cannot be
expressed without naming a product, vendor, or deployment model, it
belongs to the later technical-design skill.

## Method

Follow the staged method. The five prompt files under `prompts/` carry
the detailed instructions for each stage; the `manifest.json` records the
task graph, gate names, and failure policy.

1. **Read & detect** — read the source report and any supplementary
   artifacts. Detect available sections and the citation style.
2. **Classify input quality** — strong / usable / weak / insufficient
   (see `references/input-mapping.md`). If `insufficient`, STOP and emit
   the standardized insufficient-input failure (do not fabricate a
   blueprint). If `weak`, proceed but mark missing areas as assumptions
   or open questions.
3. **Resolve domain** — if the report spans multiple unrelated product
   domains, scope the blueprint to one. Proceed with the highest-coverage
   domain as a documented default; ask the user only when domains have
   similar evidence coverage and would yield materially different theses.
4. **Extract & classify** — extract mechanisms, methods, patterns,
   benchmarks, assumptions, contradictions, gaps, risks, and architecture
   hints. Tag each with a type and confidence grade (HIGH 🟢 / MEDIUM 🟡 /
   LOW 🔴). Use `prompts/01_extract_research_items.md`.
5. **Apply gap mapping** — `ENGINEERING` gaps → product requirements;
   `ACADEMIC` gaps → validation requirements / open questions (never an
   MVP requirement unless the product's purpose is to answer the research
   question); `OUT_OF_SCOPE` → non-goal. See `references/gap-type-mapping.md`.
6. **Translate** — convert research items into product primitives
   (capability, workflow, policy, conceptual component, information
   object, evaluation requirement, governance rule, risk control, user
   interaction, lifecycle state, integration surface). Merge overlapping
   primitives. Use `prompts/02_translate_to_product_primitives.md`.
7. **Resolve ideas** — decide ADOPT / ADAPT / MERGE / DEFER / REJECT for
   every major idea, conservatively. Use `prompts/03_resolve_ideas.md`.
8. **Size the MVP** — apply the Round History signal (a long round history
   with many remaining gaps, or `Readiness: HAS_GAPS`, means a speculative
   space → heavier DEFER/REJECT pressure). Structure §15 as MVP-0 (smallest
   demonstrable end-to-end slice) / MVP-1 (first usable version) / Safety
   Baseline / Evaluation Baseline / Deferred; keep MVP-0 minimal and do not
   translate research completeness into MVP inclusion.
9. **Compose** — generate the thesis, then all 19 sections with the
   required Mermaid coverage and tables, including §9 Product Experience
   Direction (UX intent only; see `references/product-experience-direction.md`).
   Lead the thesis with the primary
   research-backed architecture, not a conditional/secondary mechanism.
   Copy metadata (never invent; keep pipeline runs vs. gap-closure rounds
   separate; skill version from `manifest.json` or `unknown`). Keep primary
   actors/domains aligned with the thesis. Never leave a citation blank —
   gap-derived items cite `[Source Report: Research Gaps — <name>]`. A
   release gate from a MEDIUM/LOW-confidence mechanism needs a HIGH-risk
   why-now justification or is downgraded. The Contents lists every numbered
   section and every appendix present; keep the main body scannable (move
   large tables to appendices). Respect the `output_detail` length budget.
   Use `prompts/04_generate_blueprint.md` and the templates.
10. **Quality gates, self-repair & self-check** — run the gates in
    `prompts/05_quality_gate.md`, revise failing sections, and **apply safe
    wording rewrites before delivery** (detect → repair → re-check) so only
    judgement-needing warnings remain. Emit the actionable `## Appendix A`
    self-check (gate · status · finding · required action · blocks-TD)
    reflecting the post-repair document. Maximum 3 revision attempts; after
    3 failures, surface the specific failing gates to the user and stop. Do
    not deliver an unvalidated blueprint.

## Decision Categories

- **ADOPT** — use directly (HIGH confidence, product-critical).
- **ADAPT** — use with modification (MEDIUM confidence or partial fit).
- **MERGE** — combine related ideas into one capability.
- **DEFER** — valuable but not MVP-critical, or MEDIUM/LOW confidence.
- **REJECT** — weak evidence, out of scope, unsafe, or incompatible with
  the product thesis.
- **DEFER / VALIDATE** — unresolved `ACADEMIC` gaps that may be valuable
  but are not ready to become product requirements.

## Quality Gates

Pass only if ALL hold (full text in `prompts/05_quality_gate.md` and
`tests/expected_sections_checklist.md`):

1. **Input understanding, metadata & thesis emphasis** — names the source
   research question, acknowledges input quality and the targeted domain
   (if multi-domain), uses copied-not-invented metadata (skill version from
   `manifest.json` or `unknown`; pipeline runs and gap-closure rounds kept
   separate), and leads the thesis with the primary research-backed
   architecture (not a conditional/secondary mechanism).
2. **Traceability** — every major capability traces to a research
   citation (`[arxiv_id]` / `[Author, Year]`) or a constrained explicit
   design decision with rationale; otherwise mark it "Design hypothesis —
   requires validation." No citation cell is blank — gap-derived items cite
   `[Source Report: Research Gaps — <name>]`.
3. **Implementation neutrality** — no programming language, framework,
   database, cloud provider, vendor, package structure, deployment
   commands, code, or tickets.
4. **Workflow completeness** — each major workflow has a trigger, inputs,
   decision gates, steps (or a Mermaid flow), outputs, failure modes, and
   success criteria.
5. **Scope control & MVP discipline** — primary actors/domains match the
   thesis (high-stakes domains seen only as evidence stay Secondary/Future);
   §15 splits the core path into MVP-0 (smallest demonstrable) and MVP-1
   (first usable), separates Safety and Evaluation baselines, and has an
   explicit success definition; `ACADEMIC`-gap items are excluded unless the
   product validates that gap. Flag (do not auto-fail) an MVP-0 over 6
   capabilities without justification, a large Phase-1 system mislabelled
   MVP-0, or `standard` output over budget; fail only if it is no longer a
   small, testable core.
6. **Risk honesty & release-gate confidence** — HIGH-impact risks are
   explicit, mitigations are realistic (never "prompt the model better"),
   safety-critical deferred items are release gates, and risks from
   unvalidated `ACADEMIC` items are flagged. A release gate derived from a
   MEDIUM/LOW-confidence mechanism is justified only by HIGH risk impact +
   no cheaper control + an explicit why-now; otherwise downgrade it.
7. **Downstream usefulness** — a technical-design agent can choose a tech
   stack and plan an implementation without re-reading the papers; the
   Contents section exists with valid links and lists every numbered section
   and every appendix present; the main end-to-end workflow and the logical
   architecture each have a Mermaid diagram.
8. **Product Experience Gate** — §9 names a primary user, job-to-be-done,
   experience thesis, and primary interaction mode; defines
   trust/control/transparency needs, human-in-the-loop, and failure/recovery
   where relevant; and hands UX assumptions off to architecture. It stays UX
   **intent** (no wireframes, screen layout, exact CLI/MCP/API syntax, or copy).
   Full fail/warning conditions in `references/product-experience-direction.md`.

Fail immediately if any tech-stack choice, code, or implementation ticket
appears; if either required Mermaid diagram is missing; if open research
gaps are silently treated as solved; if risks are omitted; or if the
output is an unstructured essay.

## References

| File | Load when |
|------|-----------|
| `references/input-mapping.md` | Mapping report sections → blueprint inputs; quality thresholds |
| `references/gap-type-mapping.md` | Turning ACADEMIC / ENGINEERING / OUT_OF_SCOPE gaps into product actions |
| `references/borderline-cases.md` | Deciding whether a statement is implementation-neutral |
| `references/product-experience-direction.md` | Writing §9 UX intent; the boundary rule; clarification questions; the Product Experience Gate |
| `references/troubleshooting.md` | Insufficient / weak / multi-domain inputs, missing sections, gate failures |
| `templates/product_blueprint_template.md` | The 19-section output skeleton |
| `templates/workflow_template.md` | Writing a workflow with full fields + Mermaid |
| `templates/logical_architecture_template.md` | Writing §10 conceptually (not technically) |
| `templates/translation_map_template.md` | Writing §5 / §6 tables |
| `templates/evaluation_strategy_template.md` | Writing §14 scenarios |

## Final Reminder

This skill bridges research synthesis and technical architecture.

It answers: *"What product should exist, how should it behave, and what
workflow should it implement?"*

It does **not** answer: *"What tech stack should implement it?"* — that is
the next skill's job.
