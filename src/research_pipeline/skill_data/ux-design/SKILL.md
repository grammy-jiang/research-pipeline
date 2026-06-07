---
name: ux-design
description: >
  Generate a user-story-driven UX design from an architecture design document.
  Consumes architecture constraints (surfaces, state model, contracts,
  data-egress, failure/recovery, observability) and produces interaction flows,
  user stories, acceptance criteria, E2E scenario seeds, and architecture
  feedback — separating Skill Operator UX from Target Software UX. The fourth
  stage of research-pipeline → blueprint → architecture → ux-design →
  implementation-plan. Use when the user has an architecture design and asks to
  "design the UX", "create UX design", "generate user stories", "design user
  journeys", "create interaction flows", "generate E2E scenario seeds", or "turn
  architecture into UX design". Do NOT use for creating architecture (use
  `architecture`), choosing a tech stack (`architecture --mode stack`), writing
  an implementation plan (use implementation-plan), writing executable tests, or
  wireframes / visual design / CSS / final copy.
license: MIT
---

# Architecture-to-UX Design

This skill is the fourth stage of a research-driven development workflow:

```text
research-pipeline → blueprint → architecture → ux-design → implementation-plan
```

The `architecture` skill answers: **how should this product be technically
structured?**

This `ux-design` skill answers: **how should people and agents actually
experience and interact with the product, end to end?**

The later `implementation-plan` skill answers: **what exact tasks should the
coding agent build, in what order?**

It is **not a "draw screens" skill**. It is a user-story-driven experience
design skill that converts architecture constraints into detailed interaction
flows, acceptance criteria, and E2E scenario seeds. Its highest-value output is
testable user stories, failure/recovery UX, E2E scenario seeds, and architecture
feedback — not visual layout.

## When To Trigger

- "Design the UX" / "create UX design" / "design the user experience from
  architecture"
- "Generate user stories" / "design user journeys" / "create interaction flows"
- "Generate E2E scenario seeds" / "turn architecture into UX design"
- Accepting the `architecture` skill's UX-Design Handoff (its §24 Recommended
  Next Stages and Downstream Handoffs).

## Do Not Use For

- Creating architecture → `architecture` (`--mode design`)
- Choosing a tech stack → `architecture --mode stack`
- Writing an implementation plan → the implementation-plan skill
- Writing executable tests (this skill emits E2E scenario *seeds*, not tests)
- Pixel-perfect wireframes, final visual design, CSS/styling, final screen copy

## Inputs

**Required:**

- One architecture design document, normally
  `<topic-slug>-architecture-design.md` (produced by `architecture --mode
  design`).

**Strongly recommended:**

- The matching `<topic-slug>-product-blueprint.md` (carries §9 Product
  Experience Direction and §19 Recommended Next Stages).

**Optional:**

- `<topic-slug>-architecture-tech-stack.md` (read as constraints, not required)
- `<topic-slug>-architecture-update.md`
- `mode`: interactive / automatic / hybrid (default: hybrid)
- `project_name`, `excluded_surfaces`

If no architecture path is supplied, discover it automatically (see
`references/ux-question-bank.md` and the input-discovery pass): search the
working directory, `./docs`, `./design`, `./artifacts`; rank candidates by
filename (`*-architecture-design.md`), header (`# Architecture Design`), and
presence of `Experience Architecture`, `State Model`, and `Interface Contracts`.
If no architecture design document is found, **STOP** with: *"No architecture
design document found. Run `architecture --mode design` first, or pass the
architecture document explicitly."* Do not run UX design from a blueprint alone.

## Output

Primary output, co-located with the source architecture unless told otherwise:

```text
<topic-slug>-ux-design.md
```

Derive `<topic-slug>` by stripping `-architecture-design.md` from the source
filename; otherwise slugify the project name; if still ambiguous, use
`ux-design.md` and record the naming assumption. If files cannot be written,
emit the full Markdown inline and state the recommended filename.

The UX design document must contain these 22 sections, in order, and begin with
a `## Contents` section and an `## Update History` near the top (see
`templates/ux-design-template.md`):

1. Generation Metadata
2. Source Architecture Interpretation
3. Source Blueprint Interpretation
4. UX Goals and Non-Goals
5. Skill Operator UX
6. Target Software UX
7. Users, Roles, and Jobs-to-Be-Done
8. UX Decision Summary
9. UX Assumptions
10. User Stories
11. Core User Journeys
12. Surface-Specific UX
13. Human-in-the-Loop UX
14. Trust, Control, and Transparency UX
15. Error, Empty, Loading, Degraded, and Recovery States
16. Notifications and Feedback
17. Accessibility and Internationalization
18. UX Observability
19. Acceptance Criteria
20. E2E Scenario Seeds
21. Architecture Feedback / Required Architecture Updates
22. Handoff Notes for Implementation Planning

Plus `## Appendix A. UX Quality-Gate Self-Check`.

Formatting requirements:

- `## Contents` with internal links to every numbered section.
- `## Update History` table (Date · Source Architecture · UX Version · Change
  Type · Affected Sections · Notes). New documents add an initial row; updates
  append a row; never delete prior rows.
- Generation metadata block (§1) — copy, never invent; use `unknown` when
  unavailable.
- **§5 Skill Operator UX and §6 Target Software UX are always separated.**
- Story-driven §10 (see `templates/user-story-template.md`) and Gherkin-style
  §20 E2E scenario seeds (see `templates/e2e-scenario-template.md`).
- §12 includes only surfaces the architecture actually uses.
- §21 Architecture Feedback is **mandatory** (even if "No architecture
  reconciliation required").

## Operating Modes

- **Interactive** — target users / primary surface unclear, high-impact UX
  choices, important human review, ambiguous interaction surfaces. Ask one
  question at a time with a recommended answer, alternatives, and the default
  assumption; record the answer; continue until major UX decisions resolve.
- **Automatic** — architecture provides enough information and questions are
  low-impact. Infer defaults, record assumptions, flag high-risk assumptions for
  review; do not block unless a missing answer would break the UX design.
- **Hybrid (default)** — ask only high-impact UX questions, infer low-risk
  details, record assumptions, and produce a complete document even if some
  questions remain open.

## Method

Follow the staged method. The 13 prompt files under `prompts/` carry the
detailed instructions for each pass; `manifest.json` records the task graph,
dependencies, validation gates, and failure policy. If no runner exists, follow
the manifest order manually.

1. **Discover and resolve input** — find the architecture design (working dir →
   `docs`/`design`/`artifacts`), rank candidates, locate the optional blueprint
   and tech-stack; STOP if no architecture design (`prompts/01`).
2. **Parse the architecture** — project, surfaces, actors, state model,
   workflows, human-review flow, security/egress, observability, failure/
   recovery, Experience Architecture, UX handoff; mark missing sections
   (`prompts/02`).
3. **Parse the blueprint** (if present) — Product Experience Direction,
   experience thesis, JTBD, interaction mode, trust/control/transparency,
   human-in-the-loop, failure/recovery expectations, Recommended Next Stages
   (`prompts/03`).
4. **Run clarification** — interactive / automatic / hybrid; ask only
   high-impact UX questions, one at a time, each with a recommended answer and
   stated default; record assumptions (`prompts/04`,
   `references/ux-question-bank.md`).
5. **Define Skill Operator UX** — how the user drives the skill workflow itself
   (`prompts/05`).
6. **Define Target Software UX** — how end users / agents interact with the
   product being designed (`prompts/06`).
7. **Write user stories** — role / goal / value, preconditions, main +
   alternative + failure/recovery flows, user-visible states, acceptance
   criteria, E2E seeds (`prompts/07`, `templates/user-story-template.md`).
8. **Define surface-specific UX** — CLI / Web-GUI / TUI / AI-Skill / MCP / API,
   only for surfaces the architecture uses (`prompts/08`,
   `references/surface-ux-guide.md`).
9. **Define error / empty / loading / degraded / recovery states** — the
   non-happy paths (`prompts/09`).
10. **Generate E2E scenario seeds** — Gherkin-style seeds from user stories +
    acceptance criteria; not executable tests (`prompts/10`,
    `references/e2e-scenario-seed-guide.md`,
    `templates/e2e-scenario-template.md`).
11. **Produce architecture feedback** — required architecture updates / gaps;
    recommend `architecture --mode reconcile` if needed (`prompts/11`,
    `references/architecture-feedback-guide.md`).
12. **Compose the final UX document** — Contents, metadata, Update History, all
    22 sections + appendix (`prompts/12`, `templates/ux-design-template.md`).
13. **Run the UX quality-gate self-check** (`prompts/13`).

## Quality Gates

The output fails (revise; max 3 attempts, then surface failing gates and stop) —
full text in `prompts/13_quality_gate_self_check.md` and
`tests/expected_sections_checklist.md`:

1. **Source architecture consumed** — §2 reflects the architecture; missing
   architecture sections are marked, not invented.
2. **Product Experience Direction preserved** — blueprint §9 UX intent is
   carried, not changed (or n/a if no blueprint, with a recorded warning).
3. **Skill Operator UX and Target Software UX both defined** — and kept separate.
4. **User stories defined** — structured, with preconditions, main / alternative
   / failure-recovery flows, user-visible states, and acceptance criteria.
5. **Failure / recovery flows defined** — non-happy paths (§15) are explicit;
   UX quality is judged by failure handling, not just the happy path.
6. **Human-review UX defined where needed** — if the architecture has a human
   review flow, §13 defines its UX.
7. **E2E scenario seeds generated** — testable Gherkin-style seeds (§20), not
   executable tests.
8. **Architecture feedback section present** — §21 exists (even if "No
   architecture reconciliation required").
9. **Surface scope controlled** — §12 includes only architecture-supported
   surfaces; WARN if too many surfaces for MVP, CLI-first for non-technical
   users without mitigation, or MCP exposed without a clear agent user.
10. **No out-of-scope output** — no executable tests, no architecture/tech-stack
    decisions, no pixel-level layout / CSS / final copy / exact CLI flags.
11. **Phase and testability metadata** — every user story has a phase tag
    (MVP-0 / MVP-1 / Phase 2 / Phase 3 / Future), primary surface, release gate,
    and depends-on; every E2E seed carries a testability metadata block (phase,
    surface, release gate, deterministic, requires real LLM, CI suitable,
    required fixtures, must mock, required architecture contracts, required
    implementation components); §20 ends with a Testability Summary Table;
    implementation-plan can convert E2E seeds into concrete test tasks without
    guessing.

### Fail Conditions

```text
Architecture document is missing.
Target users are undefined. / Primary surface is undefined.
No user stories exist. / Failure/recovery flows are missing.
Human review exists in architecture but no human-review UX is defined.
E2E scenario seeds are missing. / Architecture feedback section is absent.
No E2E seeds have phase tags.
MVP-0 stories cannot be distinguished from MVP-1/future stories.
E2E seeds lack enough information for implementation-plan to create test tasks.
No CI-suitable MVP-0 E2E seed exists for the core happy path.
```

## Anti-Patterns (do NOT)

- Draw pixel-perfect screens, choose colours, or write final copy.
- Re-decide architecture, the state model, contracts, or the tech stack.
- Emit executable tests (emit E2E scenario *seeds* only).
- Invent user-visible states, surfaces, or operations the architecture does not
  support — record them as architecture feedback instead.
- Include every possible surface in full detail; cover only what the
  architecture uses.
- Ask low-value visual/copy questions (button colour, exact wording, exact CLI
  flags) — those belong to later implementation or visual design.

## References

| File | Load when |
|------|-----------|
| `references/artifact-contract.md` | The Cross-Skill Artifact Contract the UX-design output must satisfy |
| `references/ux-question-bank.md` | Choosing high-impact clarification questions; input discovery + ranking |
| `references/surface-ux-guide.md` | Writing §12 surface-specific UX (CLI / Web / TUI / AI-Skill / MCP / API) |
| `references/e2e-scenario-seed-guide.md` | Turning user stories + acceptance criteria into §20 E2E seeds |
| `references/architecture-feedback-guide.md` | Writing §21 architecture feedback; deciding when to recommend reconcile |
| `templates/ux-design-template.md` | The 22-section + appendix output skeleton |
| `templates/user-story-template.md` | Per-story skeleton (§10) |
| `templates/e2e-scenario-template.md` | Gherkin-style E2E seed skeleton (§20) |
| `examples/translation-system-ux-design.md` | A worked UX design from the translation architecture |

## Final Reminder

This skill bridges technical architecture and implementation planning. Its core
rule:

> UX design consumes architecture constraints and expresses how people and
> agents experience the product — testable user stories, interaction flows,
> failure/recovery UX, and E2E scenario seeds. It never re-decides architecture
> or the tech stack, never writes executable tests, and never produces
> pixel-level visual design. When UX needs something the architecture lacks, it
> records architecture feedback and recommends `architecture --mode reconcile`.
