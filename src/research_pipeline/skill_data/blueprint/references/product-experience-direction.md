# Product Experience Direction (§9)

This reference governs §9 of the blueprint. It captures **Product Experience
Direction** — enough product-experience intent to keep downstream architecture
and implementation from inventing UX assumptions. It is **not** a UX-design
stage.

## Boundary rule

Use this boundary throughout the skill:

- **Blueprint** defines **UX intent**.
- **Architecture** defines **UX-enabling technical structure** (surfaces,
  states, contracts, review artifacts, observability, security boundaries).
- A later **UX-design** skill defines the **detailed experience** (journeys,
  screens, command/conversation UX, copy, accessibility).
- **Implementation-plan** turns it into build tasks.

| Stage | UX responsibility |
|---|---|
| Research report | Extract ideas, methods, findings, gaps. **No UX design.** |
| Blueprint | Define product experience **direction**. |
| Architecture | Translate UX direction into surfaces, states, contracts, security, observability, audit, and review flows. |
| UX-design (later) | Detailed journeys, screen flows, CLI/TUI UX, MCP/tool UX, conversation UX, copy, accessibility. |
| Implementation-plan | Create build tasks. |

The section answers **"What experience should the product provide?"** — never
**"What exactly does the interface look like?"**

## What §9 should include

A compact `## 9. Product Experience Direction` section with:

- **Primary Experience Thesis** — one or two sentences on how the product
  should *feel* (e.g. automated by default, conservative under quality risk,
  transparent about review and data-egress).
- **Primary User / Operator** — who the experience is for and their experience
  need.
- **Primary Job-to-Be-Done** — the job and its success outcome.
- **Primary Interaction Mode** — CLI / Web UI / Desktop GUI / TUI / API / AI
  Skill / MCP / Hybrid, with MVP stage and rationale.
- **Secondary / Future Interaction Modes** — staged, with a revisit trigger.
- **Critical Trust, Control, and Transparency Requirements** — and their
  architecture impact.
- **Human-in-the-Loop Experience** — trigger, user decision, expected product
  support, MVP stage (only where review is actually needed).
- **Failure and Recovery Expectations** — high-level, not error-message copy.
- **UX Assumptions for Architecture** — source, reversibility, revisit trigger.
- **Product Experience Handoff to Architecture** — UX decision → architecture
  impact.

Keep it **compact**: 1–2 pages maximum in `standard` output, tables over prose.

## What §9 must NOT include

These belong to a later UX-design or implementation stage, not the blueprint:

- screen layout, button placement, CSS / styling, visual hierarchy, wireframes
- full user journeys
- exact CLI command syntax or flags
- exact MCP tool schemas
- exact API route definitions
- detailed copywriting / final error-message catalogue
- mobile navigation
- a complete accessibility checklist
- implementation tasks

Quality rule: **blueprint UX content should constrain downstream decisions, not
solve detailed UX.** Only include UX detail that affects architecture or MVP
scope.

## Clarification behaviour

Ask a UX question **only when the answer materially affects** product
direction, architecture, risk, or downstream implementation. Use a
grill-me-style pattern: one question at a time, with a recommended answer and
why it matters. Never ask something the research report or supplied context
already answers. In automatic mode, infer a reasonable default and record it as
an assumption; **high-impact assumptions must be review-flagged** in
§9 "UX Assumptions for Architecture".

### Question format

```text
### Product Experience Clarification Question <N>
**Question:**
...
**Why this matters:**
...
**Recommended answer:**
...
**Alternatives:**
- ...
**If no answer is provided:**
The blueprint will assume ...
```

### Good questions (blueprint-level)

- Who is the first real user: you, a technical operator, a non-technical
  reviewer, a team, or another AI agent?
- Should the first version feel like a controlled tool, an autonomous
  assistant, or a strict workflow engine?
- Should MVP-0 be CLI-first, Web-first, API-first, AI-skill-first, MCP-first,
  or hybrid?
- When the system is uncertain, should it stop, continue with warnings, or
  require human review?
- Should the user see technical quality scores, simplified risk labels, or
  both?
- Should human review happen inside the product or through exported review
  packets?
- Should source-data egress require explicit user confirmation?
- Should audit evidence be human-readable, machine-readable, or both?

### Bad questions (too detailed for blueprint)

- What should the button say?
- Where should the progress bar sit?
- Should the dashboard use cards or tables?
- What exact CLI flag should be used?
- What exact MCP schema should be generated?
- What is the final error-message copy?

## Product Experience Gate

Add these rows to the Appendix A self-check (status `PASS` / `WARNING` /
`FAIL`, with a required action and a blocks-technical-design verdict).

| Gate | Checks |
|---|---|
| Primary user identified | §9 names a primary user/operator. |
| Primary job-to-be-done defined | §9 states the job and success outcome. |
| Primary experience thesis defined | §9 has a one/two-sentence experience thesis. |
| Primary interaction mode selected | A primary mode with rationale + MVP stage. |
| Trust / control / transparency needs defined | Present for an AI-heavy system. |
| Human-in-the-loop experience defined where needed | Present when review is required. |
| Failure / recovery expectations defined | High-level recovery experience present. |
| UX assumptions handed off to architecture | Handoff table maps UX → architecture impact. |

### Fail conditions

`FAIL` if:

- no primary user is defined;
- no job-to-be-done is defined;
- the product needs user interaction but no interaction mode is selected;
- human review is required but no human-review experience is defined;
- AI uncertainty exists but no uncertainty/review behaviour is defined;
- trust/control/transparency requirements are absent for an AI-heavy system;
- UX assumptions are not handed off to architecture.

### Warning conditions

`WARNING` if:

- multiple interaction modes are plausible but no primary mode is selected;
- MCP is selected without a clear external AI-client need;
- Web UI is selected but users are mostly technical and CLI/API may be cheaper;
- CLI is selected but the first users are non-technical;
- human review is deferred despite high quality risk;
- data egress is possible but user visibility is not defined;
- auditability is required but user-facing audit access is not defined.

## Output budget

- **Standard:** §9 is 1–2 pages maximum; use tables, not long prose; no
  journeys, wireframes, command syntax, or screen design; include only UX
  detail that affects architecture or MVP scope.
- **Detailed:** may add more user stories and journeys, but still no
  screen-level visual design and no exact command syntax (illustrative only).
