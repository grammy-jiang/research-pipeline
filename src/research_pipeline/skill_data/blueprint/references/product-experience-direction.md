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
  Skill / MCP / Hybrid, with a **classification** (see below), MVP stage, and
  rationale.
- **Secondary / Future Interaction Modes** — staged, with a classification and a
  revisit trigger.
- **Critical Trust, Control, and Transparency Requirements** — and their
  architecture impact.
- **Human-in-the-Loop Experience** — trigger, user decision, expected product
  support, MVP stage (only where review is actually needed).
- **Failure and Recovery Expectations** — high-level, not error-message copy.
- **UX Assumptions for Architecture** — source, reversibility, revisit trigger.
- **Product Experience Handoff to Architecture** — UX decision → architecture
  impact.

Keep it **compact**: 1–2 pages maximum in `standard` output, tables over prose.

## Interaction-mode classification

Architecture consumes §9, so an ambiguous interaction-mode label can mislead it
(e.g. building a separate AI-skill runtime when the intent was only a callable
wrapper). Give every mode in §9.4 (Primary) and §9.5 (Secondary / Future) a
controlled **Classification**:

| Classification | Meaning |
|---|---|
| **primary surface** | The main runtime interface users/operators drive at MVP. |
| **secondary surface** | An additional surface present at MVP, but not primary. |
| **wrapper / integration surface** | A callable wrapper around another surface (e.g. an AI skill or thin API over the CLI) — not a separate runtime product. |
| **future surface** | Deferred to a later phase, with a revisit trigger. |

Disambiguation rules:

- **"AI Skill" must be classified explicitly** — usually a *wrapper /
  integration surface* (a callable workflow around the CLI/core), occasionally a
  *primary surface* (a distinct runtime). Never leave it as a bare "AI Skill"
  that architecture has to guess at.
- **"AI Skill" is not "MCP".** MCP is a tool surface for *external AI agents* to
  call the system; an AI-skill wrapper is how *this* workflow is invoked. Tag
  them separately; do not merge them in one cell.
- Combining two modes in one cell (e.g. "CLI / AI Skill") is acceptable **only**
  if both share the same classification; otherwise split them into separate rows.

This is still UX **intent** — classify the surface, do not design it (no command
syntax, schemas, or routes).

## Agent/tool-driven authorization boundary

If the primary interaction mode, or any secondary mode promoted into MVP scope,
is **agent-callable / tool-driven** (for example AI Skill, MCP, external-agent
API, or another delegated action surface), §9 must include a blueprint-level
**READ/ACT authorization boundary**:

- **READ:** what information the agent/tool caller may inspect, and what remains
  user-only or human-review-only.
- **ACT:** what the agent/tool caller may change, trigger, submit, delete, or
  approve without another human decision.
- **Human approval:** which actions require explicit user/operator approval,
  especially irreversible, externally visible, or cross-scope actions.
- **Downstream owner:** architecture/security review owns exact contracts,
  permission models, and runtime enforcement.

Also add a **matching §13 risk row** for **agent authority-confusion** and/or
**prompt injection** whenever the mode can cause actions, data egress, or
decision delegation. This is a named requirement because generic trust/control
wording has not reliably produced the boundary in downstream reviews.

Keep the boundary at product-blueprint altitude: do not define exact MCP tool schemas,
API routes, permission tables, request/response payloads, or
implementation tasks. Those belong to architecture, security-review, and
implementation-plan.

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
| Interaction modes classified | Each §9.4/§9.5 mode carries a controlled classification; "AI Skill" is disambiguated and not conflated with MCP (only when multiple modes are listed). |
| Trust / control / transparency needs defined | Present for an AI-heavy system. |
| Human-in-the-loop experience defined where needed | Present when review is required. |
| Failure / recovery expectations defined | High-level recovery experience present. |
| Agent/tool authorization boundary defined where needed | Agent-callable / tool-driven primary or promoted secondary modes state READ/ACT boundaries. |
| Matching agent authority risk row present where needed | §13 includes agent authority-confusion / prompt injection risk when agent/tool modes can act, delegate, or expose data. |
| UX assumptions handed off to architecture | Handoff table maps UX → architecture impact. |

### Fail conditions

`FAIL` if:

- no primary user is defined;
- no job-to-be-done is defined;
- the product needs user interaction but no interaction mode is selected;
- human review is required but no human-review experience is defined;
- AI uncertainty exists but no uncertainty/review behaviour is defined;
- trust/control/transparency requirements are absent for an AI-heavy system;
- primary or promoted secondary agent-callable / tool-driven modes lack a
  READ/ACT authorization boundary;
- agent-callable / tool-driven modes can act, delegate, or expose data but §13
  lacks a matching agent authority-confusion / prompt injection risk row;
- UX assumptions are not handed off to architecture.

### Warning conditions

`WARNING` if:

- multiple interaction modes are plausible but no primary mode is selected;
- MCP is selected without a clear external AI-client need;
- Web UI is selected but users are mostly technical and CLI/API may be cheaper;
- CLI is selected but the first users are non-technical;
- human review is deferred despite high quality risk;
- data egress is possible but user visibility is not defined;
- auditability is required but user-facing audit access is not defined;
- an interaction-mode label is ambiguous or unclassified (e.g. a bare "AI Skill"
  not tagged as a wrapper/integration vs primary surface, or conflated with MCP).

## Output budget

- **Standard:** §9 is 1–2 pages maximum; use tables, not long prose; no
  journeys, wireframes, command syntax, or screen design; include only UX
  detail that affects architecture or MVP scope.
- **Detailed:** may add more user stories and journeys, but still no
  screen-level visual design and no exact command syntax (illustrative only).
