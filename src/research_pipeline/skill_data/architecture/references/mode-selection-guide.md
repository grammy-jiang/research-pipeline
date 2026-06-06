# Mode-Selection Guide

The `architecture` skill is **one skill with internal modes**, not many
user-facing skills. Architecture *design* and technology *stack selection* are
related but different decision types, so they are separate modes. This guide
defines the mode vocabulary and how to pick a mode. Load it from
`prompts/00_mode_resolver.md`.

## Why modes (not separate skills)

Keep the public skill name `architecture`. Splitting into
`architecture-design` / `tech-stack-selection` / `architecture-update` /
`architecture-reconciliation` / `architecture-review` skills causes sprawl and
breaks artifact continuity. One skill with modes gives a simple user-facing
name, clear internal responsibility separation, less duplication, and easier
artifact continuity.

## The five modes

### `design` (implemented, default)

- **Purpose:** generate the technical architecture from a product blueprint.
- **Input:** product blueprint.
- **Output:** `<topic-slug>-architecture-design.md`.
- **Owns:** system context, C4 views, containers/components, workflow/runtime
  view, state model, interface + data contracts, security/trust boundaries,
  data-egress architecture, observability/audit/telemetry, failure/recovery,
  **Experience Architecture**, the Traditional-vs-AI responsibility boundary,
  **provisional** technology assumptions, and handoffs to stack / UX / security
  / test stages.
- **Must not own:** final technology stack selection, final framework / database
  / cloud / frontend choices, exact package layout, detailed UX flows,
  implementation tasks.
- **Key rule:** design mode may record *provisional* tech assumptions, but the
  *final* technology choice belongs to `stack` mode unless the stack is already
  fixed by the blueprint or by the user.

### `stack` (implemented)

- **Purpose:** select the concrete technology stack.
- **Input:** blueprint + architecture design.
- **Output:** `<topic-slug>-architecture-tech-stack.md`.
- **Owns:** language/runtime, frameworks, storage, queue/background jobs,
  deployment target, LLM provider abstraction, AI orchestration, MCP SDK,
  observability stack, testing stack, packaging — each with alternatives, risk,
  reversibility, and architecture-impact notes; plus an explicit
  *Architecture Update Required?* verdict.
- **Must not own:** the product thesis, MVP scope, the core architecture, UX
  intent, or new workflows not present in the architecture.
- **Key rule:** stack mode selects technologies that **satisfy** the
  architecture; it must not redesign the architecture. If it finds a genuine
  conflict, it emits `Architecture Update Required: yes` with reason + affected
  sections, and stops short of rewriting the architecture (that is `update`).

### `review` (implemented)

- **Purpose:** evaluate architecture quality without changing it. Output:
  `<topic-slug>-architecture-review.md` (`review_tasks`,
  `references/architecture-review-guide.md`).
- **Owns:** a 10-point score breakdown with justifications; blocking / warning /
  polish issue classification; readiness assessments (blueprint fidelity, PED
  preservation, next-stages consumption, security/egress, observability, tech
  stack, UX-design readiness, implementation-plan readiness); recommended next
  actions.
- **Key rule:** **non-mutating** — it never rewrites or patches the architecture.

### `update` (implemented)

- **Purpose:** apply already-accepted decisions into the architecture. Output:
  `<topic-slug>-architecture-update.md` (an update note; optionally a proposed
  `…-architecture-design.updated.md`) — `update_tasks`,
  `references/architecture-update-guide.md`.
- **Owns:** Accepted Decisions Applied; Sections Requiring Update; Architecture
  Patch Summary; updated ADRs/decision register; updated handoffs; a
  compatibility check; the update quality gate.
- **Key rule:** apply only **accepted** decisions (priority source: a tech-stack
  with `Architecture Update Required? = Yes`, then accepted reconciliation,
  security-review, newer blueprint, explicit user decision — **never** ux-design
  directly). **Does not overwrite** `…-architecture-design.md` by default.

### `reconcile` (implemented)

- **Purpose:** resolve conflicts/gaps between the architecture and a downstream
  artifact (ux-design / test-design / security-review / implementation-plan
  feedback). Output: `<topic-slug>-architecture-reconciliation.md`
  (`reconcile_tasks`, `references/architecture-reconciliation-guide.md`).
- **Owns:** a findings table (severity + traceable source); missing-architecture-
  support mapping; a minimal patch plan; the `Architecture Update Required?`
  verdict + handoff to `update`.
- **Key rule:** conflict-driven; runs only when a downstream artifact exposes a
  real mismatch (primarily a ux-design Architecture Feedback section). **Does not
  patch** the architecture by default; it recommends, and `update` applies the
  accepted changes. Do not blindly accept a downstream artifact that contradicts
  the blueprint.

## Selection logic

### Explicit mode

```text
architecture --mode design     <blueprint.md>
architecture --mode stack       <architecture-design.md>
architecture --mode update      <architecture-design.md> <architecture-tech-stack.md>
architecture --mode review      <architecture-design.md>
architecture --mode reconcile   <architecture-design.md> <ux-design.md>
```

### Automatic detection

| Situation | Mode |
|---|---|
| blueprint input, no architecture exists | `design` |
| "choose the tech stack / select frameworks / decide database / deployment" | `stack` |
| "review / evaluate / score" / "is this ready for implementation?" | `review` |
| "update architecture based on this stack/decision" / "apply this stack" | `update` |
| architecture + ux/test/security findings exposing a mismatch / "reconcile" | `reconcile` |
| bare `architecture` and an architecture already exists | `review` (safest) |

### Ambiguity

- **Bare `architecture` with an existing architecture defaults to `review`** —
  it is non-mutating and safest. Do **not** default to `update`.
- Default to `design` only when no architecture exists.
- Prefer the non-mutating mode when changing an existing document would be risky.
- Ask the user only when the mode materially changes the output and cannot be
  inferred from the request or the artifacts present.

## Downstream flow

```text
blueprint
  -> architecture --mode design
  -> architecture --mode stack
  -> architecture --mode update      (if stack decisions change the architecture)
  -> ux-design
  -> architecture --mode reconcile   (if UX exposes gaps)
  -> implementation-plan
```

The `ux-design` skill consumes this architecture's `design` output (its
Experience Architecture section and UX-Design Handoff) and emits
`<topic-slug>-ux-design.md`. If `ux-design` exposes architecture gaps, it records
them as architecture feedback and recommends `architecture --mode reconcile`.
