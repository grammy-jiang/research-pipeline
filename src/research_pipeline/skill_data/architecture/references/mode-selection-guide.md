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

### `update` (implemented via design mode)

- **Purpose:** patch an existing architecture after an accepted decision changes
  (selected stack, changed blueprint, accepted user decision, security finding).
- **Mechanism:** reuse design mode's existing update-mode machinery —
  `regenerate` / `patch` / `compare` / `adr-only` / `resume` (see `prompts/02`
  and `tests/update_behavior_checklist.md`). Append an Update History row;
  supersede ADRs rather than overwriting them. Update mode applies *known
  accepted changes*; it is not open-ended critique.

### `review` (recognized — deferred)

- **Purpose:** evaluate architecture quality without changing it.
- **Status:** recognized so the vocabulary is stable, but not yet a dedicated
  mode. For now, produce **read-only** commentary by applying the design
  quality-gate lens (`prompts/23`) to the existing document. It must not
  silently rewrite the architecture.

### `reconcile` (recognized — deferred)

- **Purpose:** resolve conflicts between architecture and a downstream artifact
  (ux-design / test-design / security-review / implementation-plan feedback).
- **Status:** conflict-driven and deferred — it should run only when a
  downstream artifact exposes a real mismatch or gap. Until built, explain the
  precondition and offer `review`. Do not fabricate a downstream artifact.

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
| blueprint input, architecture exists, accepted change to apply | `update` |
| blueprint input, architecture exists, no specific change | `review` |
| "choose the tech stack / select frameworks / decide database / deployment" | `stack` |
| architecture + ux/test/security findings exposing a mismatch | `reconcile` |
| "review / evaluate / give comments" on an architecture | `review` |

### Ambiguity

- Default to `review` when changing an existing document would be risky.
- Default to `design` when no architecture exists.
- Ask the user only when the mode materially changes the output and cannot be
  inferred from the request or the artifacts present.

## Future flow

```text
blueprint
  -> architecture --mode design
  -> architecture --mode stack
  -> architecture --mode update      (if stack decisions change the architecture)
  -> ux-design                       (future skill)
  -> architecture --mode reconcile   (if UX exposes gaps)
  -> implementation-plan
```

`ux-design` is a future skill and is intentionally **not** built until the
`architecture` skill has a stable `design` mode, a separated `stack` mode, and
at least one architecture output that includes an Experience Architecture
section — otherwise it would consume unstable architecture boundaries.
