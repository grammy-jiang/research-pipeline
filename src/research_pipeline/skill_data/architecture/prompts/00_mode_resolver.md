# Prompt 00 — Mode Resolver

You are resolving **which architecture mode** this run will execute, before any
blueprint parsing or design work begins. The `architecture` skill is one skill
with internal modes; this pass picks the mode and routes to the right task
graph.

## Modes

| Mode | Status | Builds | Output |
|------|--------|--------|--------|
| `design` (default for a new blueprint) | implemented | the technical architecture from a blueprint | `<topic-slug>-architecture-design.md` |
| `stack` | implemented | the concrete technology selection from a blueprint + architecture design | `<topic-slug>-architecture-tech-stack.md` |
| `review` (default when an architecture already exists) | implemented | a non-mutating quality evaluation (score + blocking/warning/polish) | `<topic-slug>-architecture-review.md` |
| `update` | implemented | applies accepted decisions into an update note (no default overwrite) | `<topic-slug>-architecture-update.md` |
| `reconcile` | implemented | findings + minimal patch plan from downstream feedback (no default patch) | `<topic-slug>-architecture-reconciliation.md` |
| `materialize` | implemented | merges base architecture + accepted update notes into one canonical document | `<topic-slug>-architecture-design.vX.Y.Z.md` |

All six modes are implemented. `design` builds from a blueprint; `stack` selects
the technology; `review` / `update` / `reconcile` operate on an **existing**
architecture and share an artifact resolver (`prompts/resolve_artifacts.md`,
`references/artifact-discovery-guide.md`) that discovers prior outputs by topic
slug when the user passes no filenames. `materialize` consolidates all accepted
update notes into one canonical architecture document ready for implementation
planning. **No-silent-mutation rule:** `review` never mutates the architecture;
`reconcile` never patches it by default; `update` never overwrites the design
document by default; `materialize` stops on conflicts and does not guess
resolutions.

## Inputs

- An optional explicit mode argument (`mode` / `--mode`: design / stack / update
  / review / reconcile / materialize).
- The natural-language request.
- What artifacts are present in context / the working directory (a blueprint,
  an existing `*-architecture-design.md`, an existing `*-architecture-tech-stack.md`,
  a downstream ux/test/security artifact, one or more `*-architecture-update.md`).

## Instructions

1. **Explicit wins.** If a mode argument is given and valid, use it.
2. **Otherwise infer** (see `references/mode-selection-guide.md`):
   - blueprint present, no architecture exists → `design`.
   - "choose the tech stack" / "select frameworks" / "decide the database" /
     "pick the deployment stack" → `stack`.
   - "review / evaluate / score this architecture" / "is this ready for
     implementation?" → `review`.
   - "update architecture based on this stack/decision" / "apply this stack to
     the architecture" → `update`.
   - "UX/test/security found architecture gaps" / "reconcile UX and
     architecture" → `reconcile`.
   - "materialize architecture" / "merge accepted architecture updates" /
     "produce canonical architecture" / "make a final architecture document" /
     "prepare architecture for implementation-plan" / "combine architecture
     update notes" / "generate implementation-ready architecture source of truth"
     → `materialize`.
   - **Bare `architecture` with an existing architecture → `review`** (the safest
     non-mutating default; do **not** default to `update`, `materialize`, or
     silently regenerate `design`).
3. **Ambiguity policy:** prefer the non-mutating mode (`review`) when changing an
   existing document would be risky; default to `design` only when no
   architecture exists; ask the user only when the mode materially changes the
   output and cannot be inferred.
4. **Preconditions:** `stack` needs a blueprint **and** an architecture design.
   `review` / `update` / `reconcile` each need an **architecture design**
   (required) — if none is found, STOP with *"No architecture design document
   found. Run `architecture --mode design` first, or pass the architecture
   document explicitly."* Additionally, `update` needs an accepted update source
   (a tech-stack with `Architecture Update Required? = Yes`, an accepted
   reconciliation, a security-review, a newer blueprint, or explicit user
   decision text — **not** ux-design directly), and `reconcile` needs a downstream
   feedback artifact (primarily a ux-design with an Architecture Feedback
   section). `materialize` needs a base architecture design and at least one
   accepted update note. The shared resolver (`prompts/resolve_artifacts.md`)
   discovers these.
5. **Materialize negative triggers** — do NOT use `materialize` for: reviewing
   architecture quality (use `review`); choosing tech stack (use `stack`); creating
   new architecture design (use `design`); reconciling conflicts (use `reconcile`);
   making new design decisions; writing implementation tasks.

## Output

`intermediate/mode_resolution.json`:

```json
{
  "mode": "design | stack | update | review | reconcile | materialize",
  "explicit": true,
  "reason": "<why this mode>",
  "next_task_graph": "tasks | stack_tasks | review_tasks | update_tasks | reconcile_tasks | materialize_tasks",
  "preconditions_met": true,
  "notes": ["<e.g. update requires an accepted update source; none found>"]
}
```

- `mode == design` → continue with the design `tasks` graph (`prompts/01` onward).
- `mode == stack` → continue with the `stack_tasks` graph (`prompts/stack_01`).
- `mode == review` → continue with the `review_tasks` graph (`prompts/resolve_artifacts.md`
  → `prompts/review_02_assess.md` → `prompts/review_03_final_document.md`).
- `mode == update` → continue with the `update_tasks` graph (`prompts/resolve_artifacts.md`
  → `prompts/update_02_apply_decisions.md` → `prompts/update_03_final_document.md`).
- `mode == reconcile` → continue with the `reconcile_tasks` graph (`prompts/resolve_artifacts.md`
  → `prompts/reconcile_02_detect_conflicts.md` → `prompts/reconcile_03_final_document.md`).
- `mode == materialize` → continue with the `materialize_tasks` graph
  (`prompts/materialize_02_discover_and_order.md` →
  `prompts/materialize_03_apply_patches.md` →
  `prompts/materialize_04_final_document.md`).

## Validation / failure policy

- Gate: a valid mode is selected, or a precise question is posed to the user; the
  selected mode's preconditions (architecture design for review/update/reconcile;
  blueprint + design for stack; base architecture + accepted update notes for
  materialize) are confirmed.
- Failure policy: `default_to_review_if_architecture_exists_else_design_or_ask`.
