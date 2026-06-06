# Prompt 00 — Mode Resolver

You are resolving **which architecture mode** this run will execute, before any
blueprint parsing or design work begins. The `architecture` skill is one skill
with internal modes; this pass picks the mode and routes to the right task
graph.

## Modes

| Mode | Status | Builds | Output |
|------|--------|--------|--------|
| `design` (default) | implemented | the technical architecture from a blueprint | `<topic-slug>-architecture-design.md` |
| `stack` | implemented | the concrete technology selection from a blueprint + architecture design | `<topic-slug>-architecture-tech-stack.md` |
| `update` | implemented via design mode | a patched architecture after an accepted change | updated `…-architecture-design.md` (or an update note) |
| `review` | recognized — deferred | read-only architecture-quality commentary (no rewrite) | review comments (inline) |
| `reconcile` | recognized — deferred | conflict resolution vs a downstream artifact (ux/test/security) | reconciliation note (inline) |

`design` and `stack` are the two fully-specified modes (this is Phase 1 of the
mode split). `update` reuses design mode's existing update-mode machinery
(`regenerate` / `patch` / `compare` / `adr-only` / `resume`, see `prompts/02`).
`review` and `reconcile` are recognized so the vocabulary is stable, but are
**not yet built in detail** — handle them per "Deferred modes" below; never
silently mutate an architecture in those modes.

## Inputs

- An optional explicit mode argument (`mode` / `--mode`: design / stack / update
  / review / reconcile).
- The natural-language request.
- What artifacts are present in context / the working directory (a blueprint,
  an existing `*-architecture-design.md`, an existing `*-architecture-tech-stack.md`,
  a downstream ux/test/security artifact).

## Instructions

1. **Explicit wins.** If a mode argument is given and valid, use it.
2. **Otherwise infer** (see `references/mode-selection-guide.md`):
   - blueprint present, no architecture exists → `design`.
   - blueprint present, an architecture already exists → `update` if the request
     names an accepted change to apply, else `review` (do not silently
     regenerate a document that already exists).
   - "choose the tech stack" / "select frameworks" / "decide the database" /
     "pick the deployment stack" → `stack`.
   - architecture + a ux-design / test-design / security-review artifact that
     exposes a mismatch → `reconcile`.
   - "review this architecture" / "evaluate" / "give comments" → `review`.
3. **Ambiguity policy:** default to `review` when changing an existing document
   would be risky; default to `design` when no architecture exists; ask the user
   only when the mode materially changes the output and cannot be inferred.
4. **Stack-mode precondition:** `stack` needs both a blueprint and an
   architecture design. If no `…-architecture-design.md` is found, say so and
   recommend running `design` first (or run `design` first if the user asked for
   the whole chain).
5. **Deferred modes:** if the resolved mode is `review` or `reconcile`, state
   that it is recognized but not yet a dedicated mode, then either (a) for
   `review`, produce **read-only** commentary by applying the design
   quality-gate lens (`prompts/23`) to the existing architecture without
   rewriting it, or (b) for `reconcile`, explain that it needs a specific
   downstream artifact and a detected conflict, and offer `review` instead. Do
   not fabricate a downstream artifact.

## Output

`intermediate/mode_resolution.json`:

```json
{
  "mode": "design | stack | update | review | reconcile",
  "explicit": true,
  "reason": "<why this mode>",
  "deferred": false,
  "next_task_graph": "design_tasks | stack_tasks | review_readonly | reconcile_blocked",
  "preconditions_met": true,
  "notes": ["<e.g. stack mode requires an architecture design; none found>"]
}
```

- `mode == design` (or `update`) → continue with `prompts/01` onward.
- `mode == stack` → continue with the `stack_tasks` graph (`prompts/stack_01`
  onward).

## Validation / failure policy

- Gate: a valid mode is selected, or a precise question is posed to the user;
  stack mode confirms its architecture-design precondition.
- Failure policy: `default_to_design_if_no_architecture_else_ask`.
