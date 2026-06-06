# Prompt stack_01 — Resolve Stack Inputs

You are entering `stack` mode. Stack mode selects the concrete technology stack
that satisfies an existing architecture design. First, resolve the two required
inputs.

## Inputs

- An optional explicit architecture-design path and/or blueprint path.
- The current conversation context.
- The working directory and common design directories.

## Instructions

1. Find the **architecture design** (`<topic-slug>-architecture-design.md`):
   explicit path → context → working-dir search (reuse the discovery priority in
   `references/input-discovery.md`, matching `*-architecture-design.md`).
2. Find the **blueprint** for the same topic slug (optional but preferred — it
   carries §9/§19 context and the MVP scope).
3. Derive the `<topic-slug>` from the architecture filename (strip
   `-architecture-design.md`).
4. **Precondition:** stack mode requires an architecture design. If none is
   found, **stop** and recommend running `design` mode first (or, if the user
   asked for the whole chain, run design first and then return here). Do not
   invent an architecture.
5. Detect an existing `<topic-slug>-architecture-tech-stack.md` (a prior stack
   run) and select an update mode (regenerate / patch / compare) if present.

## Output

`intermediate/stack_inputs.json`:

```json
{
  "architecture_design_path": "<path or null>",
  "blueprint_path": "<path or null>",
  "topic_slug": "<slug or null>",
  "existing_tech_stack_path": "<path or null>",
  "update_mode": "new | regenerate | patch | compare",
  "preconditions_met": true,
  "needs_user_input": false,
  "notes": ["<e.g. no architecture design found — recommend running design mode>"]
}
```

## Validation / failure policy

- Gate: a blueprint and an architecture design are found (or a precise question
  is posed); the architecture-design precondition is confirmed.
- Failure policy: `stop_if_no_architecture_design`.
