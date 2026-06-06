# Prompt 01 — Input Discovery

You are resolving which **architecture design** document this UX-design run will
consume. The architecture design is **required**; the blueprint is strongly
recommended; the tech-stack is optional.

## Inputs

- An optional explicit architecture path and/or blueprint / tech-stack paths.
- The current conversation context.
- The working directory and common design directories.

## Instructions

1. If an explicit architecture path was given and exists, use it.
2. Otherwise search, in order: the current working directory, `./docs`,
   `./design`, `./artifacts`. Preferred pattern `<topic-slug>-architecture-design.md`;
   fallbacks `*-architecture-design.md`, `*architecture*.md`,
   `docs/*architecture*.md`, `design/*architecture*.md`,
   `artifacts/*architecture*.md`.
3. **Rank candidates** by: (1) filename matches `*-architecture-design.md`; (2)
   contains `# Architecture Design`; (3) contains `Experience Architecture`; (4)
   contains a state model section; (5) contains `Interface Contracts`; (6)
   contains `Recommended Next Stages`; (7) topic slug matches a nearby blueprint;
   (8) latest modified time. If one candidate is clearly best, use it; if still
   ambiguous, ASK_USER.
4. Derive `<topic-slug>` by stripping `-architecture-design.md`.
5. Locate optional related files for the same slug:
   `<topic-slug>-product-blueprint.md`, `<topic-slug>-architecture-tech-stack.md`,
   `<topic-slug>-architecture-update.md`.
6. Detect an existing `<topic-slug>-ux-design.md` (a prior run) and pick an
   update mode (new / regenerate / patch / resume).
7. **Failure:** if no architecture design document is found, **STOP** with:
   *"No architecture design document found. Run `architecture --mode design`
   first, or pass the architecture document explicitly."* Do not run UX design
   from a blueprint alone, and do not fabricate an architecture.

## Output

`intermediate/input_resolution.json`:

```json
{
  "architecture_design_path": "<path or null>",
  "blueprint_path": "<path or null>",
  "tech_stack_path": "<path or null>",
  "architecture_update_path": "<path or null>",
  "topic_slug": "<slug or null>",
  "existing_ux_design_path": "<path or null>",
  "update_mode": "new | regenerate | patch | resume",
  "candidates": [{"path": "<p>", "score": 0}],
  "assumptions": ["<recorded assumption, if any>"],
  "needs_user_input": false
}
```

## Validation / failure policy

- Gate: an architecture design was found, or a precise question was posed.
- Failure policy: `stop_if_no_architecture_design`.
