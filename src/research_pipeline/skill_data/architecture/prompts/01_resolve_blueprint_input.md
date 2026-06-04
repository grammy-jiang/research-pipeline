# Prompt 01 — Resolve Blueprint Input

You are resolving which product blueprint this architecture run will consume.

## Inputs

- An optional explicit blueprint path argument.
- The current conversation context.
- The current working directory and common design directories.

## Instructions

1. If an explicit path was given and the file exists, use it.
2. Otherwise apply the discovery priority and candidate scoring in
   `references/input-discovery.md` (context → working dir → common dirs →
   scoring).
3. If multiple candidates score closely, ask the user in interactive mode;
   in automatic/hybrid mode pick the highest score and record the assumption.
4. Derive the `<topic-slug>` per `references/input-discovery.md`.
5. If no candidate is found, **stop** and ask the user for the blueprint path.
   Do not fabricate a blueprint.

## Output

`intermediate/input_resolution.json`:

```json
{
  "blueprint_path": "<path or null>",
  "topic_slug": "<slug or null>",
  "candidates": [{"path": "<p>", "score": 0}],
  "assumptions": ["<recorded assumption, if any>"],
  "needs_user_input": false
}
```

## Validation / failure policy

- Gate: a blueprint was found, or a precise question was posed to the user.
- Failure policy: `stop_if_no_blueprint`.
