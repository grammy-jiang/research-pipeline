# Prompt 02 — Detect Existing Architecture

You are deciding whether this is a new architecture or an update to an existing
one. This output is consumed by later passes (context preparation, solution
strategy, ADRs, draft, final document), so be precise.

## Inputs

- `intermediate/input_resolution.json` (topic slug, blueprint path).

## Instructions

1. Look for `<topic-slug>-architecture-design.md` next to the blueprint (and in
   common output directories).
2. Look for an `adr/` directory with prior `ADR-*.md`.
3. If an architecture document exists, read its generation metadata and Update
   History; compare the source-blueprint hash/timestamp and the architecture
   skill version; identify which blueprint sections appear to have changed.
4. Select an **update mode**:

   | Mode | When |
   |---|---|
   | regenerate | blueprint changed substantially |
   | patch | only some sections/clarifications changed |
   | compare | produce a diff-style review, do not change the document |
   | adr-only | only new ADRs are needed for changed decisions |
   | resume | continue from previously unresolved questions |

5. If none exists, this is a **new document**.

## Output

`intermediate/existing_architecture_status.json`:

```json
{
  "exists": false,
  "architecture_path": "<path or null>",
  "existing_adrs": ["adr/ADR-0001-....md"],
  "update_mode": "new | regenerate | patch | compare | adr-only | resume",
  "changed_blueprint_sections": ["<section>"],
  "rationale": "<why this mode>"
}
```

## Validation / failure policy

- Gate: an update mode is selected, or "new document" is asserted.
- Failure policy: `proceed_new_document`.
