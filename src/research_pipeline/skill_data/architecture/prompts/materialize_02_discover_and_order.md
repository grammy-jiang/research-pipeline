# Prompt materialize_02 — Discover, Validate, and Order Inputs

You are discovering the base architecture document and all accepted architecture
update notes, validating them, and producing an ordered patch plan for the
materialize pass.

## Purpose

`materialize` mode consolidates a base architecture plus its accepted update
notes into one canonical architecture document. This pass does the discovery
and ordering work so that the next pass can apply patches safely.

## Search Locations

Search in this order (stop at first non-empty candidate set):

```text
.
./docs
./design
./artifacts
./adr
```

## Required Inputs

### Base Architecture Document

Required:

```text
<topic-slug>-architecture-design.md
```

If no topic slug is known, rank candidates:

```text
1. Topic slug match from context or explicit argument.
2. Filename ending in -architecture-design.md.
3. Presence of "## Architecture Goals and Constraints" heading.
4. Presence of Generation Metadata block (Artifact Type = architecture_design).
5. Highest effective version in Update History.
6. Latest modified timestamp.
7. Not marked Superseded or replaced by a canonical file.
```

If multiple plausible candidates remain after scoring: **ASK_USER** — do not
guess.

If no candidate found: **STOP** with:

```text
No architecture design document found.
Run `architecture --mode design` first, or pass the architecture document
explicitly.
```

### Accepted Update Notes

Discover all accepted update notes matching the topic slug:

```text
<topic-slug>-architecture-update.md
<topic-slug>-architecture-update-2.md
<topic-slug>-architecture-update-N.md
```

Also check for:

```text
<topic-slug>-architecture-tech-stack.md
<topic-slug>-architecture-reconciliation.md   (only if converted to an update note)
```

An update note is **accepted** if ALL of the following hold:

```text
Generation Metadata: Artifact Type = architecture_update
Generation Metadata: Topic Slug matches the base architecture
Generation Metadata: Mode = update
§3 Update Source Documents lists at least one update source
§4 Accepted Decisions Applied table is present and non-empty
§11 (or current) Update Quality-Gate Self-Check overall PASS
No unresolved blocking conflict stated in the document
```

A reconciliation document (`architecture_reconciliation`) is **not** directly
materialized unless it has been explicitly converted into an accepted update
note. If a reconciliation with no corresponding update note is found, note it
as a warning: the user should run `architecture --mode update` first.

## Optional Context Artifacts (for traceability, not direct materialization)

```text
<topic-slug>-product-blueprint.md
<topic-slug>-architecture-tech-stack.md
<topic-slug>-ux-design.md
<topic-slug>-artifact-registry.md
<topic-slug>-open-question-ledger.md
adr/*.md
```

## Update Ordering

Sort accepted update notes into application order:

```text
1. Base architecture version (from Update History or Generation Metadata version field).
2. Effective version numbers in Update History rows.
3. Generated-at timestamps.
4. Filename numeric suffix if other signals are equal (update.md < update-2.md < update-N.md).
```

## Validation

For each accepted update note, validate:

```text
Topic slug matches the base architecture topic slug.
Update source is listed.
§4 Accepted Decisions Applied is non-empty.
Quality-gate self-check: PASS (or WARNING without blocking failures).
No conflict markers that were not resolved.
```

Reject an update note and warn if:

```text
Topic slug does not match.
No accepted decisions listed.
Quality gate FAIL.
Note is explicitly marked superseded.
```

## Instructions

1. Search the locations above for the base architecture document.
2. Rank candidates; select or ask.
3. Extract the topic slug from the selected base architecture.
4. Search for all accepted update notes matching the topic slug.
5. For each candidate update note, check acceptance criteria.
6. Sort accepted notes into application order.
7. For each accepted update note, extract the list of:
   - Decision IDs / patch IDs from §5 Patch Manifest (if present).
   - Target sections from §8 Sections Requiring Update.
   - Patch areas from §9 Architecture Patch Summary (or §6 in older format).
   - Open questions from §13 Remaining Open Questions.
8. Build a section-level patch plan:
   - For each target section, list which update note(s) change it.
9. Detect potential conflicts (preliminary — the next pass confirms):
   - Two update notes both target the same section.
   - Patch type `BREAKING_CHANGE` in any note.
   - A later note references text removed by an earlier note.

## Output

`intermediate/materialize_discovery.json`:

```json
{
  "topic_slug": "<slug>",
  "base_architecture": {
    "path": "<path>",
    "version": "<version>",
    "confidence": "High | Medium | Low"
  },
  "accepted_updates": [
    {
      "path": "<path>",
      "effective_version": "<version>",
      "patch_types": ["CONTRACT_PATCH", "ADR_ONLY"],
      "target_sections": ["§12.1", "§16.2"],
      "applied_in_order": 1
    }
  ],
  "section_patch_plan": {
    "§12.1": ["update-3.md"],
    "§16.2": ["update-3.md"]
  },
  "preliminary_conflicts": [],
  "warnings": [
    "<e.g. reconciliation found without update note — run architecture --mode update first>"
  ],
  "optional_context": {
    "blueprint": "<path or null>",
    "tech_stack": "<path or null>",
    "ux_design": "<path or null>",
    "existing_artifact_registry": "<path or null>",
    "existing_open_question_ledger": "<path or null>"
  }
}
```

## Validation / failure policy

- Gate: base architecture found; at least one accepted update note found; topic
  slug consistent across all inputs; application order unambiguous.
- Warning (non-blocking): reconciliation without update note; optional context
  artifacts missing; patch notes lack Patch Manifest but are still parseable.
- Failure policy: `stop_if_no_base_architecture`; `stop_if_topic_slug_mismatch`;
  `ask_user_if_multiple_candidates`.
