# Patch Manifest Guide (`update` mode)

Load this in `update` mode when filling §5 Patch Manifest. A Patch Manifest
makes each update note machine-parseable by `architecture --mode materialize`.

## Purpose

The Patch Manifest is a YAML block embedded in an architecture update note
(`<topic-slug>-architecture-update.md`). It describes, per accepted decision,
exactly which architecture section is targeted, what operation is performed,
and what patch type classifies the change.

Without a Patch Manifest, `materialize` must infer these from prose — less
reliable. With a Patch Manifest, `materialize` can apply patches safely and
detect conflicts programmatically.

## Update-Type Taxonomy

Classify every update note with one or more patch types:

| Patch Type | When to use |
|---|---|
| `NONE` | Administrative update with no architecture change |
| `NOTE_ONLY` | Adds a note or comment to a section without changing decisions |
| `ADR_ONLY` | Adds or updates ADRs without changing architecture sections |
| `CONTRACT_PATCH` | Changes an interface or data contract (§12 or §13) |
| `SECURITY_PATCH` | Changes a security boundary or control (§17) |
| `OBSERVABILITY_PATCH` | Changes observability, logging, or audit configuration (§16) |
| `STRUCTURAL_PATCH` | Changes the component or container structure (§9, §10) |
| `BREAKING_CHANGE` | Changes that are incompatible with existing consumers — requires explicit acceptance marker |

An update note can carry multiple patch types (e.g. `CONTRACT_PATCH +
OBSERVABILITY_PATCH`). Include the overall patch type in Generation Metadata §1.

## Patch Manifest Format

```yaml
patch_manifest:
  - id: <patch-id>                      # e.g. RC-1, STACK-2, SEC-3
    source: <source artifact filename>   # where the decision originated
    target_section: "<section number>"   # e.g. "12.1", "16.2", "17"
    operation: <operation>               # see operations below
    patch_type: <TYPE>                   # from taxonomy above
    blocks_implementation: true/false    # whether this patch blocks implementation
    description: "<short description of what changes>"
    content_summary: "<optional: 1-2 sentences on the new content>"
```

### Operations

| Operation | Description |
|---|---|
| `append_subsection` | Add a new subsection to the target section |
| `append_row` | Add a row to a table in the target section |
| `replace_subsection` | Replace an existing subsection |
| `update_row` | Update an existing table row |
| `update_adrs` | Add or supersede ADRs in §21 |
| `add_note` | Add a prose note to the section |
| `remove_provisional` | Remove provisional wording that is now resolved |

## Example

A UX reconciliation update with two patches:

```yaml
patch_manifest:
  - id: RC-1
    source: architecture-reconciliation.md
    target_section: "12.1"
    operation: append_subsection
    patch_type: CONTRACT_PATCH
    blocks_implementation: true
    description: "Add plugin-to-orchestrator callback contract for progress events"
    content_summary: "Adds §12.1.5 Callback Contract with event schema and error handling"
  - id: RC-2
    source: architecture-reconciliation.md
    target_section: "16.2"
    operation: append_row
    patch_type: OBSERVABILITY_PATCH
    blocks_implementation: false
    description: "Add plugin_progress audit event to §16 audit log catalogue"
    content_summary: "New row: plugin_progress | INFO | every stage callback | structured JSON"
  - id: RC-3
    source: architecture-reconciliation.md
    target_section: "21"
    operation: update_adrs
    patch_type: ADR_ONLY
    blocks_implementation: false
    description: "Promote ADR-007 plugin callback design to Accepted"
```

A tech-stack update:

```yaml
patch_manifest:
  - id: STACK-1
    source: architecture-tech-stack.md
    target_section: "7"
    operation: remove_provisional
    patch_type: NOTE_ONLY
    blocks_implementation: false
    description: "Remove provisional tech stack wording — stack is now finalized"
  - id: STACK-2
    source: architecture-tech-stack.md
    target_section: "21"
    operation: update_adrs
    patch_type: ADR_ONLY
    blocks_implementation: false
    description: "Promote tech-stack ADRs to Accepted status"
```

## When Patch Manifest is missing

If an update note does not have a Patch Manifest, `materialize` will:

1. Attempt to infer patch targets from §8 Sections Requiring Update and
   §9 Architecture Patch Summary.
2. Emit a WARNING in the materialization quality gate.
3. Proceed cautiously — inferred patches are less reliable than explicit ones.

If inference fails, `materialize` will emit a FAIL for `patch_targets_found`
and stop.

## Adding Patch Manifest to existing update notes

If an update note was produced without a Patch Manifest, you may add one
retroactively by running `architecture --mode update` again with the original
sources and asking for a Patch Manifest to be added. The Update History should
record this as a metadata-only revision.
