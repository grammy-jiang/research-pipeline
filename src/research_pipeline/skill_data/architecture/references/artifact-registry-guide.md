# Artifact Registry Guide

Load this when producing `<topic-slug>-artifact-registry.md` in `materialize`
mode. The artifact registry tells any downstream agent which files to read and
which files are history.

## Purpose

As a pipeline produces multiple documents — blueprint, architecture design,
tech-stack, update notes, UX design, reconciliation — agents need to know:

```text
Which file is the current canonical source of truth?
Which files have been applied and superseded?
Which files should NOT be read for implementation?
```

The artifact registry answers all three.

## When to create / update

- Created by `architecture --mode materialize` when a canonical architecture
  is first produced.
- Updated whenever `materialize` is re-run after new accepted update notes.
- May also be created manually by a pipeline operator.

## Controlled Status Values

| Status | Meaning |
|---|---|
| `canonical` | Current source of truth for downstream agents |
| `accepted` | Accepted by the pipeline; may have been applied into canonical |
| `applied` | Content incorporated into the canonical artifact |
| `superseded` | Replaced by a canonical version; kept for audit |
| `pending` | Not yet produced; expected from a future stage |
| `deferred` | Stage skipped by routing decision |

## Registry Rules

1. Every pipeline artifact should have an entry.
2. A `canonical` artifact is the one downstream agents should read.
3. Applied update notes must be marked `superseded` after materialization.
4. Only one artifact per role should be `canonical` at a time.
5. `pending` entries tell downstream agents what to expect.

## Artifact Roles (controlled vocabulary)

| Role | Description |
|---|---|
| `product_blueprint` | The product blueprint document |
| `architecture_design` | The initial architecture design (base, pre-canonical) |
| `architecture_canonical` | The merged canonical architecture |
| `architecture_tech_stack` | The tech-stack selection document |
| `architecture_update` | An accepted architecture update note |
| `architecture_review` | An architecture review output |
| `architecture_reconciliation` | A reconciliation finding |
| `ux_design` | The UX design document |
| `security_review` | A security review output |
| `open_question_ledger` | The centralized open-question ledger |
| `artifact_registry` | This file |
| `implementation_plan` | The implementation plan (future stage) |

## Template

See `templates/artifact_registry_template.md` for the full template.

## Example

After a typical `blueprint → design → stack → update → ux-design → reconcile
→ update → materialize` pipeline:

```markdown
## Current Canonical Artifacts

| Artifact Role | Path | Version | Status | Consumer |
|---|---|---:|---|---|
| product_blueprint | pipeline-product-blueprint.md | — | canonical | architecture |
| architecture_canonical | pipeline-architecture-design.v0.4.0.md | v0.4.0 | canonical | implementation-plan |
| ux_design | pipeline-ux-design.md | — | accepted | implementation-plan |
| open_question_ledger | pipeline-open-question-ledger.md | v0.4.0 | canonical | implementation-plan |

## Superseded / Historical Artifacts

| Artifact | Role | Superseded By | Keep For |
|---|---|---|---|
| pipeline-architecture-design.md | architecture_design | pipeline-architecture-design.v0.4.0.md | audit |
| pipeline-architecture-update.md | architecture_update | pipeline-architecture-design.v0.4.0.md | audit |
| pipeline-architecture-update-2.md | architecture_update | pipeline-architecture-design.v0.4.0.md | audit |
| pipeline-architecture-update-3.md | architecture_update | pipeline-architecture-design.v0.4.0.md | audit |
```
