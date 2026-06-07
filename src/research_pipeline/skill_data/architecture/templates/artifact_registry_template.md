# Artifact Registry: <Project Name>

> Skeleton for `<topic-slug>-artifact-registry.md`.
> Updated by `architecture --mode materialize`.
> See `references/artifact-registry-guide.md`.

## Contents

- [Generation Metadata](#generation-metadata)
- [Current Canonical Artifacts](#current-canonical-artifacts)
- [Superseded / Historical Artifacts](#superseded--historical-artifacts)
- [Pipeline Stage Summary](#pipeline-stage-summary)

---

## Generation Metadata

| Field | Value |
|---|---|
| Artifact Type | artifact_registry |
| Topic Slug | `<stable-pipeline-slug>` |
| Project Name | `<Project Name>` |
| Skill Name | architecture |
| Mode | materialize |
| Generated at | `<date>` |
| Registry Version | `<v0.N.0>` |

---

## Current Canonical Artifacts

These files are the **current source of truth**. Downstream agents should read
these files and ignore the superseded artifacts below.

| Artifact Role | Path | Version | Status | Consumer |
|---|---|---:|---|---|
| product_blueprint | `<path>` | `<version>` | canonical | architecture, ux-design |
| architecture_canonical | `<topic-slug>-architecture-design.vN.md` | `<vN.0.0>` | canonical | implementation-plan |
| architecture_tech_stack | `<path or —>` | `<version or —>` | accepted / applied | architecture materialize |
| ux_design | `<path or —>` | `<version or —>` | accepted | implementation-plan, test-design |
| security_review | `<path or —>` | `<version or —>` | accepted / applied | implementation-plan |
| open_question_ledger | `<topic-slug>-open-question-ledger.md` | `<vN.0.0>` | canonical | implementation-plan |
| artifact_registry | `<topic-slug>-artifact-registry.md` | `<vN.0.0>` | canonical | implementation-plan |

---

## Superseded / Historical Artifacts

These files have been incorporated into the canonical architecture. They are
preserved for audit history but **must not** be used as the primary source of
truth for implementation planning.

| Artifact | Role | Superseded By | Keep For |
|---|---|---|---|
| `<topic-slug>-architecture-design.md` | architecture_design (base) | `<canonical path>` | audit |
| `<topic-slug>-architecture-update.md` | architecture_update | `<canonical path>` | audit |
| `<topic-slug>-architecture-update-2.md` | architecture_update | `<canonical path>` | audit |
| `<topic-slug>-architecture-update-3.md` | architecture_update | `<canonical path>` | audit |
| `<topic-slug>-architecture-reconciliation.md` | architecture_reconciliation | `<canonical path>` | audit |

---

## Pipeline Stage Summary

| Stage | Output Artifact | Status | Notes |
|---|---|---|---|
| blueprint | `<blueprint path>` | canonical | source of truth for product intent |
| architecture design | `<design path>` | superseded | incorporated into canonical |
| architecture stack | `<stack path>` | applied | ADRs promoted in canonical |
| architecture update (N) | `<update paths>` | superseded | all applied in canonical |
| ux-design | `<ux path>` | accepted | feedback closure tracked in update notes |
| architecture materialize | `<canonical path>` | canonical | implementation-plan source |
| implementation-plan | — | pending | next stage |
