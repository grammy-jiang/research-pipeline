# Architecture Materialization Blocked: <Project Name>

> This report is produced by `architecture --mode materialize` when conflicting
> accepted patches prevent safe canonical consolidation.
>
> **Do not proceed to implementation-plan.** Resolve the conflicts first with
> `architecture --mode reconcile`, then rerun `architecture --mode update`,
> then rerun `architecture --mode materialize`.

## Contents

- [1. Generation Metadata](#1-generation-metadata)
- [2. Materialization Attempt Summary](#2-materialization-attempt-summary)
- [3. Conflict Report](#3-conflict-report)
- [4. Recommended Resolution Path](#4-recommended-resolution-path)
- [5. Partial Discovery Summary](#5-partial-discovery-summary)

---

## 1. Generation Metadata

| Field | Value |
|---|---|
| Artifact Type | materialization_blocked |
| Topic Slug | `<stable-pipeline-slug>` |
| Project Name | `<Project Name>` |
| Skill Name | architecture |
| Mode | materialize |
| Generated at | `<date>` |
| Outcome | BLOCKED |

---

## 2. Materialization Attempt Summary

| Item | Value |
|---|---|
| Base architecture | `<path>` (v`<version>`) |
| Accepted updates found | `<N>` |
| Updates ordered | `<list>` |
| Conflicts found | `<N>` |
| Result | **BLOCKED — canonical architecture not produced** |

---

## 3. Conflict Report

| Conflict ID | Conflict Class | Update A | Update B | Section | Description | Severity |
|---|---|---|---|---|---|---|
| C-1 | SECTION_CONFLICT | `<update-2.md>` | `<update-3.md>` | `§12.1` | `<what each patch does differently>` | BLOCKING |
| C-2 | CONTRACT_CONFLICT | `<update-1.md>` | `<update-3.md>` | `§12.3` | `<interface contract changed differently>` | BLOCKING |

### Conflict Detail: C-1

```text
Update A (<update-2.md> v0.3.0):
  <what patch A does to §12.1>

Update B (<update-3.md> v0.4.0):
  <what patch B does to §12.1>

Incompatibility:
  <why these cannot both be applied>
```

---

## 4. Recommended Resolution Path

```text
Materialization blocked: <conflict class summary>.
Run architecture --mode reconcile to resolve.
```

Recommended steps:

1. Run `architecture --mode reconcile` — pass the conflicting update notes and
   the base architecture as inputs. The reconcile mode will produce a conflict
   analysis and a minimal patch plan.
2. Review the reconciliation findings and decide which patch should win.
3. Run `architecture --mode update` — apply the accepted reconciliation decision
   into a new update note (this becomes the authoritative patch for the
   conflicting section).
4. Rerun `architecture --mode materialize` — the new update note supersedes the
   conflicting ones.

---

## 5. Partial Discovery Summary

For audit purposes: what was found before the block.

| Artifact | Path | Confidence | Status |
|---|---|---:|---|
| architecture_design | `<path>` | High | Found — base |
| architecture_update | `<path>` | High | Found — accepted |
| architecture_update | `<path>` | High | Found — conflicting |
| architecture_update | `<path>` | High | Found — conflicting |

> The open-question ledger and artifact registry have **not** been updated
> because materialization was blocked. Rerun after conflict resolution.
