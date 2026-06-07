# Open Question Ledger: <Project Name>

> Skeleton for `<topic-slug>-open-question-ledger.md`.
> Centralized from all pipeline artifacts by `architecture --mode materialize`.
> See `references/open-question-ledger-guide.md`.

## Contents

- [Generation Metadata](#generation-metadata)
- [Open Questions](#open-questions)
- [Resolved Questions](#resolved-questions)
- [Deferred Questions](#deferred-questions)
- [Ledger Quality Check](#ledger-quality-check)

---

## Generation Metadata

| Field | Value |
|---|---|
| Artifact Type | open_question_ledger |
| Topic Slug | `<stable-pipeline-slug>` |
| Project Name | `<Project Name>` |
| Skill Name | architecture |
| Mode | materialize |
| Generated at | `<date>` |
| Ledger Version | `<v0.N.0>` |
| Source Artifacts | `<architecture-design.md, update-1.md, update-2.md, ux-design.md>` |

---

## Open Questions

Questions that must be resolved before or during implementation planning.

| ID | Question | Source Artifact | Owner Stage | Blocks | Status | Required Resolution |
|---|---|---|---|---|---|---|
| OQ-1 | `<question>` | `<artifact>` | implementation-plan | `<milestone>` | OPEN | `<action>` |
| OQ-2 | `<question>` | `<artifact>` | implementation-plan | `<milestone>` | OPEN | `<action>` |
| OQ-W1 | `<question>` | `<artifact>` | security-review | `<milestone>` | OPEN | `<action>` |

> ID prefix convention: `OQ-N` for numbered, `OQ-WN` for security/warning origin,
> `OQ-UXN` for UX-origin questions.

---

## Resolved Questions

Questions resolved during the design cycle. Retained for audit.

| ID | Question | Source Artifact | Resolved By | Resolution |
|---|---|---|---|---|
| OQ-3 | `<question>` | `<artifact>` | `<update note>` | `<how resolved>` |

---

## Deferred Questions

Questions explicitly deferred to a later stage beyond implementation-plan.

| ID | Question | Source Artifact | Deferred To | Reason |
|---|---|---|---|---|
| OQ-4 | `<question>` | `<artifact>` | `<stage>` | `<reason>` |

---

## Ledger Quality Check

| Check | Status | Notes |
|---|---|---|
| All open questions have source artifact | PASS / FAIL | ... |
| All open questions have owner stage | PASS / FAIL | ... |
| All open questions have blocking status | PASS / FAIL | ... |
| All open questions have required resolution action | PASS / FAIL | ... |
| No ownerless questions | PASS / FAIL | ... |
| All resolved questions have resolution documented | PASS / FAIL | ... |
