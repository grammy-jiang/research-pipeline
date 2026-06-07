# Architecture: <Project Name> (Canonical)

> Skeleton for the `architecture` skill **`materialize` mode** output. Replace
> every `<…>` placeholder. Co-locate with the architecture as
> `<topic-slug>-architecture-design.v<version>.md`.
>
> This is the **canonical, implementation-ready architecture**. It merges the
> base architecture design with all accepted update notes. Do not modify it
> manually — run `architecture --mode materialize` to regenerate.
> See `references/materialization-guide.md`.

## Contents

- [1. Generation Metadata](#1-generation-metadata)
- [2. Source Blueprint Interpretation](#2-source-blueprint-interpretation)
- [3. Clarification Summary](#3-clarification-summary)
- [4. Architecture Goals and Constraints](#4-architecture-goals-and-constraints)
- [5. Solution Strategy](#5-solution-strategy)
- [6. Traditional Software vs AI-Agent Boundary](#6-traditional-software-vs-ai-agent-boundary)
- [7. Tech Stack](#7-tech-stack)
- [8. System Context View](#8-system-context-view)
- [9. Container / Runtime View](#9-container--runtime-view)
- [10. Component View](#10-component-view)
- [11. AI / Skill / MCP Architecture](#11-ai--skill--mcp-architecture)
- [12. Interface Contracts](#12-interface-contracts)
- [13. Data Contracts and Schemas](#13-data-contracts-and-schemas)
- [14. State, Storage, and Data Lifecycle](#14-state-storage-and-data-lifecycle)
- [15. Workflow / Sequence Views](#15-workflow--sequence-views)
- [16. Observability, Logging, Telemetry, and Audit](#16-observability-logging-telemetry-and-audit)
- [17. Security and Trust Boundaries](#17-security-and-trust-boundaries)
- [18. Failure Handling and Recovery](#18-failure-handling-and-recovery)
- [19. Testing and Evaluation Architecture](#19-testing-and-evaluation-architecture)
- [20. Deployment Architecture](#20-deployment-architecture)
- [21. Architecture Decision Records](#21-architecture-decision-records)
- [22. Technical Risks and Trade-offs](#22-technical-risks-and-trade-offs)
- [23. Experience Architecture](#23-experience-architecture)
- [24. Recommended Next Stages and Downstream Handoffs](#24-recommended-next-stages-and-downstream-handoffs)
- [25. Open Questions](#25-open-questions)
- [26. Architecture Quality-Gate Self-Check](#26-architecture-quality-gate-self-check)
- [27. Handoff Notes for Implementation Planning](#27-handoff-notes-for-implementation-planning)
- [28. Applied Updates](#28-applied-updates)
- [29. Superseded Patch Notes](#29-superseded-patch-notes)
- [30. Implementation-Plan Readiness](#30-implementation-plan-readiness)

---

## Update History

| Date | Source | Canonical Version | Change Type | Affected Sections | Notes |
|---|---|---|---|---|---|
| `<date>` | `<base architecture>` | `<v0.1.0>` | initial | all | base design |
| `<date>` | `<update note>` | `<v0.2.0>` | update | `<§N>` | `<note>` |
| `<date>` | `<canonical materialize>` | `<v0.N.0>` | materialize | all | canonical consolidation |

---

## 1. Generation Metadata

| Field | Value |
|---|---|
| Artifact Type | architecture_canonical |
| Topic Slug | `<stable-pipeline-slug>` |
| Project Name | `<Project Name>` |
| Skill Name | architecture |
| Skill Version | `<from manifest.json or UNKNOWN — resolver could not determine this value>` |
| Skill Manifest Version | `<from manifest.json version or UNKNOWN — resolver could not determine this value>` |
| Mode | materialize |
| Generated at | `<date>` |
| Base Version | `<v0.1.0>` |
| Canonical Version | `<v0.N.0>` |
| Applied Updates | `<update-1.md, update-2.md, update-3.md>` |
| Source Architecture | `<path>` |
| Overwrites source? | No — new versioned file |

## Cross-Skill Artifact Contract

> Conforms to the Cross-Skill Artifact Contract (`references/artifact-contract.md`).

### Source Artifacts Consumed

| Artifact Role | Path | Required? | How Used |
|---|---|---:|---|
| architecture_design | `<path>` | yes | base architecture |
| architecture_update | `<path>` | no | accepted patch (v0.2.0) |
| architecture_update | `<path>` | no | accepted patch (v0.3.0) |
| architecture_update | `<path>` | no | accepted patch (v0.N.0) |
| product_blueprint | `<path or —>` | no | traceability context |
| ux_design | `<path or —>` | no | feedback closure context |

### Resolved Input Artifacts

| Artifact Role | Path | Confidence | Reason |
|---|---|---:|---|
| architecture_design | `<path>` | High | required — found by topic slug |
| architecture_update | `<path>` | High | accepted — quality gate PASS |
| architecture_update | `<path or —>` | High / Missing | accepted / not found |

### Contract Field Map

| Contract Field | Where in this document |
|---|---|
| Generation Metadata | §1 |
| Resolved Input Artifacts | §1 Cross-Skill block |
| Decision Register | §21 ADRs |
| Assumptions | §4, §25 Open Questions |
| Open Questions | §25 (centralized; see also open-question ledger) |
| Recommended Next Stage | §30 Implementation-Plan Readiness |
| Quality-Gate Self-Check | §26 + §30 Materialization Quality Gate |

---

## 2. Source Blueprint Interpretation

<Copy from base architecture — with any accepted patches applied.>

---

## 3. Clarification Summary

<Copy from base architecture — with any accepted patches applied.>

---

## 4. Architecture Goals and Constraints

<Copy from base architecture — with any accepted patches applied.>

---

## 5. Solution Strategy

<Copy from base architecture — with any accepted patches applied.>

---

## 6. Traditional Software vs AI-Agent Boundary

<Copy from base architecture — with any accepted patches applied.>

---

## 7. Tech Stack

<Copy from base architecture — with tech-stack update patches applied inline.
If a stack update note was applied, the provisional tech stack is replaced by
the accepted final stack here.>

---

## 8. System Context View

<Copy from base architecture — with any accepted patches applied.>

---

## 9. Container / Runtime View

<Copy from base architecture — with any accepted patches applied.>

---

## 10. Component View

<Copy from base architecture — with any accepted patches applied.>

---

## 11. AI / Skill / MCP Architecture

<Copy from base architecture — with any accepted patches applied.>

---

## 12. Interface Contracts

<Copy from base architecture — with contract patches applied inline (CONTRACT_PATCH
type updates replace or extend individual contract rows/subsections).>

---

## 13. Data Contracts and Schemas

<Copy from base architecture — with any accepted patches applied.>

---

## 14. State, Storage, and Data Lifecycle

<Copy from base architecture — with any accepted patches applied.>

---

## 15. Workflow / Sequence Views

<Copy from base architecture — with any accepted patches applied.>

---

## 16. Observability, Logging, Telemetry, and Audit

<Copy from base architecture — with observability patches applied inline
(OBSERVABILITY_PATCH type updates extend or replace relevant subsections).>

---

## 17. Security and Trust Boundaries

<Copy from base architecture — with security patches applied inline
(SECURITY_PATCH type updates extend or replace relevant subsections).>

---

## 18. Failure Handling and Recovery

<Copy from base architecture — with any accepted patches applied.>

---

## 19. Testing and Evaluation Architecture

<Copy from base architecture — with any accepted patches applied.>

---

## 20. Deployment Architecture

<Copy from base architecture — with any accepted patches applied.>

---

## 21. Architecture Decision Records

<Copy ADR register from base architecture — promote accepted ADR candidates from
update notes; supersede prior ADRs explicitly (never silently overwrite). Update
status from Proposed to Accepted where applicable.>

---

## 22. Technical Risks and Trade-offs

<Copy from base architecture — with any accepted patches applied.>

---

## 23. Experience Architecture

<Copy from base architecture — with any accepted patches applied.>

---

## 24. Recommended Next Stages and Downstream Handoffs

```text
Recommended Next Stage: RUN — implementation-plan
```

| Stage | Routing | Handoff Notes |
|---|---|---|
| implementation-plan | RUN | Consume this canonical architecture + open-question ledger + artifact registry |
| ux-design | DONE | UX feedback applied — see Feedback Closure Matrix in accepted update notes |
| security-review | DONE / OPEN | Security findings applied — see update notes |
| test-design | RUN / DEFER | E2E seeds available in ux-design |

---

## 25. Open Questions

<Centralized from all source artifacts. See also `<topic-slug>-open-question-ledger.md`
for full ledger with owner stages.>

| ID | Question | Source | Owner Stage | Blocks | Status |
|---|---|---|---|---|---|
| OQ-1 | `<question>` | `<source artifact>` | implementation-plan | `<milestone>` | OPEN |

---

## 26. Architecture Quality-Gate Self-Check

<Copy from the base architecture's quality gate — updated to reflect the canonical
state. All PASS entries from the base architecture should be preserved. Any gate
that was WARNING in the base architecture should show its updated status.>

| Gate | Status | Finding | Required Action |
|---|---|---|---|
| Skill contract | PASS / WARNING / FAIL | ... | ... |
| Traceability | PASS / WARNING / FAIL | ... | ... |
| Document completeness (30 sections) | PASS / WARNING / FAIL | ... | ... |
| Tech stack | PASS / WARNING / FAIL | ... | ... |
| AI boundary | PASS / WARNING / FAIL | ... | ... |
| C4 coverage | PASS / WARNING / FAIL | ... | ... |
| Contracts, data, security, observability, failure | PASS / WARNING / FAIL | ... | ... |
| MVP and ADRs | PASS / WARNING / FAIL | ... | ... |
| Metadata consistency | PASS / WARNING / FAIL | ... | ... |
| Skill version metadata known | PASS / WARNING / FAIL | ... | ... |

---

## 27. Handoff Notes for Implementation Planning

```text
Read this canonical architecture.
Do NOT read the superseded update notes (see §29).
Open questions: see <topic-slug>-open-question-ledger.md.
Artifact registry: see <topic-slug>-artifact-registry.md.
```

<Handoff notes from base architecture, updated to reflect canonical state.
Maximum five high-level build-sequencing constraints. No file-by-file ordering
or task tickets here — those belong to implementation-plan.>

---

## 28. Applied Updates

| Update Source | Applied Version | Sections Affected | Patch Types | Status |
|---|---|---|---|---|
| `<update-1.md>` | `v0.2.0` | `§7, §21` | ADR_ONLY, NOTE_ONLY | Applied |
| `<update-2.md>` | `v0.3.0` | `§17.3, §17.9` | SECURITY_PATCH | Applied |
| `<update-3.md>` | `v0.4.0` | `§12.1, §16.2` | CONTRACT_PATCH, OBSERVABILITY_PATCH | Applied |

---

## 29. Superseded Patch Notes

The following update notes are preserved for audit history but **must not** be
used as the implementation source of truth. Read this canonical document instead.

| Artifact | Status | Reason |
|---|---|---|
| `<update-1.md>` | Superseded by canonical `v0.N.0` | Applied |
| `<update-2.md>` | Superseded by canonical `v0.N.0` | Applied |
| `<update-3.md>` | Superseded by canonical `v0.N.0` | Applied |

---

## 30. Implementation-Plan Readiness

### Materialization Quality-Gate Self-Check

| Gate | Status | Finding | Required Action |
|---|---|---|---|
| Base architecture found | PASS / FAIL | ... | ... |
| Accepted update notes found | PASS / WARNING / FAIL | ... | ... |
| Topic slug consistent | PASS / FAIL | ... | ... |
| Updates sorted correctly | PASS / FAIL | ... | ... |
| Patch targets found | PASS / FAIL | ... | ... |
| No conflicting patches | PASS / FAIL | ... | ... |
| All accepted updates applied | PASS / FAIL | ... | ... |
| ADR register current | PASS / WARNING / FAIL | ... | ... |
| Open questions centralized | PASS / WARNING / FAIL | ... | ... |
| Superseded patch notes listed | PASS / WARNING / FAIL | ... | ... |
| Implementation-plan readiness declared | PASS / FAIL | ... | ... |
| Skill version metadata known | PASS / WARNING / FAIL | ... | ... |
| Cross-Skill Artifact Contract Gate | PASS / WARNING / FAIL | ... | ... |

### Readiness Declaration

| Gate | Status | Notes |
|---|---|---|
| Canonical architecture materialized | PASS / FAIL | ... |
| Accepted updates applied | PASS / FAIL | ... |
| Architecture blockers resolved | PASS / WARNING / FAIL | ... |
| Open questions assigned | PASS / WARNING / FAIL | ... |
| UX feedback applied or assigned | PASS / WARNING / FAIL | ... |
| Security findings applied or assigned | PASS / WARNING / FAIL | ... |
| E2E seeds available | PASS / WARNING / FAIL | ... |
| **Ready for implementation-plan** | **YES / NO** | ... |
