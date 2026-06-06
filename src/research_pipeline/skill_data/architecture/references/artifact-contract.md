# Cross-Skill Artifact Contract

The shared standard every artifact-producing skill in the design chain
(`blueprint → architecture → ux-design → implementation-plan → …`) must follow,
so each Markdown document is both a human-readable report **and** a
machine-readable handoff artifact for the next skill.

> **Canonical + duplicated.** This file is identical across the `blueprint`,
> `architecture`, and `ux-design` skills. Skills install independently (each is
> symlinked to its own directory), so each carries its own copy. Keep the copies
> in sync when the contract changes.

## 1. Purpose

Every major generated document must expose the same **contract fields** somewhere
in the document, using the same **controlled vocabulary**, so downstream skills
can reliably answer: what artifact is this? what generated it? what did it
consume? how were inputs resolved? what decisions/assumptions were made? what is
open? what runs next? did it pass its own gates?

This does not force every document into one identical structure — it requires the
same fields to be present and parseable. If a field is not applicable, include it
and mark `NOT_APPLICABLE — <reason>` rather than omitting it.

## 2. Artifact Type Registry

Every artifact declares its type in Generation Metadata. Controlled types:

```text
research_report
product_blueprint
architecture_design
architecture_tech_stack
architecture_update
architecture_reconciliation
architecture_review
ux_design
implementation_plan
security_review
test_design
release_plan
evaluation_report
```

## 3. Topic Slug Rules

The topic slug is the **pipeline identity**. It is derived from the earliest
stable product/topic name and must remain **stable across the pipeline** unless
the user explicitly renames the project.

Derivation priority when a skill needs the slug:

```text
1. Topic Slug field in the source artifact's Generation Metadata.
2. Source artifact filename (strip the artifact-type suffix).
3. Project name converted to a slug.
4. User-provided explicit slug.
5. Fail or ASK_USER if ambiguous.
```

## 4. Filename Rules

```text
<topic-slug>-<artifact-type-name>.md
```

| Artifact Type | Filename Suffix |
|---|---|
| `research_report` | `research-report.md` |
| `product_blueprint` | `product-blueprint.md` |
| `architecture_design` | `architecture-design.md` |
| `architecture_tech_stack` | `architecture-tech-stack.md` |
| `architecture_update` | `architecture-update.md` |
| `architecture_reconciliation` | `architecture-reconciliation.md` |
| `architecture_review` | `architecture-review.md` |
| `ux_design` | `ux-design.md` |
| `implementation_plan` | `implementation-plan.md` |
| `security_review` | `security-review.md` |
| `test_design` | `test-design.md` |

## 5. Required Metadata (Generation Metadata)

Every artifact includes a Generation Metadata block near the top:

```markdown
## Generation Metadata

| Field | Value |
|---|---|
| Artifact Type | <one of the registry types> |
| Topic Slug | <stable pipeline slug> |
| Project Name | <human name> |
| Generated At | <date> |
| Skill Name | blueprint / architecture / ux-design / … |
| Skill Version | <from manifest.json> |
| Mode | <design / stack / update / reconcile / review, or NOT_APPLICABLE> |
| Output Detail | standard / detailed / compact |
| Source Files | <files consumed, or none> |
```

Do not invent metadata — use `unknown` when unavailable. `Artifact Type` must be
a registry value; `Topic Slug` must match the pipeline slug.

## 6. Source Artifacts Consumed

Every downstream artifact lists what it consumed:

```markdown
## Source Artifacts Consumed

| Artifact Role | Path | Required? | How Used |
|---|---|---:|---|
| product_blueprint | ... | yes/no | Product thesis and MVP scope |
| architecture_design | ... | yes/no | State model, contracts, constraints |
| ux_design | ... | yes/no | User stories and architecture feedback |
```

Use controlled role names (= the artifact-type names). Be factual in "How Used",
not vague.

## 7. Resolved Input Artifacts

Required **when the skill auto-discovers files**. Makes discovery transparent:

```markdown
## Resolved Input Artifacts

| Candidate | Selected? | Confidence | Reason |
|---|---:|---:|---|
| ./<slug>-architecture-design.md | yes | High | Exact topic slug and expected title |
| ./old-architecture-design.md | no | Low | Different topic slug |
```

If every input was explicitly supplied: `NOT_APPLICABLE — all input artifacts
were explicitly supplied by the user.`

## 8. Decision Register

Required when the artifact makes decisions:

```markdown
## Decision Register

| Decision | Status | Source | Reason | Revisit Trigger |
|---|---|---|---|---|
```

Status ∈ controlled values (§13). Downstream skills must not treat every
statement as accepted — especially technology choices, data-egress policy, MCP
exposure, human-review workflow, security assumptions, and deployment targets.

## 9. Assumptions

Listed **separately from decisions**:

```markdown
## Assumptions

| Assumption | Source | Confidence | Risk if Wrong | Review Trigger |
|---|---|---:|---|---|
```

Source ∈ {explicit_user_input, upstream_artifact, inferred_from_context,
default_policy, missing_information}. High-risk assumptions (content leaving the
local boundary, human review is rare, users are technical, MCP clients are
trusted, CLI output is sufficient for MVP) must be flagged and become
`ASK_USER` / downstream review triggers when necessary.

## 10. Open Questions

```markdown
## Open Questions

| Question | Owner / Next Stage | Blocks Next Stage? | Recommended Action |
|---|---|---:|---|
```

Each open question has an owner, a blocking status, and a recommended action.

## 11. Recommended Next Stage

```markdown
## Recommended Next Stage

| Stage | Decision | Reason | Required Input |
|---|---|---|---|
| architecture --mode design | RUN / SKIP / DEFER / ASK_USER | ... | ... |
| ux-design | RUN / SKIP / DEFER / ASK_USER | ... | ... |
| implementation-plan | RUN / SKIP / DEFER / ASK_USER | ... | ... |
```

Use only the controlled stage-decision values (§13). The `Required Input` column
names what the next stage needs (drives default artifact discovery). Never use
vague values (maybe / consider / probably / nice to have).

## 12. Quality-Gate Self-Check

Every artifact ends with a quality-gate self-check (PASS / WARNING / FAIL /
NOT_APPLICABLE) **and** the shared contract gate:

```markdown
## Cross-Skill Artifact Contract Gate

| Gate | Status | Finding | Required Action |
|---|---|---|---|
| Generation metadata present | PASS / WARNING / FAIL | ... | ... |
| Topic slug present and stable | PASS / WARNING / FAIL | ... | ... |
| Source artifacts listed | PASS / WARNING / FAIL | ... | ... |
| Resolved input artifacts recorded (when discovery is used) | PASS / WARNING / FAIL / NOT_APPLICABLE | ... | ... |
| Decisions and assumptions separated | PASS / WARNING / FAIL | ... | ... |
| Open questions assigned to a next stage | PASS / WARNING / FAIL | ... | ... |
| Recommended next stage present | PASS / WARNING / FAIL | ... | ... |
```

Common fail conditions: topic slug missing; source artifacts not listed; a
required input artifact missing; a high-impact assumption hidden; quality gate
absent; downstream handoff missing; decision status ambiguous.

## 13. Controlled Vocabulary

```text
Artifact types     : research_report, product_blueprint, architecture_design,
                     architecture_tech_stack, architecture_update,
                     architecture_reconciliation, architecture_review, ux_design,
                     implementation_plan, security_review, test_design,
                     release_plan, evaluation_report
Decision status    : accepted, provisional, deferred, rejected, superseded,
                     requires_user_confirmation
Stage decision     : RUN, SKIP, DEFER, ASK_USER
Confidence         : High, Medium, Low
Quality gate status: PASS, WARNING, FAIL, NOT_APPLICABLE
Severity           : Blocking, Warning, Polish
Assumption source  : explicit_user_input, upstream_artifact, inferred_from_context,
                     default_policy, missing_information
```

## 14. Skill-Specific Requirements

### blueprint (`product_blueprint`)
Generation Metadata · Source Artifacts Consumed · Decision Register · Assumptions
· Open Questions · Recommended Next Stages · Product Experience Direction ·
Quality-Gate Self-Check. (Resolved Input Artifacts only if it performs file
discovery.)

### architecture (all modes)
Generation Metadata · Source Artifacts Consumed · **Resolved Input Artifacts** ·
Decision Register · Assumptions · Open Questions · Recommended Next Stage ·
Quality-Gate Self-Check. Mode → type: `design → architecture_design`,
`stack → architecture_tech_stack`, `update → architecture_update`,
`reconcile → architecture_reconciliation`, `review → architecture_review`.

### ux-design (`ux_design`)
Generation Metadata · Source Artifacts Consumed · Resolved Input Artifacts · UX
Assumptions · User Stories · E2E Scenario Seeds · Architecture Feedback /
Required Architecture Updates · Recommended Next Stage · Quality-Gate Self-Check.

### implementation-plan / security-review / test-design (future)
Must comply from day one: Generation Metadata · Source Artifacts Consumed ·
Resolved Input Artifacts · (Task Breakdown / Test Task Mapping for
implementation-plan) · Decision Register · Assumptions · Open Questions ·
Recommended Next Stage · Quality-Gate Self-Check.

## 15. Alignment, Not Duplication

A skill's template may already express these fields under its own section names
(e.g. architecture's §3 Clarification Summary is a decision register; §4.9 is its
assumptions). **Do not duplicate** — map existing sections to the contract fields
(a "Contract Field Map" table is a good way) and add only what is genuinely
missing (the identity metadata, a Source Artifacts Consumed table, and the
Cross-Skill Artifact Contract Gate). The goal is one parseable interface, not
boilerplate.
