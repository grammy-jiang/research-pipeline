# Artifact Discovery Guide (shared by review / update / reconcile)

Load this for `prompts/resolve_artifacts.md`. The `review`, `update`, and
`reconcile` modes all operate on an **existing** architecture and its sibling
artifacts. When the user does not pass filenames, this shared resolver discovers
and consumes the most relevant prior skill/mode outputs automatically.

## Required vs optional

For **all three** modes the architecture design document is **required**. The
other artifacts are optional and mode-dependent (see each mode's guide). Never
infer architecture from a blueprint alone in review / update / reconcile.

## Search locations (in order)

```text
.            (current working directory)
./docs
./design
./artifacts
```

Optional future locations (only if the minimum set yields nothing):
`./reports`, `./outputs`, `./dist`.

## Expected artifact file patterns

```text
<topic-slug>-product-blueprint.md
<topic-slug>-architecture-design.md
<topic-slug>-architecture-tech-stack.md
<topic-slug>-architecture-update.md
<topic-slug>-architecture-reconciliation.md
<topic-slug>-architecture-review.md
<topic-slug>-ux-design.md
<topic-slug>-security-review.md
<topic-slug>-test-design.md
<topic-slug>-implementation-plan.md
```

Fallback patterns: `*blueprint*.md`, `*architecture-design*.md`,
`*architecture*tech*stack*.md`, `*architecture-update*.md`,
`*architecture-reconciliation*.md`, `*architecture-review*.md`, `*ux-design*.md`,
`*security-review*.md`, `*test-design*.md`, `*implementation-plan*.md`.

## Topic-slug derivation

Derive the slug from the **architecture design** document first: strip
`-architecture-design.md` from its filename
(`llm-agent-translation-system-architecture-design.md` →
`llm-agent-translation-system`). Then find same-topic siblings
(`…-product-blueprint.md`, `…-architecture-tech-stack.md`, `…-ux-design.md`, …).
If the filename has no obvious slug, use metadata inside the file (Project Name,
Topic Slug, Source Blueprint, Generated From). If no slug can be derived, fall
back to the newest suitable file **only** if there is a single clear candidate.

## Candidate scoring

When multiple candidates exist for a role, rank by:

```text
1. Same topic slug as the architecture design.
2. Expected filename suffix.
3. Contains the expected document title.
4. Contains the expected key sections (see markers below).
5. Latest modified time.
6. Same directory as the architecture design.
7. Referenced by metadata in the architecture design.
```

### Expected section markers

| Artifact Type | Useful Markers |
|---|---|
| Architecture design | `# Architecture Design`, `Experience Architecture`, `State Model`, `Interface Contracts`, `Data Contracts` |
| Tech stack | `# Architecture Tech Stack`, `Technology Decision Drivers`, `Architecture Update Required?` |
| UX design | `# UX Design`, `Architecture Feedback / Required Architecture Updates`, `E2E Scenario Seeds` |
| Security review | `Security Review`, `Threat`, `Trust Boundary`, `Data Egress`, `Mitigation` |
| Test design | `Test Design`, `E2E`, `Scenario`, `Acceptance Criteria`, `Test Matrix` |
| Implementation plan | `Implementation Plan`, `Tasks`, `Milestones`, `Test Tasks`, `Dependencies` |

If candidates remain ambiguous, **ASK_USER** — do not guess when selecting the
wrong artifact would change architecture decisions.

## Resolved Input Artifacts table

Every review / update / reconcile output must include this table (built by the
resolver) so the reader knows exactly what was consumed:

```markdown
## Resolved Input Artifacts

| Artifact Role | Path | Confidence | Reason |
|---|---|---:|---|
| architecture_design | ... | High | matched topic slug and expected title |
| product_blueprint | ... | Medium | same topic slug |
| architecture_tech_stack | ... | High | newest matching stack document |
| ux_design | ... | High | contains architecture feedback section |
| security_review | ... | Missing | not found |
| test_design | ... | Missing | not found |
```

## Missing-architecture failure

If no architecture design document is found, **STOP** with:

```text
No architecture design document found.
Run `architecture --mode design` first, or pass the architecture document explicitly.
```

Missing **optional** artifacts never fail the run — record them as `Missing` in
the table and continue.
