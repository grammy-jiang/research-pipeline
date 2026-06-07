# Prompt materialize_04 — Final Canonical Architecture Document

You are assembling the final canonical architecture document, the artifact
registry, and the open-question ledger, then running the materialization
quality gate.

## Inputs

- `intermediate/materialize_applied.md` — the merged architecture content.
- `intermediate/materialize_applied.json` — applied updates summary.
- `intermediate/materialize_discovery.json` — discovery metadata.
- `templates/architecture_canonical_template.md`.
- `templates/artifact_registry_template.md`.
- `templates/open_question_ledger_template.md`.
- `references/materialization-guide.md`.
- `references/artifact-registry-guide.md`.
- `references/open-question-ledger-guide.md`.
- `references/artifact-contract.md` (Cross-Skill Artifact Contract).

## Instructions

### Step 1 — Compose the Canonical Architecture Document

Filename: `<topic-slug>-architecture-design.v<canonical_version>.md`

The document must contain every section from the base architecture (all 27
sections from the `design` mode output) with accepted patches applied inline.
It must NOT paste raw update notes as appendices.

Add three required extra sections **after the existing 27 sections**:

**§28 Applied Updates**

```markdown
## Applied Updates

| Update Source | Applied Version | Sections Affected | Patch Types | Status |
|---|---|---|---|---|
| <filename> | <version> | <§N, §M> | <types> | Applied |
```

**§29 Superseded Patch Notes**

```markdown
## Superseded Patch Notes

The following update notes are now superseded by this canonical document.
Do not use them as the implementation source of truth.

| Artifact | Status | Reason |
|---|---|---|
| <filename> | Superseded by canonical v<version> | Applied |
```

**§30 Implementation-Plan Readiness**

```markdown
## Implementation-Plan Readiness

| Gate | Status | Notes |
|---|---|---|
| Canonical architecture materialized | PASS / FAIL | ... |
| Accepted updates applied | PASS / FAIL | ... |
| Architecture blockers resolved | PASS / WARNING / FAIL | ... |
| Open questions assigned | PASS / WARNING / FAIL | ... |
| UX feedback applied or assigned | PASS / WARNING / FAIL | ... |
| Security findings applied or assigned | PASS / WARNING / FAIL | ... |
| E2E seeds available | PASS / WARNING / FAIL | ... |
| Ready for implementation-plan | YES / NO | ... |
```

Declare explicitly:

```text
Ready for implementation-plan: YES
```

or:

```text
Ready for implementation-plan: NO — blocking issues:
- ...
```

### Step 2 — Generation Metadata Block

Emit in §1 of the canonical architecture (replacing the base architecture's
§1 metadata):

| Field | Value |
|---|---|
| Artifact Type | architecture_canonical |
| Topic Slug | `<stable-slug>` |
| Skill Name | architecture |
| Skill Version | `<from manifest.json or UNKNOWN — resolver could not determine this value>` |
| Mode | materialize |
| Generated at | `<date>` |
| Base Version | `<version>` |
| Canonical Version | `<version>` |
| Applied Updates | `<list of source filenames>` |
| Overwrites base architecture? | No (canonical version is a new file) |
| Source architecture | `<path>` |
| Skill Manifest Version | `<from manifest.json version or UNKNOWN — resolver could not determine this value>` |

If any field cannot be determined, write:

```text
UNKNOWN — resolver could not determine this value
```

Never write `unknown` without this explanation. The quality gate will warn on
any `UNKNOWN` field.

### Step 3 — Artifact Registry

Create or update `<topic-slug>-artifact-registry.md` using
`templates/artifact_registry_template.md`.

The registry must list:

**Current Canonical Artifacts:**

| Artifact Role | Path | Version | Status | Consumer |
|---|---|---:|---|---|
| product_blueprint | ... | ... | canonical | architecture, ux-design |
| architecture_canonical | `<canonical path>` | `<version>` | canonical | implementation-plan |
| ux_design | ... | ... | accepted | implementation-plan |
| open_question_ledger | ... | ... | canonical | implementation-plan |

**Superseded / Historical Artifacts:**

| Artifact | Superseded By | Keep For |
|---|---|---|
| `<update note path>` | `<canonical path>` | audit |

Mark every applied update note as superseded.

### Step 4 — Open-Question Ledger

Create or update `<topic-slug>-open-question-ledger.md` using
`templates/open_question_ledger_template.md`.

Collect all open questions from:

```text
Base architecture §25 Open Questions
Each accepted update note §13 Remaining Open Questions (or §10 in older format)
Any warnings from the apply pass
```

Every open question must have:

```text
ID
Question text
Source artifact
Owner stage (implementation-plan / security-review / etc.)
Blocking status (blocks which milestone)
Status (OPEN / RESOLVED / DEFERRED)
Required resolution action
```

No open question should remain ownerless.

### Step 5 — Materialization Quality Gate

```markdown
## Materialization Quality-Gate Self-Check

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
| Skill version metadata known | PASS / WARNING / FAIL | UNKNOWN fields warn | ... |
| Cross-Skill Artifact Contract Gate | PASS / WARNING / FAIL | ... | ... |
```

**Fail conditions** (→ do not produce the canonical document):

```text
base architecture is missing;
topic slug mismatch exists;
an accepted update cannot be applied;
two updates conflict;
a patch target section is missing;
canonical architecture cannot be produced safely;
implementation-plan readiness cannot be determined.
```

**Warning conditions** (→ produce the canonical document with warnings):

```text
skill version metadata is UNKNOWN;
some optional context artifacts are missing;
open questions are ownerless;
patch notes lack Patch Manifest but are still parseable;
UX feedback exists but no Feedback Closure Matrix exists in any update note.
```

## Output Files

**Required:**

```text
<topic-slug>-architecture-design.v<canonical_version>.md
<topic-slug>-artifact-registry.md
<topic-slug>-open-question-ledger.md
```

**If blocked:**

```text
<topic-slug>-architecture-materialization-blocked.md
```

**No output if gate fails:**

If any FAIL gate is found: do not write the canonical architecture. Write only
the blocked report and stop.

## Output Discipline

After writing all files, emit a short summary (do not write a separate file):

```text
Canonical architecture: <filename> (v<version>)
Accepted updates applied: N
Open questions centralized: M (OPEN: X, RESOLVED: Y)
Implementation-plan ready: YES / NO
Artifact registry: <filename>
Open-question ledger: <filename>

Next: implementation-plan — read the canonical architecture, artifact registry,
and open-question ledger.
```

## Validation / failure policy

- Gate: all materialization quality gates pass; canonical document has all 27
  original sections + §28 Applied Updates + §29 Superseded Patch Notes + §30
  Implementation-Plan Readiness; no raw patch notes pasted verbatim; artifact
  registry marks superseded update notes; open-question ledger has all open
  questions with owner stages.
- Failure policy: `stop_and_write_blocked_report` on FAIL gates;
  `warn_and_produce` on WARNING gates; `revise_max_3_then_stop` if canonical
  document composition fails validation.

## Cross-Skill Artifact Contract Compliance

Comply with `references/artifact-contract.md`. The canonical architecture must:

- Carry `Artifact Type = architecture_canonical` and a stable `Topic Slug`.
- Include a `Source Artifacts Consumed` section.
- Carry `Resolved Input Artifacts` (auto-discovered inputs).
- Separate decisions (applied), assumptions, and open questions.
- Include a `Recommended Next Stage: RUN — implementation-plan`.
- Include the `Materialization Quality-Gate Self-Check` with the
  `Cross-Skill Artifact Contract Gate`.
