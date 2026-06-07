# Architecture Update Guide (`update` mode)

Load this in `update` mode. Update **applies already-accepted decisions back into
the architecture**. It is not a review mode and not a conflict-analysis mode.

## When to use

Use `update` when a decision has already been accepted, e.g.: a tech stack was
selected; a database / deployment model was confirmed; a security review requires
a specific control; the blueprint changed; the user made an explicit architecture
decision; or a reconciliation produced **accepted** patch recommendations.

## Core discipline (no silent mutation)

- **Default output is an update note**, `<topic-slug>-architecture-update.md`,
  plus an **optional** proposed patched draft
  `<topic-slug>-architecture-design.updated.md`.
- **Do not overwrite** `<topic-slug>-architecture-design.md` by default. Overwrite
  only if the user explicitly asks, the workflow supports safe patching, the
  Update History is updated, the changed sections are listed, and the previous
  architecture can be recovered.
- **Apply only accepted decisions** — never speculative comments or suggestions.

## Default update-source priority

```text
1. Architecture tech-stack document with "Architecture Update Required? = Yes".
2. Architecture reconciliation document with accepted patch recommendations.
3. Security-review document with architecture-impacting findings.
4. Newer product blueprint with changed requirements.
5. Explicit user-provided decision text.
```

**Do not use ux-design directly as the first update source.** UX findings may be
suggestions, conflicts, or requests — they are not automatically accepted
architecture changes. UX-design should normally go through `reconcile` first; the
reconciliation's *accepted* recommendations then become an update source.

## Update-Type Taxonomy

Every update note must classify its patch type in Generation Metadata §1 and in
the §5 Patch Manifest. Allowed values (multi-type allowed, e.g.
`CONTRACT_PATCH + OBSERVABILITY_PATCH`):

| Patch Type | When to use |
|---|---|
| `NONE` | Administrative update, no architecture change |
| `NOTE_ONLY` | Adds a note without changing decisions |
| `ADR_ONLY` | Adds/updates ADRs only |
| `CONTRACT_PATCH` | Changes interface or data contract |
| `SECURITY_PATCH` | Changes security boundary or control |
| `OBSERVABILITY_PATCH` | Changes observability/logging/audit |
| `STRUCTURAL_PATCH` | Changes component or container structure |
| `BREAKING_CHANGE` | Incompatible change — requires explicit acceptance |

See `references/patch-manifest-guide.md` for the full Patch Manifest YAML format.

## Accepted Decisions Applied

```markdown
| Decision | Source | Evidence | Affected Architecture Sections | Applied? |
|---|---|---|---|---|
| ... | architecture-tech-stack | selected storage backend | Data Contracts, Deployment, Security | yes/no |
```

## Sections Requiring Update + Patch Summary

```markdown
| Architecture Section | Change Required | Reason | Priority |
|---|---|---|---|

| Patch Area | Old Assumption | New Decision | Patch Summary |
|---|---|---|---|
```

## Compatibility check

The update must verify that it does not break the architecture's invariants:

```text
Blueprint thesis still preserved.
Product Experience Direction still preserved.
State model remains consistent.
Interface contracts remain consistent.
Security boundaries are not weakened.
Observability remains sufficient.
Recommended Next Stages are still valid or updated.
```

## Feedback Closure Matrix

When this update applies feedback from a downstream artifact (ux-design
Architecture Feedback, security-review findings), produce a §12 Feedback
Closure Matrix:

```markdown
| Feedback Item | Source | Closed By | Status |
|---|---|---|---|
| W-AF1 | ux-design §21 | RC-1/RC-2 | RESOLVED |
| W-AF2 | ux-design §21 | RC-3 | RESOLVED |
| P-AF3 | ux-design §21 | RC-4 | PARTIAL |
```

If not applicable (e.g. a tech-stack update with no downstream feedback),
write `NOT_APPLICABLE — this update does not apply downstream feedback`.

## Update quality gate

The update note now has 13 sections. Quality gate (§13):

```markdown
| Gate | Status | Finding | Required Action |
|---|---|---|---|
| Source architecture found | PASS / FAIL | ... | ... |
| Update source found | PASS / FAIL | ... | ... |
| Only accepted decisions applied | PASS / WARNING / FAIL | ... | ... |
| Patch manifest present | PASS / WARNING / FAIL | ... | ... |
| Changed sections listed | PASS / WARNING / FAIL | ... | ... |
| Unaffected sections preserved | PASS / WARNING / FAIL | ... | ... |
| ADRs / decision register updated | PASS / WARNING / FAIL | ... | ... |
| Update history updated | PASS / WARNING / FAIL | ... | ... |
| Downstream handoffs still valid | PASS / WARNING / FAIL | ... | ... |
| Feedback closure matrix present (when applicable) | PASS / WARNING / FAIL / NOT_APPLICABLE | ... | ... |
| Skill version metadata known | PASS / WARNING / FAIL | ... | ... |
```

Fail if: no architecture document exists; no update source exists; the update
source is only speculative; UX findings are applied directly without
reconciliation; the update would overwrite major architecture decisions without
evidence; or no changed sections are listed.

Warn if: Patch Manifest cannot be produced (materialize will infer); skill
version metadata is UNKNOWN; Feedback Closure Matrix omitted when feedback was
applied.
