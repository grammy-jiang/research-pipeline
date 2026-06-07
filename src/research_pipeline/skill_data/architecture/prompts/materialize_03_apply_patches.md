# Prompt materialize_03 — Apply Patches and Detect Conflicts

You are applying the accepted architecture update notes — in the order
determined by the discovery pass — into the base architecture sections. If
any conflict is found, stop and produce a materialization-blocked report
instead of a canonical document.

## Inputs

- `intermediate/materialize_discovery.json` (topic slug, base architecture
  path, accepted updates in order, section patch plan, preliminary conflicts).
- The base architecture document itself.
- Each accepted update note (in application order).
- `references/materialization-guide.md`.
- `templates/architecture_canonical_template.md`.
- `templates/materialization_blocked_template.md`.

## Conflict Detection

Before applying any patch, check every conflict class below. **If any is
confirmed: stop. Do not apply patches. Proceed to write the
materialization-blocked report.**

### Conflict Classes

| Class | Description |
|---|---|
| SECTION_CONFLICT | Two updates both mutate the same section in incompatible ways |
| PATCH_TARGET_MISSING | A patch targets a section that does not exist in the base architecture |
| TEXT_ANCHOR_GONE | A patch references specific text that no longer exists in the section |
| CONTRACT_CONFLICT | Two patches change the same interface/data contract differently |
| SUPERSESSION_CONFLICT | A later update explicitly supersedes an earlier one without a clear resolution |
| BREAKING_CHANGE_UNRESOLVED | A `BREAKING_CHANGE` patch type is present without an explicit acceptance marker |

### Conflict Severity

- **BLOCKING**: stop materialization; write blocked report; recommend `architecture
  --mode reconcile`.
- **WARNING**: non-blocking; apply the patch but record the warning in the
  materialization quality gate output.

SECTION_CONFLICT, CONTRACT_CONFLICT, BREAKING_CHANGE_UNRESOLVED are always
BLOCKING. PATCH_TARGET_MISSING is BLOCKING unless the patch type is
`NOTE_ONLY` or `ADR_ONLY`. TEXT_ANCHOR_GONE and SUPERSESSION_CONFLICT are
BLOCKING by default.

## If Blocked

Write `<topic-slug>-architecture-materialization-blocked.md` using
`templates/materialization_blocked_template.md` and stop. Do not produce a
canonical architecture document.

State:

```text
Materialization blocked: <conflict class> — <summary>.
Run architecture --mode reconcile to resolve.
```

## If No Conflict

Apply the accepted update notes in order.

### Application Rules

1. **Apply in order** — earlier updates applied before later ones.
2. **Section-level surgery** — insert, replace, or append the patch content
   into the exact target section of the base architecture.
3. **Remove provisional wording** that has been resolved by an accepted patch
   (e.g. "TBD — see stack mode", "Provisional — subject to stack selection").
4. **Preserve invariants** — blueprint thesis, Product Experience Direction,
   state model, interface contracts, security boundaries must be preserved.
5. **No new design decisions** — if applying a patch would require inventing new
   content, stop and add a WARNING to the quality gate. Record it as an open
   question.
6. **ADR register** — promote accepted ADR candidates; supersede prior ADRs;
   update status from `Proposed` to `Accepted`.
7. **Open Questions** — mark resolved items `Resolved`; add unresolved items
   with owner stage `implementation-plan`.
8. **Handoff Notes** — update §24 and §27 to reflect the canonical state.

### Tracking Applied Patches

For each accepted update note, record:

```text
Update Source: <filename>
Applied Version: <effective version>
Sections Affected: <§N, §M, ...>
Patch Types: <SECURITY_PATCH, CONTRACT_PATCH, ...>
Status: Applied
```

## Output

`intermediate/materialize_applied.md` containing:

```text
TOPIC_SLUG: <slug>
BASE_ARCHITECTURE_VERSION: <version>
CANONICAL_VERSION: <version — the last applied update's effective version>
APPLIED_UPDATES: <list of applied update sources and versions>

---
<Full merged architecture sections as Markdown, preserving section numbering from
the base architecture but with applied patch content inline>

---
OPEN_QUESTIONS_RESIDUAL: <list of remaining unresolved open questions, with source>
WARNINGS: <list of non-blocking warnings>
```

`intermediate/materialize_applied.json`:

```json
{
  "canonical_version": "<version>",
  "base_version": "<version>",
  "applied_updates": [
    {
      "path": "<path>",
      "effective_version": "<version>",
      "sections_affected": ["§12.1", "§16.2"],
      "patch_types": ["CONTRACT_PATCH"],
      "status": "Applied"
    }
  ],
  "warnings": [],
  "residual_open_questions": []
}
```

## Validation / failure policy

- Gate: no conflicts detected before applying; all accepted updates applied in
  order; invariants preserved; no new design decisions invented; ADR register
  updated; open questions updated.
- Failure policy: `stop_and_write_blocked_report` on any BLOCKING conflict;
  `warn_and_continue` on WARNING-class issues; `revise` if patch application
  produces an invalid architecture section (max 3 attempts then surface).
