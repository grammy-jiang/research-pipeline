# Materialization Guide (`materialize` mode)

Load this in `materialize` mode. Materialization **consolidates the base
architecture and all accepted update notes into one canonical architecture
document**. It is a merge operation, not a design operation.

## When to use

Use `materialize` after all enabled design/update/reconcile passes have
completed and before `implementation-plan`:

```text
blueprint
  -> architecture --mode design
  -> architecture --mode stack
  -> architecture --mode update
  -> security-review
  -> architecture --mode update
  -> ux-design
  -> architecture --mode reconcile
  -> architecture --mode update
  -> architecture --mode materialize   ← here
  -> implementation-plan
```

Re-run `architecture --mode materialize` whenever a new accepted update note is
added before implementation planning or a major implementation phase begins.

## Core discipline

- **Merge accepted patches, do not create new design decisions.**
- **Stop on conflicts, do not guess the resolution.**
- **One canonical document: the implementation source of truth.**

```text
Materialize accepted architecture decisions into one canonical architecture
source of truth. Do not make new design decisions. Do not silently resolve
conflicts. Stop and request reconciliation if patches conflict.
```

## What materialize owns

```text
Discovering the base architecture document.
Discovering accepted update notes.
Ordering updates by version / update history.
Checking patch applicability.
Checking for conflicts.
Applying accepted patches into the correct sections.
Removing provisional wording that has been resolved.
Updating ADR status.
Updating open questions.
Updating handoff notes.
Producing a canonical architecture document.
Producing or updating an artifact registry.
Producing or updating an open-question ledger.
Declaring implementation-plan readiness.
```

## What materialize must not own

```text
Creating new architecture decisions.
Choosing a new tech stack.
Inventing new UX flows.
Resolving contradictory patches by guessing.
Changing the blueprint thesis.
Changing MVP scope.
Performing implementation planning.
Writing executable tests.
```

## Accepted update discovery

Only accepted update notes are applied. A note is accepted if it has:

```text
Artifact Type = architecture_update
Topic Slug matches
Mode = update
Update source listed
§4 Accepted Decisions Applied section
Update Quality-Gate Self-Check PASS
No unresolved blocking conflict
```

A reconciliation note is not applied unless it has been converted to an accepted
update note. The correct pattern is:

```text
ux-design
  -> architecture --mode reconcile
  -> architecture --mode update    ← produces an accepted update note
  -> architecture --mode materialize
```

## Update ordering

```text
1. Base architecture version (from Update History or Generation Metadata).
2. Effective version numbers in Update History rows.
3. Generated-at timestamps.
4. Filename numeric suffix if other signals are equal.
```

## Conflict classes

| Class | Severity |
|---|---|
| SECTION_CONFLICT — two updates mutate the same section incompatibly | BLOCKING |
| PATCH_TARGET_MISSING — section targeted by patch does not exist | BLOCKING |
| TEXT_ANCHOR_GONE — patch references text that no longer exists | BLOCKING |
| CONTRACT_CONFLICT — two patches change the same contract differently | BLOCKING |
| SUPERSESSION_CONFLICT — a later update supersedes an earlier one without resolution | BLOCKING |
| BREAKING_CHANGE_UNRESOLVED — BREAKING_CHANGE patch without explicit acceptance | BLOCKING |

Any BLOCKING conflict: stop, write `<topic-slug>-architecture-materialization-blocked.md`,
recommend `architecture --mode reconcile`.

## Output files

**Required (on success):**

```text
<topic-slug>-architecture-design.v<version>.md   (canonical architecture)
<topic-slug>-artifact-registry.md
<topic-slug>-open-question-ledger.md
```

**On conflict block:**

```text
<topic-slug>-architecture-materialization-blocked.md
```

## Materialization quality gate

Fail if:

```text
base architecture is missing;
topic slug mismatch exists;
an accepted update cannot be applied;
two updates conflict;
a patch target section is missing;
canonical architecture cannot be produced safely;
implementation-plan readiness cannot be determined.
```

Warn if:

```text
skill version metadata is UNKNOWN;
some optional context artifacts are missing;
open questions are ownerless;
patch notes lack Patch Manifest but are still parseable;
UX feedback exists but no Feedback Closure Matrix exists in any update note.
```
