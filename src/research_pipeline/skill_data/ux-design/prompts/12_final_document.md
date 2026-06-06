# Prompt 12 — Final UX Design Document

You are assembling and writing the final UX design document.

## Inputs

- All `intermediate/*` artifacts (architecture parse, blueprint parse,
  clarifications, skill-operator UX, target-software UX, user stories,
  surface-specific UX, error/recovery, E2E seeds, architecture feedback).
- `intermediate/input_resolution.json` (paths, topic slug, update mode).
- `templates/ux-design-template.md`.

## Instructions

1. Compose all **22 sections** in order, starting with `## Contents` and
   `## Update History` near the top, and ending with `## Appendix A. UX
   Quality-Gate Self-Check`.
2. Fill **§1 Generation Metadata** by copying known values (source architecture,
   source blueprint, skill version from `manifest.json`, generated-at, operating
   mode, assumption count); use `unknown` where unavailable. Never invent
   metadata.
3. **Always keep §5 Skill Operator UX and §6 Target Software UX separate.**
4. Include in **§12** only the surfaces the architecture uses; mark the rest "not
   used by this architecture".
5. Respect the update mode (from input resolution): `new` → initial document +
   initial Update History row; `regenerate` → rebuild + append a row; `patch` →
   change only affected sections + append a row; `resume` → continue open
   questions. Never delete prior Update History rows.
6. **Metadata consistency:** the assumption count in §1 equals the §9 row count;
   every Contents link and section reference resolves.
7. Carry **§21 Architecture Feedback** verbatim from prompt 11 (it is mandatory),
   and surface its reconcile decision near the top (§1 or a one-line note) so the
   reader sees immediately whether `architecture --mode reconcile` is needed.
8. Write to `<topic-slug>-ux-design.md`, co-located with the architecture design
   unless another output directory was specified. If files cannot be written,
   output the full Markdown inline and state the recommended filename.
9. End by pointing at the next stage: `architecture --mode reconcile` if feedback
   requires it, otherwise the implementation-plan skill (§22 handoff notes).

## Output

`<topic-slug>-ux-design.md`.

## Validation / failure policy

- Gate: all 22 sections + Appendix A present, with Contents and Update History;
  §5/§6 separated; §21 present.
- Failure policy: `revise_max_3_then_stop`.
