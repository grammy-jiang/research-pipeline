# Prompt reconcile_02 — Detect Conflicts and Gaps

You are comparing the architecture with downstream feedback artifacts to detect
gaps, conflicts, and missing architecture support. `reconcile` mode is
conflict-driven and **does not patch the architecture** — it produces findings.

## Inputs

- `intermediate/resolved_artifacts.json` (architecture design + feedback
  source(s)).
- The architecture design and the feedback artifact(s) — primarily a ux-design
  with an Architecture Feedback section; also security-review / test-design /
  implementation-plan feedback.
- `references/architecture-reconciliation-guide.md`,
  `templates/architecture_reconciliation_template.md`.

## Instructions

1. **Consume the feedback** — if a ux-design exists, treat its §21 Architecture
   Feedback as the primary input. Map each feedback item / conflict to the
   architecture sections it affects.
2. **Detect** missing user-visible/internal states, missing transitions, missing
   retry/cancel operations, missing progress/audit events, missing review-artifact
   schema, missing permission boundary, missing data-egress confirmation, missing
   API/CLI/MCP output fields, UX-flow-vs-state-machine incompatibilities,
   impossible E2E scenarios, impossible security controls, blocked implementation
   tasks.
3. **§4 Conflict Summary** — one row per finding: Finding · Source Artifact ·
   Severity (Blocking/Warning/Polish) · Architecture Gap · Recommended Change ·
   Requires Update Mode?. **Separate genuine conflicts from enhancements.**
4. **§5 Missing Architecture Support** — map each downstream need to the missing
   architecture support, affected section, and the **minimal** suggested fix.
5. **§6 Impact Analysis** — for high-severity findings, analyse the effect on the
   state model / contracts / observability / security / surfaces, and decide
   whether the architecture or the downstream artifact should change.
6. **Do not blindly accept** the downstream artifact: if a finding contradicts
   the blueprint thesis or Product Experience Direction, flag it as a conflict to
   resolve, not an automatic change.
7. **§7 Recommended Architecture Changes** + **§8 Minimal Patch Plan** — the
   smallest coherent change set; this is a plan, not a patch.

## Output

`intermediate/reconcile_findings.md` (the §4–§8 content).

## Validation / failure policy

- Gate: findings are traceable to the feedback artifact; conflicts separated from
  enhancements; changes are minimal; nothing is silently rewritten.
- Failure policy: `revise`.
