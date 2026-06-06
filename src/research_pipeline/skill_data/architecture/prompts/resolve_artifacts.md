# Prompt — Resolve Artifacts (shared: review / update / reconcile)

You are discovering and resolving the input artifacts for a `review`, `update`,
or `reconcile` run. This is the **shared** first pass of all three modes. When
the user does not pass filenames, discover the most relevant prior skill/mode
outputs automatically.

## Inputs

- The resolved `mode` (review / update / reconcile) from the mode resolver.
- Optional explicit paths the user passed.
- The current conversation context, working directory, and `docs/` / `design/` /
  `artifacts/`.
- `references/artifact-discovery-guide.md`.

## Instructions

1. **Find the architecture design** (required for all three modes): explicit
   path → context → search the working dir, `docs`, `design`, `artifacts` for
   `<topic-slug>-architecture-design.md` (fallback `*-architecture-design.md`,
   `*architecture*.md`). Derive `<topic-slug>` by stripping
   `-architecture-design.md` (or from in-file metadata).
2. **Find same-topic siblings** by slug and score them (see the guide's scoring
   rules + section markers): product blueprint, architecture tech-stack,
   ux-design, security-review, test-design, implementation-plan, prior
   architecture-update / -reconciliation / -review.
3. **Apply the mode's input priority** (each mode's guide narrows this):
   - `review` — architecture design required; blueprint / stack / ux / update /
     reconciliation / security / test optional (use only if present; never fail
     on a missing optional).
   - `update` — architecture design required; the **accepted update source** is,
     in priority order, an architecture-tech-stack with
     `Architecture Update Required? = Yes`, then an accepted
     architecture-reconciliation, then a security-review with
     architecture-impacting findings, then a newer blueprint, then explicit
     user-provided decision text. **Do not** use ux-design directly as the first
     update source — UX normally goes through `reconcile` first.
   - `reconcile` — architecture design required; the **feedback source** is, in
     priority order, a ux-design with an Architecture Feedback section, then
     security-review, then test-design, then implementation-plan feedback.
4. **Ambiguity → ASK_USER.** If two candidates for a required role score
   closely, ask the user; do not guess when the wrong choice would change
   architecture decisions.
5. **Missing architecture → STOP** with: *"No architecture design document
   found. Run `architecture --mode design` first, or pass the architecture
   document explicitly."* Do not infer architecture from a blueprint alone.
6. Build the **Resolved Input Artifacts** table (Artifact Role · Path ·
   Confidence · Reason) — it is embedded verbatim in §2/§3 of the mode output so
   the reader sees exactly what was consumed.

## Output

`intermediate/resolved_artifacts.json` + the Markdown `## Resolved Input
Artifacts` table:

```json
{
  "mode": "review | update | reconcile",
  "topic_slug": "<slug or null>",
  "architecture_design_path": "<path or null>",
  "roles": {
    "product_blueprint": {"path": "<p or null>", "confidence": "High|Medium|Low|Missing", "reason": "<...>"},
    "architecture_tech_stack": {"path": "<...>", "confidence": "...", "reason": "<...>"},
    "ux_design": {"path": "<...>", "confidence": "...", "reason": "<...>"},
    "security_review": {"path": "<...>", "confidence": "...", "reason": "<...>"},
    "test_design": {"path": "<...>", "confidence": "...", "reason": "<...>"},
    "architecture_reconciliation": {"path": "<...>", "confidence": "...", "reason": "<...>"}
  },
  "primary_source_role": "<e.g. architecture_tech_stack for update, ux_design for reconcile>",
  "needs_user_input": false,
  "stop": false
}
```

## Validation / failure policy

- Gate: the architecture design is found (or a precise question is posed), and a
  Resolved Input Artifacts table is produced.
- Failure policy: `stop_if_no_architecture_design`.
