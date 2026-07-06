# Troubleshooting

## Insufficient input

If the source report has no research question, no identifiable
mechanisms/findings, or no evidence, classify it `insufficient`, STOP, and
emit this standardized failure document (do not fabricate a blueprint):

```markdown
# Blueprint Generation Failed: Insufficient Research Input

## Reason

The source report does not contain enough evidence to generate a
research-grounded product blueprint.

## Missing Inputs

- Research question: present / missing
- Mechanisms or findings: present / missing
- Evidence or citations: present / missing
- Gap classification: present / missing

## Recommended Next Step

Run or re-run the `research-pipeline` skill to produce a complete research
report before invoking `blueprint`.
```

## Weak input (proceed with caveats)

`weak` means there is a research question plus either mechanisms or
evidence, but key sections are thin. Proceed, but explicitly mark each
missing area as an assumption or an open question (§17). Do not stop.

## Multi-domain report

Scope to one domain (highest coverage by default) and document the
decision in §2. Ask the user only when domains have similar evidence
coverage and would yield materially different theses. Never emit a single
blueprint that tries to cover all domains.

## Missing sections in the source report

- No Confidence-Graded Findings → infer confidence from paper counts in
  Methodology / Papers Reviewed, and label inferred grades as such.
- No gap classification → infer ACADEMIC vs ENGINEERING from wording
  ("no consensus / open question" → ACADEMIC; "not yet built at scale /
  needs productionizing" → ENGINEERING) and flag the inference.
- No Risk Register → derive risks from contradictions, unvalidated
  assumptions, and security-relevant mechanisms.
- No Round History → treat as a single round; use standard MVP scope.

## Citations missing or malformed

Preserve whatever citation the report uses. If a finding has no citation,
either trace it to the `## References` list or mark the derived capability
"Design hypothesis — requires validation."

## Quality gates keep failing

The composition loop is bounded to **3 revision attempts**. After 3
failures, do **not** deliver the blueprint. Surface the specific failing
gate names, their locations, and the required fixes to the user, then
stop. Common repeat failures:

- Tech-stack leakage → re-check against `borderline-cases.md`.
- Missing Mermaid diagram for the main workflow or the architecture.
- Untraceable capabilities → add a citation or mark as a design
  hypothesis.
- MVP too large → push more ideas to DEFER.

## Amend an existing blueprint with a new decision (no new research report)

Use this playbook when a downstream product, architecture, UX, security, or
planning decision changes the blueprint **without** a replacement research
report. This is a surgical amendment, not research regeneration: preserve source
research findings and citations, change only the accepted product decision and
the sections that depend on it, and record the reason in §2 Update History or
the decision table.

1. Name the new decision and classify it as a product-design decision,
   architecture feedback, UX feedback, security feedback, implementation
   constraint, or user override.
2. Identify the affected load-bearing fact classes and update every dependent
   section before delivery:
   - interaction mode → §3, §8, §9.4/9.5, §10, §16, §18, §19, Appendix A
   - MVP roster → §7, §12, §13, §14, §15, Appendix A
3. Check for stale references to the old decision in the thesis, actors,
   workflows, interaction-mode tables, logical architecture boundaries, roadmap,
   risk rows, evaluation rows, MVP staging, downstream-stage recommendations,
   traceability appendix, and self-check.
4. Preserve research confidence grades unless a new source report is supplied.
   If the amendment is product-only, label it as a product-design decision with
   rationale rather than research-derived evidence.
5. Run the pre-delivery propagation check from `prompts/04_generate_blueprint.md`
   and then re-run the quality gate. If any dependent section is intentionally
   unchanged, say why in that section or Appendix A.

Do not use this playbook to smuggle in new research evidence. A new or materially
changed research report still follows the full update path below.

## Updating an existing blueprint

When a new research report replaces the prior one **and the user asks to
update**:

1. Rename the existing blueprint to
   `<topic-slug>-product-blueprint.<YYYY-MM-DD>.md` as a snapshot.
2. Read the new report; compare new confidence grades against the prior
   §6 decision table.
3. Promote DEFER→ADOPT when evidence upgrades MEDIUM→HIGH; demote
   ADOPT→ADAPT/DEFER when evidence downgrades; promote a closed
   `ACADEMIC` gap (now HIGH-confidence) to a product requirement.
4. Carry forward unresolved open questions.
5. **Regenerate the full blueprint from scratch** — never append. Record
   the change in the §2 Update History table.

Do not auto-update on every report change; require explicit user
confirmation.
