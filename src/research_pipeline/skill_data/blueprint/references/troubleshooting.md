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
