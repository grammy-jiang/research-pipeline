# Iterative Gap-Closure Rounds

Every research run iterates up to **4 rounds**, stopping early when the
report has no remaining gaps. This applies to **all** goals — pure
literature reviews, system-building tasks, surveys — not only
system-design work. Gaps are the visible signal; closing them
systematically is what raises a one-shot literature dump into a research
report that actually converges.

## Round definition

A round is one full pipeline cycle that **replaces** the prior report
(resume-on-top from `SKILL.md`):

1. Read the prior report's gap sections (both academic and engineering).
2. Translate each open gap into a targeted search seed.
3. Run the full pipeline end-to-end for that seed (plan → search →
   screen → download → convert → extract → summarize).
4. Regenerate the final report from scratch, integrating prior
   paper IDs (via the global index + `research-pipeline expand
   --paper-ids ...`) with the new evidence. The new report is the
   single source of truth for the next round.

The prior report is snapshot-renamed with the date
(`<topic-slug>-research-report.<YYYY-MM-DD>.md`) before regeneration,
but is never referenced from the new report body.

## Per-round procedure

1. **Extract open gaps** from the most recent report:
   - Prefer the report's own `Research Gaps` /
     `Unresolved Questions` / `Assumption Map` / `Risk Register`
     sections.
   - Classify each gap as `ACADEMIC` (needs new papers) or
     `ENGINEERING` (needs implementation knowledge / best practices).
   - Drop gaps that are truly out-of-scope or that the user has
     explicitly de-prioritized.

2. **Handle engineering gaps locally**:
   - Fill using implementation knowledge, documentation, and reliable
     sources. Do **not** spin up a new pipeline run for these.
   - Record the resolution inline in the corresponding theme of the
     regenerated report (not a separate "answers" section).

3. **Handle academic gaps with a new pipeline iteration**:
   - For each academic gap (or a small merged cluster of related
     gaps), derive a narrower, more specific research question — one
     that the broader parent query did not answer.
   - Feed that question, together with the prior paper IDs as
     `expand --paper-ids` seeds and any prior
     `must_terms`/`nice_terms` as variants, into a new run:
     ```bash
     research-pipeline plan "<gap-specific topic>" --config CFG
     # edit query_plan.json if needed (tighter must_terms,
     #   synonym-rich variants)
     research-pipeline search --run-id <NEW_RUN_ID> --source all --config CFG
     research-pipeline screen --run-id <NEW_RUN_ID> --diversity --config CFG
     research-pipeline expand --run-id <NEW_RUN_ID> \
       --paper-ids "<prior-paper-ids>" --direction both --config CFG
     research-pipeline download --run-id <NEW_RUN_ID> --config CFG
     research-pipeline convert --run-id <NEW_RUN_ID> --config CFG
     research-pipeline extract --run-id <NEW_RUN_ID> --config CFG
     research-pipeline summarize --run-id <NEW_RUN_ID> --config CFG
     ```
   - Merge the new evidence with the prior corpus; the global
     SQLite paper index deduplicates downloads and conversions
     automatically.

4. **Regenerate the report** using `research-pipeline report` and the
   resume-on-top flow. The gap that motivated this round must either
   appear as **resolved** (with citations) or be **explicitly
   reclassified** (e.g., from `ACADEMIC, HIGH` to
   `ACADEMIC, LOW — no new literature found`, or to
   `ENGINEERING`).

5. **Append to the round history table** at the top of the report
   (under `## Round History`, right after `## Contents`):

   ```markdown
   | Round | Run ID | Topic / Gap Focus | New Papers | Gaps Addressed | Remaining Gaps |
   |-------|--------|-------------------|------------|----------------|----------------|
   | 1 | abc123 | original topic | 8 | initial shortlist | 4 academic, 2 engineering |
   | 2 | def456 | temporal memory indexing (academic gap) | 3 | 2 academic | 2 academic, 2 engineering |
   | 3 | ghi789 | memory conflict resolution (academic gap) | 2 | 2 academic | 2 engineering |
   | 4 | jkl012 | — (engineering polish only) | 0 | 2 engineering | 0 |
   ```

## Convergence rules

Stop iterating when **any** of these are true:

- The regenerated report has **no open gaps** (both academic and
  engineering sections are empty or contain only accepted limitations).
- **4 rounds have been completed.** Hard cap, no exceptions.
- A round searched and found **no new relevant papers** after
  screening (`screen` shortlist empty or entirely duplicated against
  the global index).
- The user has flagged remaining gaps as out-of-scope.

When stopping, the final round's report must clearly state:

- How many rounds were run and why iteration stopped.
- What gaps (if any) remain, with classification and why they were
  not closed (out-of-scope, no literature, deferred to
  implementation, etc.).
- Recommendations for manual follow-up on any remaining gaps.

## When there are no starting gaps (round 1 produced a clean report)

If the first report already has no open gaps, **do not** invent rounds
to burn the budget. Report to the user that iteration converged in a
single round and stop.

## System-Design Handover

For system-building requests, after the iterative loop converges:

1. Present a concise study result summary.
2. Ask the user whether to proceed to `req-analysis` skill for
   requirements clarification → story generation → architecture design.
3. If yes: invoke `req-analysis` skill with the final research report
   path.
4. If no: end normally.
