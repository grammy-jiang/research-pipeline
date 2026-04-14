# Iterative Synthesis for System Building

When the user's goal is to **build a system** (not just a literature review),
the synthesis must be evaluated for implementation-readiness.

Detect from keywords: "build", "implement", "design", "create", "develop",
"architecture", "system", or explicit statements about building.

## After paper-synthesizer Completes

1. **Read the synthesis report** and its readiness assessment
2. **Check the verdict**:
   - `IMPLEMENTATION_READY` — sufficient for system design
   - `HAS_GAPS` — gaps need filling

3. **If `HAS_GAPS`**, process each gap by type:

### Engineering Gaps

Gaps in implementation details, best practices, configuration, tooling,
deployment patterns, or integration approaches.

**Action**: Fill using your own knowledge, web searches, and best practices.
Append to the synthesis report as: `## Engineering Gap Resolution`

### Academic Gaps

Gaps requiring additional research papers — missing algorithms, unexplored
approaches, theoretical foundations, or evaluation methodologies.

**Action**: For each academic gap:
1. Extract suggested search terms from synthesizer
2. Run a new pipeline iteration:
   ```bash
   research-pipeline plan "<gap-specific topic>" --config CFG
   research-pipeline search --run-id <NEW_RUN_ID> --config CFG
   research-pipeline screen --run-id <NEW_RUN_ID> --config CFG
   ```
3. Download, convert, extract, summarize the new papers
4. Run paper-analyzer and paper-synthesizer on the new run
5. **Merge** findings into original synthesis

## Iteration Rules

- **Maximum 3 iterations** to prevent infinite loops
- Each iteration MUST narrow the search (more specific terms)
- Track all run IDs and relationships
- Final report must include ALL iterations
- Report iteration history:
  ```
  | # | Run ID | Topic | Papers | Gaps Filled |
  |---|--------|-------|--------|-------------|
  | 1 | abc123 | original topic | 8 | initial synthesis |
  | 2 | def456 | temporal memory indexing | 3 | temporal reasoning |
  | 3 | ghi789 | memory conflict resolution | 2 | conflict resolution |
  ```

## Convergence

Stop iterating when:
- Synthesizer returns `IMPLEMENTATION_READY`
- Maximum 3 iterations reached
- No new academic gaps identified
- New searches return no relevant papers

Report to user: remaining gaps (if any), why iteration stopped,
recommendations for manual investigation.

## System-Design Handover

After the final report, if the goal was system-building:

1. Present a concise study result summary
2. Ask the user whether to proceed to `req-analysis` skill for
   requirements clarification → story generation → architecture design
3. If yes: invoke `req-analysis` skill with research report path
4. If no: end normally
