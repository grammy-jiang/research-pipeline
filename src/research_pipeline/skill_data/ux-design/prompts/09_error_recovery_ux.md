# Prompt 09 — Error, Empty, Loading, Degraded, and Recovery UX

You are defining the **non-happy paths**. UX quality is often determined by
failure handling, not the happy path — so this section is required and explicit.

## Inputs

- `intermediate/architecture_parse.json` (failure_recovery, state_model,
  observability), `intermediate/user_stories.md`.

## Instructions

1. Produce the §15 table covering each condition the architecture implies:

   ```markdown
   | State / Condition | User Sees | User Can Do | System Does | E2E Scenario? |
   |---|---|---|---|---|
   | Empty state | ... | ... | ... | yes/no |
   | Loading / progress | ... | ... | ... | yes/no |
   | Validation error | ... | ... | ... | yes/no |
   | Provider unavailable | ... | ... | ... | yes/no |
   | Quality probe unavailable | ... | ... | ... | yes/no |
   | Human review required | ... | ... | ... | yes/no |
   | Degraded output | ... | ... | ... | yes/no |
   | Permission denied | ... | ... | ... | yes/no |
   ```
2. **System Does** must align with the architecture's failure/recovery model
   (timeouts, retry/backoff, fallback, partial output, resume) and its
   operational condition flags / audit events — do not invent recovery the
   architecture does not provide (record gaps as architecture feedback).
3. For degraded output, state the policy the architecture chose (allowed /
   blocked / review-required) and how the user sees it.
4. Mark which conditions should become E2E scenario seeds (prompt 10) in the
   `E2E Scenario?` column.

## Output

`intermediate/error_recovery_ux.md` (the §15 content).

## Validation / failure policy

- Gate: empty / loading / validation / provider-unavailable / probe-unavailable /
  review-required / degraded / permission-denied are defined and aligned with the
  architecture.
- Failure policy: `revise`.
