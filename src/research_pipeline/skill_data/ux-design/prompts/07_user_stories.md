# Prompt 07 — User Stories

You are writing the **story-driven** core of the UX design. Every major user
story is structured and testable.

## Inputs

- `intermediate/skill_operator_ux.md`, `intermediate/target_software_ux.md`,
  `intermediate/architecture_parse.json`, `intermediate/clarifications.md`.
- `templates/user-story-template.md`.

## Instructions

1. Identify the major user stories from the workflows, actors, and JTBD —
   covering the primary happy path, the key alternative paths, and the important
   failure/recovery paths. Include reviewer and agent/MCP stories where those
   actors exist.
2. For **each** story, use the template format:

   ```markdown
   ## User Story: <Story Name>
   **As a:** <role>  **I want:** <goal>  **So that:** <value>
   ### Preconditions
   ### Main Flow
   ### Alternative Flows
   ### Failure / Recovery Flows
   ### User-Visible States
   ### Acceptance Criteria
   ### E2E Scenario Seeds
   ```
3. **User-Visible States** must resolve to the architecture's state model
   (lifecycle states vs condition flags vs audit events). Do not introduce a
   user-visible state with no architecture counterpart — record it as
   architecture feedback instead.
4. **Acceptance Criteria** must be specific and testable (observable outcome +
   condition), not vague ("works well").
5. **E2E Scenario Seeds** are short pointers here (one line each); the full
   Gherkin-style seeds are expanded in prompt 10.
6. Keep stories at the experience level — no exact CLI flags, API schemas, or
   visual layout.
7. Add **phase and release-gate metadata** to every story header, using these
   controlled values:

   ```markdown
   **Phase:** MVP-0 / MVP-1 / Phase 2 / Phase 3 / Future
   **Primary Surface:** CLI / API / Web / MCP / AI Skill / filesystem
   **Release Gate:** blocks MVP-0 / blocks MVP-1 / regression / optional
   **Depends On:** <architecture contracts / stack decisions / previous stories>
   ```

   | Phase | Meaning |
   |---|---|
   | MVP-0 | Minimum slice required to prove the core end-to-end workflow |
   | MVP-1 | Next capability layer (e.g. human review, AI wrapper, extended CLI) |
   | Phase 2 | Larger capability expansion |
   | Phase 3 | Integration expansion (MCP / multi-agent / external surfaces) |
   | Future | Not planned for near-term implementation |

   Every story must have a phase. Do not mark a story MVP-0 unless it is truly
   required for the minimum working product.

## Output

`intermediate/user_stories.md` (the §10 content; feeds §11 journeys, §19
acceptance criteria, §20 E2E seeds).

## Validation / failure policy

- Gate: each major story has preconditions, main + alternative + failure/recovery
  flows, user-visible states (resolving to the state model), testable acceptance
  criteria, and a phase / release-gate header.
- Failure policy: `revise`.
