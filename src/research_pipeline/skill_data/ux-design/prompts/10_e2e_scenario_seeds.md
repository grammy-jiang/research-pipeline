# Prompt 10 — E2E Scenario Seeds

You are generating **E2E scenario seeds**, not executable tests. A seed is a
human-readable, Gherkin-style scenario derived from a user story and its
acceptance criteria. Later stages convert seeds into executable tests.

## Inputs

- `intermediate/user_stories.md`, `intermediate/error_recovery_ux.md`.
- `references/e2e-scenario-seed-guide.md`, `templates/e2e-scenario-template.md`.

## Instructions

1. For each story marked for E2E coverage (and each failure condition flagged in
   §15), write a seed in Gherkin style:

   ```gherkin
   Feature: <capability>
   Scenario: <happy / alternative / failure scenario>
     Given <precondition>
     And <precondition>
     When <user action>
     Then <observable outcome>
     And <observable outcome>
   ```
2. Cover, at minimum: the primary happy path; the main failure/recovery paths;
   each user-visible state that must be testable; and the human-review path if
   review exists. Tie each seed to a surface (CLI / Web / API / MCP / AI skill)
   per the architecture.
3. **Seeds are not tests.** Do not write step definitions, assertions in code,
   fixtures, or test-runner config. Keep them declarative and surface-agnostic
   where possible (note the intended surfaces).
4. State the seed → test pipeline once: UX user story → acceptance criteria →
   E2E scenario seed → implementation-plan test task → executable test.
5. Add **testability metadata** as a header block before each Gherkin scenario:

   ```markdown
   **Phase:** MVP-0 / MVP-1 / Phase 2 / Phase 3 / Future
   **Surface:** CLI / API / Web / MCP / AI Skill / filesystem / audit store
   **Release Gate:** blocks MVP-0 / blocks MVP-1 / regression / optional
   **Deterministic:** yes / no
   **Requires Real LLM:** yes / no
   **CI Suitable:** yes / no
   **Required Fixtures:** <fixture list, or none>
   **Must Mock:** <what to mock, e.g. LLM provider, clock; or none>
   **Required Architecture Contracts:** <§12 contract references, or none>
   **Required Implementation Components:** <module/component names, or none>
   ```

   Rules: every MVP-0 seed must be CI Suitable = yes unless a real LLM call is
   structurally required. Justify every "Requires Real LLM = yes" in the metadata.
6. Emit a **Testability Summary Table** at the end of §20:

   | E2E Seed | Phase | Surface | Deterministic? | Real LLM? | CI Suitable? | Release Gate |
   |---|---|---|---|---|---|---|
   | <E2E-XX Name> | <phase> | <surface> | yes/no | yes/no | yes/no | <gate> |

## Output

`intermediate/e2e_scenario_seeds.md` (the §20 content).

## Validation / failure policy

- Gate: testable Gherkin-style seeds exist for the primary path, key failures,
  and testable user-visible states; no executable test code is emitted; every
  seed has phase and testability metadata; at least one MVP-0 CI-suitable seed
  exists for the core happy path.
- Failure policy: `revise`.
