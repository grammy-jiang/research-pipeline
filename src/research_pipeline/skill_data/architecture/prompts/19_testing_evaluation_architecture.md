# Prompt 19 — Testing and Evaluation Architecture

You are defining how the system is tested and how AI behavior is evaluated.

## Inputs

- `intermediate/interface_contracts.md`, `intermediate/failure_handling.md`,
  `intermediate/blueprint_parse.json` (evaluation strategy).

## Instructions

Define the test matrix: unit; integration; contract (against §12); end-to-end
(main workflow); golden (deterministic fixtures); AI evaluation fixtures;
security tests; observability tests; failure-mode tests (one per §18 concern);
regression tests.

For AI systems, additionally define: prompt regression tests; model output
evaluation; human-review sampling (if needed); tool-call safety tests;
adversarial / prompt-injection tests.

Tie each test type to what it protects (a contract, a workflow, a failure mode,
a security control).

## Output

`intermediate/testing_evaluation.md` → populates §19.

## Validation / failure policy

- Gate: every critical workflow and every interface contract has a
  corresponding test type; AI behavior has evaluation + injection tests.
- Failure policy: `revise`.
