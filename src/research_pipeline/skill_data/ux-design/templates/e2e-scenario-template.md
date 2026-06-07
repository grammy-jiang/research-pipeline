# E2E Scenario Seed Template

Use this skeleton for §20. A **seed** is human-readable and declarative — it is
**not** an executable test (no step definitions, code assertions, fixtures, or
runner config).

**Phase:** MVP-0 / MVP-1 / Phase 2 / Phase 3 / Future
**Surface:** CLI / API / Web / MCP / AI Skill / filesystem / audit store
**Release Gate:** blocks MVP-0 / blocks MVP-1 / regression / optional
**Deterministic:** yes / no
**Requires Real LLM:** yes / no
**CI Suitable:** yes / no
**Required Fixtures:** <fixture list, or none>
**Must Mock:** <what to mock, e.g. LLM provider, clock; or none>
**Required Architecture Contracts:** <§12 contracts this seed exercises, or none>
**Required Implementation Components:** <module/component names needed, or none>

```gherkin
Feature: <capability under test>

Scenario: <happy / alternative / failure scenario name>
  Given <precondition tied to architecture state>
  And <precondition>
  When <user / agent action>
  Then <observable outcome the user can see>
  And <observable outcome>
  And an audit record is created
```

## Worked shape (illustration, not a recommendation)

**Phase:** MVP-0
**Surface:** CLI + filesystem
**Release Gate:** blocks MVP-0
**Deterministic:** yes
**Requires Real LLM:** no
**CI Suitable:** yes
**Required Fixtures:** short academic paper fixture, fake LLM provider
**Must Mock:** LLM provider, clock
**Required Architecture Contracts:** submit-job CLI contract, audit-write contract
**Required Implementation Components:** job-intake, worker, audit-writer

```gherkin
Feature: Translation job submission

Scenario: Successful CLI translation with no review required
  Given a valid English academic paper
  And a valid academic domain profile
  And external model use is allowed
  When the operator submits a translation job
  Then the system creates a translation job
  And the operator sees progress
  And the system returns a Chinese translation
  And the quality summary says review is not required
  And an audit record is created
```

## Rules

- One `Feature` per capability; one `Scenario` per happy / alternative / failure
  path that must be testable.
- **Tie each seed to a surface** (CLI / Web / API / MCP / AI skill) per the
  architecture; a seed may note "applies to CLI and API".
- Cover at minimum: the primary happy path; the main failure/recovery paths;
  each user-visible state that must be testable; and the human-review path if
  review exists.
- **Seeds are not tests.** Later stages convert a seed into an
  implementation-plan test task and then an executable test (CLI / API / Web /
  MCP / AI-skill), depending on the selected surfaces.
- Keep `Then` clauses to **observable** outcomes (what the user/agent sees or
  what is recorded), not internal implementation details.
