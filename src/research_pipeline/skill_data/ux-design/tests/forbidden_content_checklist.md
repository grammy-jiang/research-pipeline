# Checklist: Forbidden Content

The `ux-design` skill must stay on the UX side of the boundary. The output must
**not** contain any of the following (each is a quality-gate FAIL):

- [ ] **Executable tests** — step definitions, assertions written in a
      programming language, fixtures/mocks, or test-runner / CI configuration.
      (E2E scenario *seeds* in Gherkin are allowed; executable tests are not.)
- [ ] **Architecture decisions** — (re)defining the state model, interface/data
      contracts, trust boundaries, or workflows. UX consumes these; gaps become
      §21 architecture feedback.
- [ ] **Tech-stack selection** — choosing languages, frameworks, databases,
      queues, providers, or deployment targets. That is `architecture --mode
      stack`.
- [ ] **Implementation tasks** — task tickets, code patches, migrations, or
      file-by-file build steps. That is the implementation-plan skill.
- [ ] **Pixel-level visual design** — wireframes, layouts, colours, spacing, CSS,
      or component styling.
- [ ] **Final screen copy** — exact user-facing wording / microcopy.
- [ ] **Final exact CLI flags / API schemas** — unless the architecture already
      specifies them. UX describes behaviour conceptually.

If the UX needs something on this list to exist, it does **not** create it — it
records the need as architecture feedback (§21) or defers it to the appropriate
downstream skill.
