# Expected Sections & Acceptance Checklist

Use this checklist as the pass/fail contract for a generated blueprint. It
combines the seven quality gates with the acceptance criteria.

## Structure

- [ ] `## Contents` section with working internal links.
- [ ] All 18 required sections present, in order:
  - [ ] 1. Executive Product Thesis
  - [ ] 2. Source Research Interpretation
  - [ ] 3. Target Users and System Actors
  - [ ] 4. Product Goals and Non-Goals
  - [ ] 5. Research-to-Product Translation Map
  - [ ] 6. Adopt / Adapt / Merge / Defer / Reject Decisions
  - [ ] 7. Core Product Capabilities
  - [ ] 8. Workflow Model
  - [ ] 9. Logical Architecture
  - [ ] 10. Conceptual Information Model
  - [ ] 11. Decision Policies
  - [ ] 12. Risk, Governance, and Safety Model
  - [ ] 13. Evaluation Strategy
  - [ ] 14. MVP Scope
  - [ ] 15. Roadmap and Future Extensions
  - [ ] 16. Open Questions and Validation Plan
  - [ ] 17. Handoff Notes for Technical Design
  - [ ] 18. Traceability Appendix

## Content quality

- [ ] Product thesis is clear and specific (not a paper summary).
- [ ] Research mechanisms are translated into product primitives.
- [ ] Major ideas are classified ADOPT / ADAPT / MERGE / DEFER / REJECT.
- [ ] Each research-derived decision includes a research citation.
- [ ] Each non-research design decision includes an explicit rationale and
      is limited to connecting, operationalizing, or governing
      research-backed capabilities.
- [ ] Core capabilities are explicit (not vague).
- [ ] Workflows include trigger, inputs, decision gates, steps, outputs,
      failure modes, and success criteria.
- [ ] Main end-to-end workflow has a Mermaid diagram.
- [ ] Logical architecture is conceptual (not technical) with a Mermaid
      diagram.
- [ ] Complex / safety-critical / high-risk workflows have extra Mermaid
      diagrams.
- [ ] Conceptual information objects defined with lifecycle states.
- [ ] Decision policies are explicit with defaults and escalation rules.
- [ ] Risks and governance controls are explicit and realistic.
- [ ] `ACADEMIC` gaps → validation requirements, not MVP features.
- [ ] `ENGINEERING` gaps → product requirements.
- [ ] Evaluation strategy has ≥1 scenario per capability.
- [ ] MVP scope is small with an explicit success definition.
- [ ] Future roadmap exists with deferred items justified.
- [ ] Open questions are not hidden.
- [ ] Handoff notes explicitly list what technical design must decide.

## Hard fails

- [ ] No tech stack selected (language, framework, database, cloud,
      vendor, UI library, deployment model).
- [ ] No code generated.
- [ ] No implementation tasks/tickets generated.
- [ ] Quality gates run and pass — or failures are surfaced to the user
      after at most 3 revision attempts.
