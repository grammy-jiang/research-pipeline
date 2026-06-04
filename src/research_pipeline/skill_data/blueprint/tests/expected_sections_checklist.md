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
- [ ] MVP scope separates a minimal Core Value Path from Safety and
      Evaluation baselines, with an explicit success definition.
- [ ] Future roadmap exists with deferred items justified.
- [ ] Open questions are not hidden.
- [ ] Handoff notes explicitly list what technical design must decide.

## Integrity & discipline

- [ ] Metadata is copied, not invented: skill version from `manifest.json`
      or `unknown`; **pipeline runs integrated** and **gap-closure rounds**
      are separate fields (no run count relabelled as a round count).
- [ ] The thesis leads with the primary research-backed architecture; a
      conditional/secondary/escalation-only mechanism is not promoted to
      product identity.
- [ ] Every primary actor / MVP domain is named (or implied) by the
      thesis; high-stakes domains seen only as evidence are
      Secondary/Future, not primary.
- [ ] Each major claim is research-backed, engineering-extrapolation,
      product-design-decision, or relocated (speculative → Open Questions;
      unsupported → removed or marked design hypothesis).
- [ ] No citation cell is blank: gap-derived items cite
      `[Source Report: Research Gaps — <name>]`; blank only for an explicit
      internal design hypothesis.
- [ ] §14 splits the core path into MVP-0 (smallest demonstrable) and MVP-1
      (first usable); a large Phase-1 system is not labelled MVP-0.
- [ ] Release gates from MEDIUM/LOW-confidence mechanisms are justified
      (HIGH risk + no cheaper control + why-now) or downgraded.
- [ ] Runtime/architecture-leaning wording is rephrased or surfaced as a
      WARNING (not silently accepted); only forbidden tech/vendor choices
      hard-fail.
- [ ] `standard` output respects the length budget, or switches to
      `detailed`.
- [ ] An `## Appendix A: Blueprint Quality-Gate Self-Check` is present;
      every WARNING has a required action and a blocks-technical-design
      verdict; any FAIL is resolved before delivery.

## Hard fails

- [ ] No tech stack selected (language, framework, database, cloud,
      vendor, UI library, deployment model).
- [ ] No code generated.
- [ ] No implementation tasks/tickets generated.
- [ ] Quality gates run and pass — or failures are surfaced to the user
      after at most 3 revision attempts.
