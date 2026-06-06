# Checklist: Expected Sections

The UX design document must contain `## Contents`, `## Update History`, all 22
numbered sections in order, and `## Appendix A. UX Quality-Gate Self-Check`.

- [ ] `## Contents` (links every numbered section + Appendix A)
- [ ] 1. Generation Metadata
- [ ] `## Update History` (table, ≥ 1 row)
- [ ] 2. Source Architecture Interpretation
- [ ] 3. Source Blueprint Interpretation
- [ ] 4. UX Goals and Non-Goals
- [ ] 5. Skill Operator UX
- [ ] 6. Target Software UX
- [ ] 7. Users, Roles, and Jobs-to-Be-Done
- [ ] 8. UX Decision Summary
- [ ] 9. UX Assumptions
- [ ] 10. User Stories (each: preconditions, main/alternative/failure-recovery
      flows, user-visible states resolving to the architecture state model,
      acceptance criteria, E2E seeds)
- [ ] 11. Core User Journeys
- [ ] 12. Surface-Specific UX (only architecture-supported surfaces; others
      marked "not used by this architecture")
- [ ] 13. Human-in-the-Loop UX (required if the architecture has a review flow)
- [ ] 14. Trust, Control, and Transparency UX
- [ ] 15. Error, Empty, Loading, Degraded, and Recovery States
- [ ] 16. Notifications and Feedback
- [ ] 17. Accessibility and Internationalization
- [ ] 18. UX Observability
- [ ] 19. Acceptance Criteria
- [ ] 20. E2E Scenario Seeds (Gherkin-style; not executable tests)
- [ ] 21. Architecture Feedback / Required Architecture Updates (mandatory; with
      a reconcile decision)
- [ ] 22. Handoff Notes for Implementation Planning
- [ ] `## Appendix A. UX Quality-Gate Self-Check` (PASS/WARNING/FAIL table)

Hard rules:

- [ ] §5 Skill Operator UX and §6 Target Software UX are present and **separate**.
- [ ] §21 is present even when empty ("No architecture reconciliation required.").
- [ ] User-visible states resolve to the architecture state model (no invented
      states — invented needs become §21 feedback).
