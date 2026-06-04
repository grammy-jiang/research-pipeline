# Contents Template

Every architecture document begins with a `## Contents` section linking to all
25 numbered sections plus the `Update History` block. Keep the order below.
GitHub-style anchors lowercase the heading, drop punctuation, and replace
spaces with hyphens (a `/` becomes `--`).

```markdown
## Contents

- [1. Executive Architecture Summary](#1-executive-architecture-summary)
- [Update History](#update-history)
- [2. Source Blueprint Interpretation](#2-source-blueprint-interpretation)
- [3. Clarification Summary](#3-clarification-summary)
- [4. Architecture Goals and Constraints](#4-architecture-goals-and-constraints)
- [5. Solution Strategy](#5-solution-strategy)
- [6. Traditional Software vs AI-Agent Boundary](#6-traditional-software-vs-ai-agent-boundary)
- [7. Recommended Tech Stack](#7-recommended-tech-stack)
- [8. System Context View](#8-system-context-view)
- [9. Container / Runtime View](#9-container--runtime-view)
- [10. Component View](#10-component-view)
- [11. AI / Skill / MCP Architecture](#11-ai--skill--mcp-architecture)
- [12. Interface Contracts](#12-interface-contracts)
- [13. Data Contracts and Schemas](#13-data-contracts-and-schemas)
- [14. State, Storage, and Data Lifecycle](#14-state-storage-and-data-lifecycle)
- [15. Workflow / Sequence Views](#15-workflow--sequence-views)
- [16. Observability, Logging, Telemetry, and Audit](#16-observability-logging-telemetry-and-audit)
- [17. Security and Trust Boundaries](#17-security-and-trust-boundaries)
- [18. Failure Handling and Recovery](#18-failure-handling-and-recovery)
- [19. Testing and Evaluation Architecture](#19-testing-and-evaluation-architecture)
- [20. Deployment Architecture](#20-deployment-architecture)
- [21. Architecture Decision Records](#21-architecture-decision-records)
- [22. Technical Risks and Trade-offs](#22-technical-risks-and-trade-offs)
- [23. Open Questions](#23-open-questions)
- [24. Architecture Quality-Gate Self-Check](#24-architecture-quality-gate-self-check)
- [25. Handoff Notes for Implementation Planning](#25-handoff-notes-for-implementation-planning)
```

Rule: the Contents must list every numbered section that appears in the body.
If you add an appendix, add it here too.
