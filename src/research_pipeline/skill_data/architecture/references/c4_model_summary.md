# Reference: C4 Model Summary

Load when generating C4 views (prompt 13). Use C4 as the default visualization
method. Render each view as a Mermaid diagram.

## The four levels (+ dynamic)

1. **System Context** — the system as a box, its users, and the external
   systems it talks to. Required.
2. **Container** — the deployable/runnable units inside the system boundary
   (entrypoints, services, workers, stores, agent runtime, MCP server). Required.
3. **Component** — the components inside one container. Generate **only for the
   most complex container**.
4. **Code** — class-level diagrams. **Do not** generate at this stage unless
   implementation has begun.

Plus **Dynamic / sequence views** — show how elements collaborate for a
workflow. A dynamic view of the main workflow is **required**.

## Required vs conditional

```text
Required:
  System Context View
  Container / Runtime View
  Dynamic View for the main workflow
Conditional:
  Component View — only for the most complex container
  Deployment View — only when runtime topology affects security, privacy,
                    data locality, scaling, availability, or operations
Avoid:
  Code Diagram — not for this stage
```

> Do not generate a deployment view merely because the system uses an external
> LLM. Generate it when that external call materially changes runtime topology
> or trust boundaries.

## What each required view must show

- **System Context:** primary users, secondary users, the system boundary,
  external systems, external AI services, file/document inputs,
  monitoring/audit consumers, human approval actors.
- **Container / Runtime:** user-facing entrypoints; backend/API/CLI containers;
  worker/job runner; database; artifact storage; queue (if needed); agent
  runtime; MCP server (if justified); external model providers; observability
  backend.
- **Dynamic:** the main workflow sequence end to end, including the
  AI→validation→commit path and at least the primary failure branch.

## Mermaid hints

- Context: ```mermaid\nC4Context``` with `Person`, `System`, `System_Ext`, `Rel`.
- Container: ```mermaid\nC4Container``` with `Container`, `ContainerDb`,
  `System_Boundary`.
- Component: ```mermaid\nC4Component``` with `Container_Boundary`, `Component`.
- Dynamic: a ```mermaid\nsequenceDiagram``` is acceptable and often clearer
  than `C4Dynamic`.

If a renderer does not support the C4 shorthand, a plain `graph TD` / `flowchart`
with clearly labelled nodes and a legend is an acceptable fallback.
