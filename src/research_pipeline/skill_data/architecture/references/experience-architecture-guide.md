# Experience Architecture Guide

Load this when authoring **§23 Experience Architecture** in `design` mode. This
section consumes the blueprint's **§9 Product Experience Direction** and
translates UX *intent* into UX-enabling *technical structure*.

## What this is — and is not

```text
blueprint §9 Product Experience Direction:  UX intent (what the experience should be)
architecture §23 Experience Architecture:   technical support for that intent
ux-design (future skill):                    detailed journeys, screens, command/conversation UX, copy
```

Experience Architecture is **architecture-level UX support**, not detailed UX
design. Do **not** produce screen layouts, wireframes, exact CLI syntax, exact
API/MCP schemas, copywriting, or implementation tasks — those belong to the
future `ux-design` skill. Keep it compact (tables over prose).

## How to consume blueprint §9

Read the blueprint's §9 sub-sections (Primary Experience Thesis; Primary User /
Operator; Job-to-Be-Done; Primary + Secondary Interaction Modes; Trust / Control
/ Transparency requirements; Human-in-the-Loop experience; Failure and Recovery
expectations; UX Assumptions for Architecture; Product Experience Handoff to
Architecture) and turn each into a technical structure. Preserve the experience
thesis — never change the blueprint's UX intent.

## §23 Experience Architecture structure

```markdown
## 23. Experience Architecture

### 23.1 UX Direction Inherited from Blueprint

| Blueprint UX Decision (§9) | Architecture Interpretation |
|---|---|

### 23.2 Interaction Surface Matrix

| Actor | MVP Surface | Later Surface | Notes |
|---|---|---|---|

### 23.3 User-Visible State Model

| User-Visible State | Internal State (§14) | User Meaning | Allowed User Action |
|---|---|---|---|

### 23.4 Feedback and Progress Model

| Workflow Step | User Feedback | Technical Support |
|---|---|---|

### 23.5 Error and Recovery Model

| Error / Condition | User Sees | User Can Do | System Does (§18) |
|---|---|---|---|

### 23.6 Human Review Technical Flow

| Trigger | Review Artifact / Surface | User Action | Audit Event (§16) |
|---|---|---|---|

### 23.7 Trust and Transparency Support

| User Trust Need | Technical Support |
|---|---|

### 23.8 Interaction Observability

| Interaction Event | Signal Captured | Where (§16) |
|---|---|---|

### 23.9 UX Handoff to UX-Design

| UX Area | Architecture Constraint | Open UX Question |
|---|---|---|
```

## Cross-section consistency

- **23.3 User-Visible State Model** must map onto the canonical §14 state model
  (lifecycle states vs operational condition flags vs audit events) — do not
  invent user-visible states that have no internal counterpart.
- **23.5 Error and Recovery Model** must align with §18 Failure Handling.
- **23.6 Human Review Technical Flow** must produce §16 audit events.
- **23.8 Interaction Observability** routes to the §16 observability plan.
- **23.9 UX Handoff** is the bridge to the future `ux-design` skill; list the
  architecture constraints UX must honour and the open UX questions, but do not
  answer them here.

## Routing dependency

The depth of §23 follows the blueprint's §19 routing for `ux-design` (see
`references/next-stages-and-handoffs-guide.md`):

```text
ux-design = RUN or DEFER:
  produce Experience Architecture + a UX handoff (23.9),
  but NOT detailed user journeys or screen/command design.

ux-design = SKIP:
  produce a minimal Experience Architecture (surfaces, user-visible states,
  trust/transparency) and note that detailed UX is out of scope.
```
