# Recommended Next Stages and Downstream Handoffs Guide

Load this when authoring **§24 Recommended Next Stages and Downstream Handoffs**
in `design` mode, and when keeping the tech stack provisional in §7. This
section consumes the blueprint's **§19 Recommended Next Stages** (the first
adaptive stage-gate router) and reflects its routing.

## What blueprint §19 provides

The blueprint routes downstream stages with controlled decisions
(**RUN / SKIP / DEFER / ASK_USER**) for:

```text
tech-stack-selection   -> architecture `stack` mode
ux-design              -> the ux-design skill
security-review        -> security review
test-design            -> test / E2E design
architecture-update    -> architecture `update` mode
architecture-reconciliation -> architecture `reconcile` mode
```

Design mode **reflects** that routing — it does not silently expand or collapse
the pipeline. Recommendations are overrideable defaults.

## How routing changes design-mode output

| Blueprint §19 routing | Design-mode obligation |
|---|---|
| `tech-stack-selection = RUN` | Do **not** over-finalize the tech stack. Keep §7 provisional, fill §7.x Provisional Tech Assumptions + Tech-Stack Selection Handoff. |
| `tech-stack-selection = SKIP` (stack fixed) | The §7 stack may be treated as fixed; say so and record who fixed it. |
| `ux-design = RUN` or `DEFER` | Produce §23 Experience Architecture + a UX handoff (23.9), but **not** detailed user journeys / screen / command design. |
| `ux-design = SKIP` | Minimal §23 Experience Architecture; note detailed UX is out of scope. |
| `security-review = RUN` | Make data flow, trust boundary, data egress, and audit explicit (§14/§16/§17) and fill the Security-Review Handoff. |
| `architecture-update = DEFER` | Define the **update triggers** (what accepted change re-opens the architecture). |
| `architecture-reconciliation = DEFER` | Define the **reconciliation triggers** (what downstream conflict re-opens the architecture). |

## Provisional-tech discipline in §7

Because `stack` mode owns final technology selection, `design` mode keeps §7
provisional whenever tech-stack-selection is RUN/DEFER.

### Allowed in design mode

```text
provisional technology assumptions
technology constraints and decision drivers
runtime capability requirements
stack-sensitive architecture notes
handoff questions for stack mode
```

### Not allowed in design mode (when stack mode is recommended)

```text
final framework choice without justification
final database choice when multiple viable options exist
final cloud/deployment provider choice
final AI orchestration framework choice
final MCP SDK choice
```

### §7.x Provisional Tech Assumptions

```markdown
| Area | Provisional Assumption | Reason | Must Be Confirmed In Stack Mode? |
|---|---|---|---|
| Runtime | <e.g. Python likely> | <reason> | Yes |
| Storage | <e.g. embedded or server store possible> | <reason> | Yes |
| LLM abstraction | <e.g. gateway or direct SDK possible> | <reason> | Yes |
```

### §7.x Tech-Stack Selection Handoff

```markdown
| Decision Needed | Architecture Constraint | Candidate Options | Risk if Wrong |
|---|---|---|---|
| Database | <audit/job-state/registry constraint> | <opt A / opt B> | <impact> |
| LLM provider abstraction | <providers / routing / egress logging> | <opt A / opt B / opt C> | <impact> |
```

## §24 Recommended Next Stages and Downstream Handoffs structure

```markdown
## 24. Recommended Next Stages and Downstream Handoffs

### 24.1 Recommended Next Stages Consumed

| Stage | Blueprint §19 Decision | Architecture Response | Trigger / Depends On |
|---|---|---|---|
| tech-stack-selection | RUN/SKIP/DEFER/ASK_USER | <e.g. §7 kept provisional; see Tech-Stack Selection Handoff> | <when> |
| ux-design | … | <e.g. §23 Experience Architecture + UX handoff produced> | … |
| security-review | … | <e.g. §17 egress/trust made explicit> | … |
| test-design | … | <e.g. testing architecture in §19; E2E seeds noted> | … |
| architecture-update | DEFER | <update triggers defined below> | … |
| architecture-reconciliation | DEFER | <reconciliation triggers defined below> | … |

### 24.2 Tech-Stack Selection Handoff

(Pointer to §7.x, plus any additional stack questions raised during design.)

### 24.3 UX-Design Handoff

| UX Area | Architecture Constraint | Open UX Question |
|---|---|---|

### 24.4 Security-Review Handoff

| Concern | What the Architecture Already Decided | What Security Review Must Validate |
|---|---|---|

### 24.5 Test-Design / E2E Handoff

| Critical Workflow | Architecture States/Contracts to Cover | Suggested E2E Scenario Seed |
|---|---|---|

### 24.6 Update and Reconciliation Triggers

| Trigger | Re-opens | Mode |
|---|---|---|
| <e.g. stack selection changes the storage backend> | §14/§20 | update |
| <e.g. ux-design exposes a missing state> | §14/§23 | reconcile |
```

Keep §24 compact (tables over prose). It mirrors the blueprint's §19 routing —
if a stage is SKIP, say so and keep its handoff minimal; do not invent stages
the blueprint did not route.
