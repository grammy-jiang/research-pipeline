# Guideline: Designing Complex Multi-Step Skills as Governed Task-Manager Workflows

## 1. Core Principle

A complex AI-agent skill should not be implemented as a long natural-language checklist.

It should be designed as a governed workflow:

```text
Thin Skill
  -> Workflow Orchestrator / Task Manager
      -> Deterministic Scripts
      -> MCP Tools
      -> AI Worker Wrappers
      -> Validators
      -> Reviewer Gates
      -> Workflow State
      -> Artifact Store
      -> Final Synthesis
```

The skill should start the workflow.
The orchestrator should own the workflow truth.
Workers and subagents should execute small bounded tasks.
Validators should decide whether a task is actually complete.

The governing principle is:

> A complex skill should not ask an AI agent to remember and obey a long process. It should launch a task manager that decomposes work into small validated tasks, executes deterministic checks through tools, delegates bounded reasoning to workers, stores evidence as artifacts, and only synthesizes accepted results.

---

## 2. Recommended Mental Model

Use this model:

```text
Skill = entry point
Orchestrator = task manager and state authority
Task = small executable unit
Executor = script / MCP tool / AI worker / human gate
Artifact = durable evidence
Validator = acceptance authority
Reviewer = quality gate
State file = workflow memory
Final report = synthesis of accepted artifacts only
```

Do not rely on the agent's memory as the workflow state.

The workflow should be evidence-driven:

```text
No artifact -> no claim
No schema validation -> no acceptance
No accepted prior step -> no downstream step
No verified result -> no final synthesis
```

---

## 3. Skill Responsibility

The skill should be thin.

### The skill should do this

```text
- Recognize when the workflow applies.
- Explain the workflow purpose.
- Start the orchestrator.
- Tell the agent not to bypass the orchestrator.
- Tell the agent to report orchestrator results only.
```

### The skill should not do this

```text
- Contain all task logic.
- Depend on the agent remembering seven or eight ordered steps.
- Let the agent self-certify completion.
- Let the agent skip mandatory checks.
- Let the agent produce final conclusions without artifacts.
- Let the agent use conversational memory as workflow state.
```

### Example thin `SKILL.md`

```markdown
# Code Health Review Skill

When invoked, run:

python tools/agent_workflow/runner.py --manifest .agent/manifest.json

Rules:

- Do not claim completion unless the workflow state says complete.
- Do not summarize findings unless `.agent/artifacts/final_report.md` exists.
- Do not invent results for tests, linting, type checks, or security scans.
- If the runner reports blocked or failed steps, report the exact step ID and reason.
- Do not bypass the orchestrator.
```

---

## 4. Reference Workflow Architecture

Recommended repository structure:

```text
repo/
  .agent/
    manifest.json
    workflow_state.json
    artifacts/
    logs/
    schemas/
    steps/
    subagents/
  .claude/
    skills/
      your-skill/
        SKILL.md
    agents/
    settings.json
  .github/
    copilot-instructions.md
    workflows/
      skill-validation.yml
  AGENTS.md
  tools/
    agent_workflow/
      runner.py
      orchestrator.py
      scheduler.py
      executors/
      validators/
      reviewers/
      planners/
```

### Responsibility split

| Component | Responsibility |
|---|---|
| `SKILL.md` | Entry point and high-level rules |
| `manifest.json` | Ordered workflow definition |
| `workflow_state.json` | Durable workflow memory |
| `runner.py` | Starts and controls execution |
| `orchestrator.py` | Owns scheduling, validation, and state transitions |
| `scheduler.py` | Selects ready tasks and handles dependency ordering |
| `executors/` | Runs scripts, MCP tools, AI workers, or human gates |
| `validators/` | Decides whether outputs are acceptable |
| `reviewers/` | Performs second-pass critique and evidence checks |
| `planners/` | Proposes follow-up tasks from accepted results |
| `artifacts/` | Stores durable evidence and outputs |
| `logs/` | Stores execution logs and audit records |
| CI | Re-runs mandatory deterministic verification |

---

## 5. Fixed Skeleton plus Dynamic Task Queue

A reliable workflow should use a fixed control skeleton with a dynamic internal task queue.

### Fixed mandatory gates

These should not be dynamically removed:

```text
repo snapshot
git status
dependency check
format check
lint check
type check
unit tests
security scan
artifact validation
reviewer gate
final synthesis boundary
```

### Dynamic analysis tasks

These may be created during the workflow:

```text
inspect changed module
analyze affected API
check missing tests
review security risk
review performance risk
inspect migration impact
generate documentation update plan
inspect an unexpected dependency
review a failed test cluster
```

### Design rule

> The orchestrator may dynamically add tasks, but it must not dynamically remove mandatory gates.

Bad:

```text
The worker says tests are unnecessary, so the orchestrator skips tests.
```

Good:

```text
The worker proposes an additional API compatibility review.
The orchestrator validates the proposal and schedules it as a new task.
Mandatory tests still run.
```

---

## 6. Task Definition Pattern

Every task should be explicit.

```yaml
id: architecture-impact-analysis
type: llm_reasoning
phase: analysis
depends_on:
  - repo-snapshot
  - changed-file-inventory
inputs:
  - .agent/artifacts/repo_snapshot.json
  - .agent/artifacts/changed_files.json
executor:
  kind: llm_worker
  name: architecture-impact-analyst
output:
  path: .agent/artifacts/impact_analysis.json
  schema: .agent/schemas/impact_analysis.schema.json
validation:
  - artifact_exists
  - schema_valid
  - evidence_present
  - no_unsupported_claims
failure_policy:
  on_failure: block
  retries: 1
```

Each task should define:

```text
- task ID
- task type
- phase
- required inputs
- dependencies
- executor kind
- allowed tools
- forbidden actions
- expected artifact
- output schema
- validation rules
- failure policy
```

---

## 7. Executor Types

Classify executors honestly.

```yaml
executor_kind:
  - deterministic_script
  - deterministic_mcp_tool
  - llm_worker
  - llm_reviewer
  - human_approval
  - ci_check
```

### Deterministic executor

Use for:

```text
git status
formatting
linting
type checking
unit tests
builds
security scanners
dependency checks
static analysis
```

Example:

```yaml
id: unit-tests
executor:
  kind: deterministic_script
  command: pytest -q --junitxml=.agent/artifacts/unit_tests.xml
validation:
  - exit_code_zero
  - artifact_exists
failure_policy:
  on_failure: block
```

### LLM worker executor

Use for:

```text
intent analysis
architecture impact analysis
test gap analysis
security reasoning
performance risk assessment
migration risk assessment
final synthesis
```

Example:

```yaml
id: security-reasoning
executor:
  kind: llm_worker
  name: security-review-worker
validation:
  - schema_valid
  - evidence_present
  - reviewer_required
failure_policy:
  on_failure: block
```

### Critical rule

If a Python script or MCP tool calls another AI agent internally, treat it as `llm_worker`, not as deterministic.

---

## 8. Workflow State Model

The orchestrator must maintain durable state.

Example:

```json
{
  "workflow_id": "code-health-review",
  "run_id": "2026-05-13T120000Z",
  "status": "running",
  "repo_revision": "abc123",
  "steps": {
    "repo-snapshot": {
      "status": "accepted",
      "executor_kind": "deterministic_script",
      "outputs": [".agent/artifacts/repo_snapshot.json"],
      "started_at": "2026-05-13T12:00:01Z",
      "ended_at": "2026-05-13T12:00:03Z"
    },
    "architecture-impact-analysis": {
      "status": "pending",
      "depends_on": ["repo-snapshot", "changed-file-inventory"],
      "outputs": []
    }
  }
}
```

Valid task statuses:

```text
pending
ready
running
claimed_complete
accepted
rejected
retrying
blocked
failed
skipped_by_policy
```

Important distinction:

```text
claimed_complete != accepted
```

A worker may claim completion.
Only the orchestrator can accept completion.

---

## 9. Artifact Design

Every meaningful step should produce an artifact.

Examples:

```text
.agent/artifacts/repo_snapshot.json
.agent/artifacts/git_status.json
.agent/artifacts/lint_report.json
.agent/artifacts/typecheck_report.txt
.agent/artifacts/unit_tests.xml
.agent/artifacts/changed_files.json
.agent/artifacts/intent_analysis.json
.agent/artifacts/impact_analysis.json
.agent/artifacts/security_review.json
.agent/artifacts/test_gap_analysis.json
.agent/artifacts/final_report.md
```

Artifact rules:

```text
- Artifacts must be durable files.
- Artifacts must be referenced in `workflow_state.json`.
- LLM reasoning artifacts should use JSON schemas where possible.
- Every claim should cite evidence.
- Final reports should only use accepted artifacts.
```

---

## 10. AI Worker Result Format

AI workers should return structured results.

Example:

```json
{
  "task_id": "security-reasoning",
  "claimed_status": "complete",
  "artifact": ".agent/artifacts/security_review.json",
  "summary": "Two medium-risk security hypotheses were identified.",
  "confidence": 0.74,
  "evidence": [
    {
      "file": "src/auth/session.py",
      "lines": "88-120",
      "claim": "Session expiry logic changed."
    }
  ],
  "assumptions": [
    "The authentication middleware is used for all protected routes."
  ],
  "proposed_next_tasks": [
    {
      "id": "add-negative-session-expiry-test",
      "type": "test_gap_followup",
      "reason": "No negative test covers expired sessions after the logic change.",
      "evidence": ["tests/auth/test_session.py"]
    }
  ]
}
```

The worker may propose next tasks, but it must not directly schedule them.

---

## 11. Proposed Next Tasks Policy

Worker-generated next tasks are only proposals.

The orchestrator must decide whether to:

```text
accept
reject
merge
deduplicate
modify
defer
escalate to human review
```

Recommended validation for proposed tasks:

```text
- Does the proposed task have evidence?
- Is it within workflow scope?
- Does it duplicate an existing task?
- Does it require a mandatory gate?
- Does it need human approval?
- Does it introduce unsafe file edits?
- Is it worth the token/time cost?
```

Never allow this:

```text
worker proposes next task
  -> workflow blindly executes it
```

Use this instead:

```text
worker proposes next task
  -> orchestrator validates proposal
  -> scheduler adds accepted task
```

---

## 12. Subagent / Worker Job Contract

For AI-backed workers or subagents, use a contract.

```yaml
subagent_job:
  id: architecture-impact-review
  role: architecture-impact-analyst
  parent_step: impact-analysis
  purpose: >
    Identify affected modules, APIs, callers, and downstream risks caused by the current code changes.

  required_inputs:
    - .agent/artifacts/changed_files.json
    - .agent/artifacts/diff_stats.json
    - .agent/artifacts/repo_snapshot.json

  allowed_tools:
    - Read
    - Grep
    - Glob
    - symbol_search
    - call_graph_lookup

  forbidden_actions:
    - edit_files
    - run_tests
    - modify_workflow_state
    - produce_final_report
    - schedule_tasks_directly

  output:
    path: .agent/artifacts/impact_analysis.json
    schema: .agent/schemas/impact_analysis.schema.json

  evidence_requirements:
    - every risk must cite at least one file path
    - every API claim must cite a symbol or line range
    - every downstream impact must cite a caller, import, dependency edge, or test reference

  completion_criteria:
    - output artifact exists
    - schema validates
    - no unsupported claims

  failure_policy:
    retries: 1
    then: block_parent_step
```

---

## 13. Parent Orchestrator Responsibilities

The parent orchestrator owns:

```text
workflow decomposition
task scheduling
dependency tracking
state updates
artifact validation
worker result acceptance
conflict reconciliation
retry policy
human approval routing
final completion decision
final synthesis boundary
```

The parent orchestrator must not blindly trust:

```text
worker prose
subagent summaries
claimed test results
uncited reasoning
unstored conclusions
```

The parent accepts work only when:

```text
artifact exists
schema validates
required evidence exists
required prior steps passed
tool-use rules were followed
reviewer gate passed
```

---

## 14. Worker / Subagent Responsibilities

Workers should own:

```text
narrow task execution
bounded analysis
evidence collection
schema-conforming artifact generation
proposal of follow-up tasks
self-reported confidence
assumption disclosure
```

Workers should not own:

```text
global workflow state
mandatory gate removal
final workflow completion decision
direct scheduling authority
unbounded codebase exploration
uncontrolled file edits
final synthesis unless explicitly assigned
```

---

## 15. Validation Gates

Use different validation for different task types.

| Task Type | Validation |
|---|---|
| Script task | Exit code, output file exists |
| Test task | Exit code, test report exists |
| Security scan | Exit code, normalized report, severity policy |
| LLM analysis | Schema, evidence, assumptions, confidence |
| Subagent task | Artifact, schema, evidence, allowed-tool compliance |
| Final report | Only references accepted artifacts |
| Dynamic next task | Scope, evidence, non-duplicate, policy check |

Validation should be implemented outside the worker wherever possible.

---

## 16. Reviewer Gates

For important LLM outputs, use reviewer gates.

Reviewer should check:

```text
- Is the schema valid?
- Are claims supported by evidence?
- Are assumptions separated from facts?
- Are file and line references real?
- Did the worker stay within scope?
- Did the worker skip required inputs?
- Are there unsupported conclusions?
- Is confidence justified?
- Are proposed next tasks reasonable?
```

Example reviewer result:

```json
{
  "reviewer_task_id": "review-impact-analysis",
  "target_artifact": ".agent/artifacts/impact_analysis.json",
  "status": "accepted",
  "issues": [],
  "unsupported_claims": [],
  "required_fixes": [],
  "confidence": 0.81
}
```

---

## 17. Final Synthesis Boundary

The final report must not introduce new analysis.

It should only summarize:

```text
accepted deterministic results
accepted LLM artifacts
accepted reviewer findings
known failures
known assumptions
open risks
recommended next actions
```

Final synthesis rules:

```text
- Do not invent new findings.
- Do not claim tests passed unless test artifacts prove it.
- Do not hide failed or blocked steps.
- Do not convert assumptions into facts.
- Do not omit unresolved risks.
- Cite artifact names and evidence.
```

---

## 18. Failure Policy

Define failure behavior explicitly.

```yaml
failure_policy:
  deterministic_failure: block
  missing_artifact: block
  schema_invalid: retry_once_then_block
  unsupported_claims: reject_and_retry
  low_confidence: require_reviewer
  conflicting_worker_outputs: require_reconciliation
  stale_artifact_after_edit: invalidate_downstream
  final_report_before_validation: block
```

Important rule:

> Failed deterministic checks are facts, not suggestions.

If lint, type check, tests, or security scans fail, the workflow should not pretend the result is clean.

---

## 19. Dynamic Replanning Policy

Dynamic replanning is allowed, but controlled.

The orchestrator may add tasks when:

```text
new evidence appears
a worker identifies a credible risk
a deterministic check fails
a reviewer finds missing evidence
a changed file expands the impact surface
```

The orchestrator should not add tasks when:

```text
the task is vague
the task lacks evidence
the task duplicates an existing task
the task expands scope without reason
the task bypasses mandatory gates
the task requires unsafe edits without approval
```

---

## 20. Recommended Workflow Lifecycle

```text
1. Receive user prompt.
2. Skill starts orchestrator.
3. Orchestrator creates run ID.
4. Orchestrator reads manifest.
5. Orchestrator creates initial task graph.
6. Deterministic baseline tasks run.
7. Artifacts are written.
8. Validators accept or reject artifacts.
9. LLM workers run bounded reasoning tasks.
10. Workers produce structured artifacts.
11. Reviewers validate reasoning artifacts.
12. Workers may propose next tasks.
13. Orchestrator validates and schedules follow-up tasks.
14. Failed or blocked tasks stop unsafe downstream work.
15. Final synthesis runs only after required artifacts are accepted.
16. Final report is generated.
17. CI reruns mandatory deterministic checks.
```

---

## 21. Anti-Patterns

### Anti-pattern 1: Fat skill

```text
A very large SKILL.md tells the agent to do everything.
```

Problems:

```text
- The agent skips steps.
- The agent merges steps.
- The agent forgets requirements.
- There is no durable state.
- There is no reliable done check.
```

### Anti-pattern 2: Worker self-certification

```text
Subagent says: "I checked everything. Looks good."
```

Problems:

```text
- No artifact.
- No schema.
- No evidence.
- No validation.
```

### Anti-pattern 3: Fully dynamic autonomous loop

```text
Agent keeps creating tasks until it feels done.
```

Problems:

```text
- No stop condition.
- Token waste.
- Scope creep.
- Unpredictable output.
```

### Anti-pattern 4: AI wrapper disguised as deterministic tool

```text
Python script calls an LLM but is treated as deterministic.
```

Problems:

```text
- False confidence.
- Invalid validation model.
```

### Anti-pattern 5: Final report as fresh reasoning

```text
Final report performs new analysis not present in artifacts.
```

Problems:

```text
- Unsupported conclusions.
- Hidden assumptions.
- Unverifiable output.
```

---

## 22. Minimum Viable Implementation

Start with this:

```text
1. Thin SKILL.md
2. manifest.json
3. workflow_state.json
4. runner.py
5. deterministic steps:
   - git status
   - lint
   - type check
   - unit tests
6. one LLM worker:
   - changed-code intent analysis
7. one reviewer:
   - artifact/evidence validation
8. final_report.md generated only from accepted artifacts
```

This is enough to prove the architecture.

---

## 23. Production-Grade Implementation

Add:

```text
- dynamic task graph
- multiple AI workers
- MCP codebase tools
- schema validators
- evidence validators
- unsupported-claim detector
- subagent contracts
- reviewer gates
- stale artifact invalidation
- human approval gates
- CI-backed verification
- branch protection
- audit logs
```

Production-grade architecture:

```text
Thin Skill
  -> Orchestrator
      -> Fixed mandatory gates
      -> Dynamic task queue
      -> Deterministic executors
      -> AI worker executors
      -> Validators
      -> Reviewers
      -> Artifact store
      -> Workflow state
      -> Final synthesis boundary
  -> CI verification
```

---

## 24. Design Checklist

Before writing a complex skill, answer these questions:

```text
1. What is the real workflow entry point?
2. What mandatory gates must never be skipped?
3. Which steps are deterministic?
4. Which steps require LLM reasoning?
5. Which steps should be subagent or worker tasks?
6. What artifact does each task produce?
7. What schema validates each artifact?
8. Who decides whether a task is accepted?
9. Can a worker propose next tasks?
10. How are proposed tasks validated?
11. What happens when a task fails?
12. What happens when artifacts become stale?
13. What can run in parallel?
14. What must run sequentially?
15. What requires human approval?
16. What must CI verify?
17. What is the final synthesis allowed to say?
18. How is workflow state persisted?
19. How are unsupported claims detected?
20. How is completion proven?
```

---

## 25. One-Sentence Rule

> Do not hide complexity inside a prompt. Make it explicit through a task graph, state file, artifacts, schemas, validators, reviewer gates, and CI-backed verification.
