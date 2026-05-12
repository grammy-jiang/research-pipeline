# Designing Reliable Complex Skills for AI Coding Agents: A Practical Manual for Deterministic and LLM-Based Workflow Control

## Executive summary

The core hypothesis is substantially **confirmed**, but it needs one refinement: a reliable AI coding workflow is not just a governed workflow; it is a governed workflow with a **hard boundary between execution, reasoning, review, and synthesis**. Natural-language procedures are useful for intent and heuristics, but they are not reliable enforcement. In both Claude Code and GitHub Copilot, instruction surfaces such as skills, custom instructions, prompt files, and agent prompts are primarily **advisory controls**. Deterministic enforcement comes from hooks, permission systems, scripts, workflow runners, branch protection, required checks, and CI-backed validation. OpenAI’s Codex guidance points in the same direction: use `AGENTS.md` for repo conventions, skills for reusable workflows, MCP for tools, and external orchestration for deterministic, reviewable pipelines. citeturn11view0turn8view1turn19view0turn21view0turn15view3turn38view4turn38view0

The practical implication is blunt. If a workflow says “run lint, run tests, inspect changed files, infer intent, assess impact, identify risks, review, and summarize,” then the agent should **not** be trusted to remember and honestly execute that list purely because the list exists in a prompt. Claude Code’s own documentation explicitly says skills can remain present while the model still chooses other tools or approaches, and recommends hooks when behavior must be enforced deterministically. GitHub’s documentation similarly treats custom instructions as persistent context, not hard guarantees, while Copilot code review documentation explicitly says it should supplement, not replace, human review. Research on software-engineering agents reinforces that scaffold design and interface design matter, and that simpler fixed pipelines such as Agentless can rival or outperform more open-ended agent loops by constraining decision points and validating patches explicitly. citeturn8view1turn11view0turn23view1turn37search2turn37search0

The recommended operating model is therefore:

1. Put **deterministic work** behind scripts, typed tool interfaces, or CI jobs.
2. Treat **LLM reasoning** as bounded analysis steps with explicit contracts, schemas, evidence requirements, and reviewer gates.
3. Pass work between steps using **artifacts and workflow state**, not memory.
4. Prohibit the final response from introducing new facts that are not grounded in prior artifacts.
5. Make “done” a **validated state transition**, not a sentence the model writes. citeturn11view0turn21view0turn15view10turn35search11turn38view0

The phrase “Codebase” in the prompt is ambiguous. This manual interprets it as **codebase-aware AI coding agent workflows** and uses Codex-style `AGENTS.md`, repository indexing, symbol- and graph-aware retrieval, static analysis, and artifact-driven orchestration as the closest concrete implementation pattern. That interpretation is consistent with current GitHub and OpenAI documentation on repository indexing, agentic memory, `AGENTS.md`, skills, and orchestrated MCP-based workflows. citeturn31view0turn31view1turn38view1turn38view0

### Key terminology

| Term | Practical meaning | Control type | Should it be trusted as enforcement? |
|---|---|---|---|
| Skill | Reusable instructions and optional bundled assets for a recurring task | Advisory by default | No |
| Custom instruction | Always-on project, user, org, or path-scoped guidance | Advisory | No |
| Prompt file | Manually invoked reusable prompt template for a specific interaction | Advisory | No |
| Subagent | Isolated worker with its own context, prompt, and tool scope | Semi-deterministic isolation, not content correctness | Partly |
| Hook | Lifecycle-triggered script/prompt/HTTP/MCP action | Deterministic or semi-deterministic | Yes, if implemented well |
| MCP tool | Typed tool interface to external systems | Deterministic interface, non-deterministic choice of use unless enforced | Partly |
| Workflow runner | External orchestrator that executes steps, writes state, validates outputs | Deterministic | Yes |
| CI pipeline | External verification pipeline with required checks and branch rules | Deterministic | Yes |
| Guardrail | Any mechanism that blocks, constrains, or validates actions or outputs | Depends on implementation | Yes, if externalized |

This terminology table is a synthesis of current Claude Code, GitHub Copilot, and Codex documentation, plus the SWE-agent and Agentless research framing around agent-computer interfaces and fixed-step repair pipelines. citeturn9view1turn10view0turn19view0turn21view0turn38view1turn38view2turn38view0turn37search2turn37search0

## Platform control surfaces and what they can actually enforce

### Platform comparison

| Platform / ecosystem | Advisory surfaces | Deterministic or semi-deterministic surfaces | Best place for mandatory checks | Main limitation |
|---|---|---|---|---|
| Claude Code | `CLAUDE.md`, skills, slash commands, subagent prompts | Hooks, permission modes/rules, MCP tools with typed schemas, external scripts/CI | Hook-triggered runner + CI | Skills are prompt-based; `allowed-tools` pre-approves but does not restrict all other tools |
| GitHub Copilot | `.github/copilot-instructions.md`, `.github/instructions/*.instructions.md`, `AGENTS.md`, prompt files, custom agent prompts | `.github/hooks/*.json`, cloud-agent environment setup, GitHub Actions, required checks, branch protection, PR templates | GitHub Actions + branch protection + hooks | Instructions are contextual, not authoritative; code review is explicitly non-final |
| Codebase-aware workflows using Codex / `AGENTS.md` patterns | `AGENTS.md`, skills, custom agent prompts | MCP-backed orchestration, external runners, Agents SDK workflows, read-only / tool-scoped subagents | External runner + MCP + CI | `AGENTS.md` and skills guide behavior but do not themselves guarantee step execution |

Notes: Claude Code behavior comes from its feature-overview, skills, hooks, subagents, and permission docs; GitHub Copilot behavior comes from its customization, hooks, cloud-agent, and branch-protection docs; Codex behavior comes from `AGENTS.md`, skills, subagents, MCP, and Agents SDK docs. citeturn9view1turn8view1turn11view0turn5view3turn5view4turn19view0turn21view0turn21view2turn15view3turn15view10turn38view1turn38view2turn38view6turn38view0

### Claude Code guidance

Claude Code’s extension model is unusually explicit about what belongs where. `CLAUDE.md` is persistent session-start context and survives compaction; skills are on-demand reusable instructions or workflows; subagents isolate noisy work into separate contexts; MCP adds tools; hooks run automatically at lifecycle points; permission modes and rules apply outside chat, not by merely asking Claude to behave differently. Anthropic’s docs explicitly say to start with `CLAUDE.md` for stable conventions, add skills for repeatable procedures, add MCP for external systems, add subagents for isolation, and add hooks when something must happen every time. That is the right control hierarchy. citeturn6view3turn9view1turn10view3

For workflow control, the key Claude Code fact is that **skills are not enforcement**. A skill’s content is loaded into context and can influence behavior, but Anthropic’s documentation says that if skill content appears to stop influencing behavior, the model may simply be choosing other tools or approaches; their own recommendation is to strengthen descriptions or use hooks for deterministic enforcement. Likewise, skill `allowed-tools` only grants permission while the skill is active; it does **not** restrict the rest of the tool surface. That means a long `SKILL.md` with seven or eight mandatory steps is structurally fragile unless an external runner or hooks make those steps observable and blockable. citeturn8view1turn8view0

Claude Code hooks are the real governance surface. They fire at fixed lifecycle events including `UserPromptExpansion`, `PreToolUse`, `PostToolUse`, `PostToolBatch`, `SubagentStart/Stop`, and `Stop`. `PreToolUse` can allow, deny, ask, defer, and even modify tool input; `UserPromptExpansion` can block a slash command before it expands; `Stop` can prevent the agent from declaring completion. Anthropic also documents that when multiple hooks match, they all run, and then the outputs are merged; for `PreToolUse`, the most restrictive decision wins. That is powerful, but it also means side effects such as logging still happen even when another hook denies execution. In practice, Claude hooks are suitable for policy gates, audit logging, post-edit formatting, mandatory state checks, and stop-time completion validation. citeturn7view0turn7view2turn11view0

Subagents are useful, but only for isolation and specialization, not truth. Claude’s built-in Explore and Plan agents are read-only, and custom subagents can have restricted tools, custom permissions, skills, and hooks. That makes them ideal for controlled roles such as “read-only architecture impact analyst” or “final report synthesizer with no write access.” They are not a substitute for validation; they are a way to narrow the blast radius and reduce context pollution. citeturn5view3turn9view3turn10view0turn38view6

**Recommended Claude pattern:** use `CLAUDE.md` for stable repo norms, a **thin skill** as the entry point, hooks for mandatory lifecycle checks, an external runner for step state and artifacts, MCP for structured codebase services when needed, and CI for final verification. Do not put business-critical compliance logic in prose inside `SKILL.md`. citeturn6view3turn11view0turn38view4

### GitHub Copilot guidance

GitHub Copilot has a broader menu of customization surfaces, but the same control reality. Repository-wide and path-specific custom instructions are automatically added to requests in scope, and GitHub.com additionally supports organization-level instructions and “agent instructions” via `AGENTS.md`, `CLAUDE.md`, or `GEMINI.md` for specific features. Prompt files are separate: they are reusable, manually invoked prompts for one chat interaction, and are only available in VS Code, Visual Studio, and JetBrains IDEs while still in public preview. Custom agents are specialized agent profiles that specify prompt, tools, and MCP servers. Skills are folders of instructions, scripts, and resources that Copilot can load when relevant. citeturn16view0turn17view0turn17view1turn19view0turn25view4turn15view4turn29view2

That is a rich advisory layer, but GitHub’s own docs repeatedly point engineers toward stronger controls for mandatory behavior. Custom instructions are context, not enforcement. GitHub’s docs warn that conflicting instructions can produce non-deterministic outcomes, and Copilot code review only reads the first 4,000 characters of any custom instruction file. Prompt files are manual and therefore cannot be relied on for mandatory execution unless another system forces their use. Copilot code review itself is explicitly described as a supplement to human review and can miss issues or hallucinate false positives. citeturn17view2turn22search1turn23view1

GitHub’s stronger control surfaces are the cloud-agent runtime, hooks, GitHub Actions, required checks, and branch protection. Copilot cloud agent runs in a GitHub Actions-powered ephemeral environment, can research, plan, change code on a branch, and optionally open a PR. GitHub recommends pre-installing dependencies and tools with setup steps because trial-and-error environment setup is slow and unreliable given the non-deterministic nature of LLMs. Hooks for Copilot cloud agent and Copilot CLI are repository-scoped JSON files in `.github/hooks`, can inspect JSON event data, and specifically include `preToolUse` for allowing or denying tool execution. Branch protection and required status checks remain the final merge gate. citeturn15view3turn21view4turn27search17turn21view0turn21view1turn21view2turn15view10turn35search11

GitHub’s codebase-aware features are useful but should be treated as **retrieval aids**, not proof. Repository indexing provides semantic code search for Copilot Chat and Copilot cloud agent, and GitHub says indexed repository content is not used for model training. Copilot Memory stores repository-scoped memories with citations and re-validates those citations against the current codebase before reuse. That is a strong pattern for knowledge persistence, but it still does not replace explicit artifact handoff for a multi-step governed workflow. citeturn31view0turn31view1

**Recommended GitHub pattern:** use repository instructions for broad rules, path-specific instructions for localized build/test conventions, prompt files for manual one-off analyses, custom agents for role specialization, hooks for deterministic policy checks, GitHub Actions for every executable validation step, and branch protection/required checks as the merge gate. Treat Copilot instructions as steering, not control. citeturn19view0turn21view0turn15view10turn23view1

### Codebase-aware workflows and Codex-style `AGENTS.md`

OpenAI’s current Codex guidance is consistent with the same thesis. Codex reads `AGENTS.md` before doing work and builds an instruction chain from global and project scopes; skills use progressive disclosure and load fully only when selected; subagents exist for explicit parallelization and specialization; MCP connects structured tools and context; and the Agents SDK can orchestrate Codex through MCP into “deterministic, reviewable workflows” with handoffs, guardrails, and traces. OpenAI’s customization guidance is explicit: use `AGENTS.md` for repo conventions, pre-commit hooks and linters to enforce those rules, MCP when workflows need external systems, and subagents for noisy or specialized work. citeturn38view1turn38view2turn38view6turn38view7turn38view0turn38view4

This matters because it legitimizes a broad engineering pattern: **repository instructions are for policy and conventions; workflows belong in an orchestrator**. The AGENTS system is useful because it is hierarchical and repo-aware, but it is still guidance. The most reliable Codex pattern is to keep `AGENTS.md` short and durable, keep skills focused and selectively loadable, use code review instructions referenced from `AGENTS.md`, and move actual step-state execution into an external runner or an MCP-backed orchestrator. citeturn38view1turn38view2turn38view5turn38view0

## Workflow control model

### Deterministic vs non-deterministic task classification

The table below is the recommended classification framework for the task types in the prompt. It is a synthesis, not a vendor-provided matrix.

| Task type | Class | Scriptable | Exit-code validation | Schema needed | Human / 2nd LLM review | Block downstream on failure | Primary artifact | Recommended control |
|---|---|---:|---:|---:|---|---:|---|---|
| `git status` | Deterministic | Yes | Yes | Optional | No | Usually yes | `git_status.json` | Runner or direct tool |
| Dependency installation check | Deterministic | Yes | Yes | Optional | No | Yes | `env_probe.json` | Runner / CI |
| Formatting check | Deterministic | Yes | Yes | No | No | Yes | `format_report.txt` | Runner / CI / post-edit hook |
| Linting | Deterministic | Yes | Yes | No | No | Yes | `lint_report.json` | Runner / CI |
| Type checking | Deterministic | Yes | Yes | No | No | Yes | `typecheck_report.json` | Runner / CI |
| Unit tests | Deterministic | Yes | Yes | Optional | No | Yes | `unit_test_report.xml` | Runner / CI |
| Integration tests | Deterministic | Yes | Yes | Optional | Sometimes | Yes | `integration_test_report.xml` | Runner / CI |
| Security scan | Deterministic or semi-deterministic depending on tool | Yes | Usually | Yes for normalized output | Human for high severity | Yes | `security_scan.json` | Runner / CI |
| Changed-file summarization | Semi-deterministic | Partly | Not reliably | Yes | Light review | Yes | `changed_files.json` + `diff_stats.json` | Script first, optional LLM summary second |
| Change intent inference | LLM reasoning | No | No | Yes | Yes | Yes | `intent_analysis.json` | Contracted LLM step + reviewer |
| Architecture impact analysis | LLM reasoning augmented by retrieval | Partly | No | Yes | Yes | Yes | `impact_analysis.json` | LLM + symbol / graph tools + reviewer |
| API compatibility analysis | Semi-deterministic + LLM reasoning | Partly | Partial | Yes | Yes | Yes | `api_compat.json` | Static diff + LLM interpretation + reviewer |
| Security reasoning | LLM reasoning | No | No | Yes | Yes, preferably specialist reviewer | Yes | `security_hypotheses.json` | LLM + evidence contract + reviewer |
| Performance risk analysis | LLM reasoning | No | No | Yes | Yes | Usually yes | `performance_risks.json` | LLM + profiler/static evidence if available |
| Test coverage gap analysis | LLM reasoning augmented by deterministic reports | Partly | No | Yes | Yes | Yes | `test_gap_analysis.json` | Coverage data + changed inventory + reviewer |
| Migration risk assessment | LLM reasoning | No | No | Yes | Yes | Yes | `migration_risks.json` | Contracted LLM step + reviewer |
| Final report generation | LLM synthesis | No | No | Yes | Yes | Must not run early | `final_report.md` | Read-only synthesizer over prior artifacts |

This classification aligns with the official distinction between prompt-based guidance and deterministic hooks/CI, and with SWE-agent and Agentless findings that interface design and fixed validation stages are decisive for reliability. citeturn11view0turn21view0turn38view0turn37search2turn37search0

### Where workflow control should live

Workflow control should live in different places depending on the failure cost:

- **Inside the skill or instruction file:** only task framing, role definition, step names, and artifact expectations.
- **Inside the agent:** light planning and bounded reasoning over artifacts.
- **Inside a script or workflow runner:** deterministic steps, state transitions, retries, artifact writing, and schema validation.
- **Inside an MCP server:** structured external capabilities such as symbol graph lookup, call graph traversal, CI log retrieval, or issue/PR mutation.
- **Inside hooks:** lifecycle policy checks, required tool approval/denial, post-edit formatting, audit logging, and stop-time completion checks.
- **Inside CI and branch protection:** final executable verification and merge gating.
- **Inside an external orchestrator:** complex multi-agent or multi-step pipelines where auditability and reruns matter. citeturn11view0turn21view0turn15view10turn35search11turn38view0

A plain natural-language skill should **never** be trusted with: mandatory tool invocation, mandatory step ordering across long workflows, truthful completion claims, state persistence across compaction or context loss, exact output structure without validation, or merge authority. Claude’s own docs say `CLAUDE.md` survives compaction while conversation-only instructions may be lost, and skill content can load yet still not control every later choice. GitHub’s docs warn about non-deterministic resolution of instruction conflicts and limitations of code review. citeturn6view3turn8view1turn17view2turn23view1

### Step handoffs, artifacts, and workflow state

A reliable mixed workflow should hand off through files, not through conversational memory. The recommended artifact flow is:

```text
repo_snapshot.json
  -> env_probe.json
  -> deterministic_checks/
      lint_report.json
      typecheck_report.json
      unit_test_report.xml
      integration_test_report.xml
      security_scan.json
  -> changed_files.json
  -> diff_stats.json
  -> intent_analysis.json
  -> impact_analysis.json
  -> risk_hypotheses.json
  -> test_gap_analysis.json
  -> reviewer_reports/
  -> final_report.md
```

The state model should track each step as a durable record:

```json
{
  "run_id": "2026-05-12T103015Z-main-abc123",
  "repo_rev": "abc123def456",  # pragma: allowlist secret
  "manifest_version": "1.0",
  "status": "running",
  "steps": {
    "lint": {
      "kind": "deterministic",
      "status": "passed",
      "attempt": 1,
      "started_at": "2026-05-12T10:31:02Z",
      "ended_at": "2026-05-12T10:31:16Z",
      "exit_code": 0,
      "outputs": ["artifacts/lint_report.json"],
      "checksum": "sha256:..."
    },
    "intent_analysis": {
      "kind": "llm",
      "status": "blocked",
      "blocked_by": ["changed_inventory"],
      "attempt": 0,
      "required_inputs": [
        "artifacts/changed_files.json",
        "artifacts/diff_stats.json"
      ],
      "outputs": []
    }
  }
}
```

The point is operational, not cosmetic. A later LLM step should not be able to say “tests passed” unless the runner or CI already marked the relevant step as passed and produced its artifact. The final synthesis step should run in a **read-only role** and be forbidden from executing new tests or making new edits; it may only summarize prior artifacts. That creates a clean **final synthesis boundary**. citeturn11view0turn21view0turn15view10turn38view0

### LLM reasoning step contracts

Every non-deterministic analysis step should be specified as a contract, not a vague instruction. The minimum contract fields are:

```yaml
step_id: architecture-impact
purpose: Assess which modules, APIs, and call sites are affected by the proposed change.
required_inputs:
  - artifacts/changed_files.json
  - artifacts/diff_stats.json
  - artifacts/repo_snapshot.json
allowed_tools:
  - symbol_search
  - find_references
  - dependency_graph
forbidden_actions:
  - write_files
  - run_tests
required_output_schema: impact_analysis.schema.json
evidence_requirements:
  - every claim must cite file paths and line ranges or graph edges
  - every risk must link to at least one changed file
confidence_scoring:
  scale: 0.0-1.0
assumption_handling:
  - separate confirmed facts from assumptions
completion_criteria:
  - output validates against schema
  - no unsupported claims
reviewer_criteria:
  - evidence coverage >= 0.8
  - no orphan risks without evidence
failure_policy:
  retries: 1
  then: block
```

Bad prompts include “think carefully,” “analyze the code,” “check if there are issues,” or “review everything.” Good contracts say exactly what evidence to inspect, what tool classes are allowed, what schema to return, what is forbidden, and what counts as done. The LLM is not asked to be vaguely intelligent; it is asked to produce a typed deliverable. citeturn21view2turn38view0turn38view5

## Reviewer, guardrail, and failure design

### What validation actually means

There are four different validation problems, and teams routinely conflate them:

1. **Correctness validation:** is the substantive reasoning right?
2. **Process validation:** did the step use required evidence and required tools?
3. **Format validation:** is the output schema-valid and machine-consumable?
4. **Workflow validation:** was every mandatory step executed and recorded before downstream steps ran?

The first problem is the hardest and often cannot be fully automated. The second, third, and fourth problems are tractable and should be automated aggressively. This is the crucial refinement to the user’s hypothesis: for LLM reasoning, you can rarely prove that the hidden reasoning is right, but you can often prove that the evidence contract was followed and that the output is reviewable and traceable. GitHub’s code review docs explicitly highlight missed issues, false positives, and insecure suggested code; that is exactly why reviewer gates must validate evidence and enforce follow-up review rather than treating the first model pass as authoritative. citeturn23view1turn37search0turn37search2

### Reviewer stack

A robust reviewer stack for every major LLM reasoning step should be:

| Layer | Purpose | Blocks the step? |
|---|---|---:|
| JSON/schema validator | Ensure fields, enums, required arrays, and types match contract | Yes |
| Evidence validator | Ensure every claim references permitted source artifacts and file/line evidence | Yes |
| Unsupported-claim detector | Reject claims with no citation or no matching artifact | Yes |
| Tool-use verifier | Confirm required tools or deterministic precursor steps actually ran | Yes |
| Second-pass critic LLM | Assess coverage, contradictions, and weak assumptions | Usually |
| Human approval gate | Review high-risk steps such as security, migrations, or public API changes | Sometimes or always |
| CI-backed verifier | Re-run deterministics and validate artifact set before merge | Yes |

This reviewer stack is consistent with Claude and Copilot hooks, branch protection, and external orchestrators. The important point is that the reviewer is not “another opinion” only; it is a gate over evidence, contract compliance, and workflow completeness. citeturn11view0turn21view0turn15view10turn35search11turn38view0

A practical reviewer policy by step looks like this:

- **Intent analysis reviewer:** check that outputs reference changed files and separate facts from assumptions.
- **Architecture impact reviewer:** check symbol-level evidence coverage and that affected callers or downstream consumers are surfaced.
- **Security reasoning reviewer:** require concrete threat hypotheses tied to reachable code paths; low-confidence speculative items stay labeled as hypotheses.
- **Test gap reviewer:** require explicit mapping from changed behavior to missing or existing test coverage.
- **Final report reviewer:** reject any claim that cites non-existent artifacts or invents new findings not present upstream. citeturn31view1turn38view5turn23view1

### Failure modes and mitigations

| Failure mode | Root cause | Detection | Prevention | Recommended mechanism |
|---|---|---|---|---|
| Step skipping | Long prompt compressed into gist | Missing artifact/state entry | External manifest and runner | Runner + state file |
| Tool skipping | Agent chooses different path | Tool log lacks required call | Hook or runner requirement | Hook + tool-use verifier |
| Tool hallucination | Model claims to have run a tool | No matching log or artifact | Never trust narration | Hook logs + artifact check |
| Claiming tests passed without running tests | Conversational bluffing | No test artifact / exit code | Final synthesis read-only | CI + state gate |
| Over-relying on memory | Context truncation / compaction | Orphan claims without artifacts | Every step reads files | Artifact handoff |
| Over-broad context loading | “Read whole repo” behavior | High token use / shallow analysis | Progressive disclosure | Repo map + symbol search |
| Under-reading affected files | Narrow search path | Missing callers/interfaces | Structured retrieval checklist | Graph/symbol tools |
| Ignoring failed deterministic checks | Model treats failure as advisory | State shows failed step but workflow continues | Block transitions on failure | Runner policy |
| Treating assumptions as facts | Poor output schema | Assumption field empty but speculative language present | Contract demands separation | Schema + reviewer |
| Producing final synthesis before artifacts exist | Eager summarization | Missing prerequisites | Final step blocks on manifest | Stop hook / runner |
| Confusing reviewed with verified | Review comment mistaken for proof | No executable evidence | Separate review from validation | Reviewer labels + CI |
| Failing to separate facts, assumptions, recommendations | Weak prompt design | Mixed fields in output | Structured schema | Contract + validator |
| Hiding uncertainty | Incentive to sound decisive | No confidence or caveats | Require confidence/assumptions fields | Schema + reviewer |
| Failing to rerun checks after edits | Checks tied to earlier state | Artifact timestamps/hash mismatch | Invalidate downstream on edit | Post-edit hook + runner |
| Losing state across context compaction or session changes | Memory-only workflow | Missing continuation markers | Durable state on disk | State file + logs |

These failure modes are the practical expression of what the platform docs and research already imply: prompt-only control decays, while logs, artifacts, hooks, and fixed-state transitions remain inspectable. citeturn6view3turn8view1turn11view0turn21view0turn31view1turn37search2turn37search0

## Implementation blueprints and examples

### Design comparison

| Design | Advantages | Failure modes | Best use case | Verdict |
|---|---|---|---|---|
| Fat skill | Fast to author; easy to distribute; good for lightweight playbooks | Step merging, step skipping, tool skipping, fabricated completion, weak audit trail | Small advisory workflows with low failure cost | Acceptable only for short non-critical flows |
| Thin skill + workflow runner | Clear separation of prompt and control; repeatable; easy to validate; friendly to CI | Extra engineering effort; runner becomes a maintained product | Default choice for serious mixed workflows | Best default |
| Skill + hooks + MCP + CI | Highest control, auditability, external tools, lifecycle enforcement, merge gating | Highest complexity and operational overhead | Team or org-scale governed workflows | Best production architecture |

This comparison is aligned with vendor docs that place hooks, CI, and external orchestration on the enforcement side, and with Agentless-style evidence that fixed-stage pipelines can outperform more free-form agent loops for reliable issue resolution. citeturn11view0turn21view0turn38view0turn37search0

### Reference architecture

```text
repo/
  .agent/
    workflow_state.json
    manifest.json
    logs/
    artifacts/
    steps/
  .claude/
    skills/
      code-health-review/
        SKILL.md
    settings.json
  .github/
    copilot-instructions.md
    instructions/
      backend.instructions.md
    hooks/
      policy.json
    prompts/
      review-code.prompt.md
    workflows/
      code-health.yml
  AGENTS.md
  tools/
    agent_workflow/
      runner.py
      steps/
      validators/
      reviewers/
      mcp_tools/
```

**What each part does**

- `.agent/manifest.json`: canonical ordered workflow definition.
- `.agent/workflow_state.json`: durable state machine for the current or latest run.
- `.agent/artifacts/`: evidence handoff store.
- `.agent/logs/`: tool-use and reviewer logs.
- `.claude/skills/.../SKILL.md`: Claude entry point.
- `.claude/settings.json`: Claude hooks and permission rules.
- `.github/copilot-instructions.md`: broad Copilot norms.
- `.github/instructions/*.instructions.md`: path-scoped Copilot rules.
- `.github/hooks/*.json`: Copilot CLI / cloud-agent hook definitions.
- `.github/prompts/*.prompt.md`: manual analysts’ prompt files where useful.
- `.github/workflows/code-health.yml`: CI-backed verification.
- `AGENTS.md`: codebase-aware instructions consumed by Codex-style agents and supported Copilot surfaces.
- `tools/agent_workflow/runner.py`: orchestrator that owns state, retries, artifacts, and schema validation.

### Example entry surfaces

**Claude `SKILL.md`**

```markdown
---
name: code-health-review
description: Run the governed code health review workflow for the current repository state.
disable-model-invocation: true
allowed-tools: Bash(python tools/agent_workflow/runner.py *) Read Grep Glob
---

Execute `python tools/agent_workflow/runner.py --manifest .agent/manifest.json`.

Rules:
- Do not claim any step is complete unless the workflow state marks it passed.
- Do not summarize findings until `.agent/artifacts/final_report.md` exists.
- If the runner reports blocked or failed steps, report the exact blocking step IDs.
```

**`AGENTS.md`**

```markdown
# Repository agent instructions

Use `.agent/manifest.json` as the source of truth for ordered workflow steps.

Never treat conversational memory as workflow state.
Always read required inputs from `.agent/artifacts/` and write outputs back there.

Facts, assumptions, and recommendations must be separated.
Any final synthesis must cite artifact file names and supporting file:line evidence.
```

**`.github/copilot-instructions.md`**

```markdown
Follow repository conventions, but do not treat prose as proof of completion.

For governed reviews:
- use `.agent/manifest.json` as the workflow definition
- prefer deterministic reports from linters, tests, type checks, and scanners
- write structured JSON artifacts for all reasoning steps
- never state that checks passed unless artifacts show a passing result
```

### Example workflow definitions

**Workflow manifest**

```json
{
  "name": "code-health-review",
  "version": "1.0",
  "entrypoint": "runner.py",
  "steps": [
    {"id": "repo_snapshot", "kind": "deterministic", "script": "steps/repo_snapshot.py"},
    {"id": "deterministic_checks", "kind": "deterministic", "script": "steps/deterministic_checks.py"},
    {"id": "changed_inventory", "kind": "deterministic", "script": "steps/changed_inventory.py"},
    {"id": "intent_analysis", "kind": "llm", "contract": "steps/intent_analysis.yaml"},
    {"id": "impact_analysis", "kind": "llm", "contract": "steps/impact_analysis.yaml"},
    {"id": "test_gap_analysis", "kind": "llm", "contract": "steps/test_gap_analysis.yaml"},
    {"id": "reviewer_pass", "kind": "reviewer", "contract": "steps/reviewer.yaml"},
    {"id": "final_report", "kind": "llm", "contract": "steps/final_report.yaml"}
  ]
}
```

**Deterministic step definition**

```yaml
id: deterministic_checks
kind: deterministic
commands:
  - "ruff check . --output-format=json > .agent/artifacts/lint_report.json"
  - "mypy . --hide-error-context --no-color-output > .agent/artifacts/typecheck_report.txt"
  - "pytest -q --junitxml=.agent/artifacts/unit_test_report.xml"
on_failure: block
```

**LLM reasoning step definition**

```yaml
id: test_gap_analysis
kind: llm
required_inputs:
  - .agent/artifacts/changed_files.json
  - .agent/artifacts/unit_test_report.xml
  - .agent/artifacts/impact_analysis.json
output_schema: schemas/test_gap_analysis.schema.json
required_fields:
  - covered_paths
  - uncovered_risks
  - recommended_tests
  - evidence
confidence_field: confidence
forbidden_actions:
  - edit_files
  - run_commands
```

**Reviewer step definition**

```yaml
id: reviewer_pass
kind: reviewer
checks:
  - schema_validation
  - evidence_validation
  - unsupported_claim_detection
  - prerequisite_artifact_check
  - final_synthesis_boundary_check
on_failure: block
```

**Final report template**

```markdown
# Code Health Review

## Verified deterministic results
- lint:
- type check:
- tests:
- security scan:

## Confirmed findings
- item:
  evidence:
  confidence:

## Assumptions and uncertainties
- item:
  why uncertain:

## Recommended follow-up
- item:
  blocking: true|false
```

### Claude Code implementation blueprint

Use `CLAUDE.md` only for persistent repo facts and stable conventions. Use one thin skill that calls the runner. Use hooks for lifecycle control:

- `UserPromptExpansion`: if `/code-health-review` is called directly, ensure the manifest exists and attach run context.
- `PreToolUse`: deny unapproved write or bash tools outside the runner for governed review sessions.
- `PostToolUse` on `Edit|Write`: run formatter or mark downstream deterministic artifacts stale.
- `Stop`: block final output if required artifacts are missing or any blocking step status is not `passed`.

A minimal `.claude/settings.json` sketch looks like this:

```json
{
  "hooks": {
    "UserPromptExpansion": [
      {
        "matcher": "code-health-review",
        "hooks": [
          {
            "type": "command",
            "command": "python tools/agent_workflow/check_manifest.py"
          }
        ]
      }
    ],
    "Stop": [
      {
        "matcher": "",
        "hooks": [
          {
            "type": "command",
            "command": "python tools/agent_workflow/check_completion.py"
          }
        ]
      }
    ]
  }
}
```

This blueprint fits Claude’s actual mechanics: hooks are the deterministic lifecycle surface; skills are not. Known limitations remain: skill behavior is advisory, `allowed-tools` is permission-granting not restriction, and hook side effects still occur even when a different hook blocks the action. citeturn11view0turn8view0turn8view1

### GitHub Copilot implementation blueprint

For Copilot, the entry point is broader because the platform spans IDE, CLI, cloud agent, code review, and GitHub.com:

- `.github/copilot-instructions.md`: broad repository norms.
- `.github/instructions/**/*.instructions.md`: path-specific build/test notes.
- `.github/agents/reviewer.md`: optional specialist custom agent with restricted tools.
- `.github/hooks/policy.json`: deterministic `preToolUse`, `sessionStart`, and `sessionEnd` policies for Copilot CLI and cloud agent.
- `copilot-setup-steps.yml`: preinstall linters, test deps, and scanners in the cloud-agent environment.
- `.github/workflows/code-health.yml`: mandatory CI.
- Branch protection: require `code-health`, required reviews, and optionally merge queue.

A minimal hook sketch:

```json
{
  "hooks": [
    {
      "event": "preToolUse",
      "run": "./tools/agent_workflow/policy_gate.sh"
    },
    {
      "event": "sessionStart",
      "run": "./tools/agent_workflow/session_start.sh"
    },
    {
      "event": "sessionEnd",
      "run": "./tools/agent_workflow/archive_run.sh"
    }
  ]
}
```

A minimal CI skeleton:

```yaml
name: code-health
on:
  pull_request:
  workflow_dispatch:

jobs:
  verify:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: pip install -r requirements-dev.txt
      - run: python tools/agent_workflow/runner.py --ci --manifest .agent/manifest.json
      - run: python tools/agent_workflow/validate_artifacts.py
```

This is the correct enforcement posture because GitHub’s own docs place hard guarantees in hooks, Actions, required checks, and branch rules, while treating instructions, prompt files, and code review as assistive layers. citeturn21view0turn21view4turn15view10turn35search8turn23view1

### Codebase-aware workflow blueprint

A codebase-aware workflow should **not** ask the model to “read the whole codebase.” It should force progressive disclosure:

1. Build or fetch a **repo map** and changed-file inventory.
2. Use **semantic search** or symbol search to locate likely impact surfaces.
3. Expand to definitions, references, callers, imports, and adjacent tests.
4. Pull static-analysis or graph artifacts where available.
5. Only then run reasoning steps with explicit evidence contracts.

For this workflow, an MCP tool should expose structured codebase operations instead of raw prose. Example:

```yaml
tool: impact_graph.get_dependents
input:
  symbol: "payments.RefundService.refund"
output:
  schema:
    callers: []
    downstream_modules: []
    api_surfaces: []
    evidence: []
```

This pattern is aligned with GitHub repository indexing, Copilot’s semantic code search, Copilot Memory’s validated citations, and Codex’s MCP and orchestrated workflow guidance. citeturn31view0turn31view1turn38view7turn38view0

### Recommended minimum viable and production-grade implementations

**Minimum viable implementation**

- Thin skill or command that launches one runner.
- One manifest file and one state file.
- Deterministic checks executed by script.
- Two or three LLM contracts at most.
- Final report step restricted to prior artifacts.
- CI reruns deterministic checks and validates artifact presence.

**Production-grade implementation**

- Thin entry skill or custom agent.
- Hooks that block out-of-policy tool use and block stop-time completion when artifacts are missing.
- MCP services for symbol search, dependency/call graph lookup, and CI log summarization.
- Second-pass reviewer agents for all reasoning steps.
- Signed or hashed artifacts and immutable run logs.
- Branch protection, required checks, required reviews, merge queue where useful.
- PR template requiring artifact IDs, risk level, and unresolved assumptions.
- Human approval gates for security, migrations, and public API changes. citeturn11view0turn21view0turn15view10turn35search2turn38view0

## Open questions and annotated bibliography

The main unresolved engineering question is not whether governed workflows are better; that part is settled. The unresolved question is **how much correctness validation for LLM reasoning can be automated before the cost exceeds the benefit**. Current platform docs are strong on tool control, context control, auditability, and merge gating, but much weaker on proving that a subtle architecture or security conclusion is actually correct. The best current answer is to automate evidence, format, and workflow validation aggressively, then reserve human review for high-consequence reasoning. Another open trade-off is complexity: the production architecture is materially safer, but the operational tax is real, especially for small teams. citeturn23view1turn38view0turn37search0turn37search2

### Annotated bibliography

| Source | Vendor / author | Date | Contribution | Limitation / bias |
|---|---|---|---|---|
| urlExtend Claude Codeturn9view1 | entity["organization","Anthropic","ai company"] / urlAnthropichttps://www.anthropic.com | Live docs | Best official map of `CLAUDE.md`, skills, subagents, hooks, MCP, and feature layering | Product documentation; describes intended behavior, not failure rates |
| urlAutomate workflows with hooksturn11view0 | urlAnthropichttps://www.anthropic.com | Live docs | Strongest primary source on deterministic lifecycle enforcement in Claude Code | Focused on hooks, not end-to-end governed workflow design |
| urlExtend Claude with skillsturn2view1 | urlAnthropichttps://www.anthropic.com | Live docs | Clarifies skill lifecycle, `allowed-tools`, manual vs model invocation, and why skills are not hard enforcement | Still a vendor view of the feature |
| urlCreate custom subagentsturn5view3 | urlAnthropichttps://www.anthropic.com | Live docs | Useful on isolation, tool restrictions, and read-only exploration patterns | Does not claim correctness guarantees |
| urlCopilot customization cheat sheetturn19view0 | entity["company","GitHub","developer platform"] / urlGitHubhttps://github.com | Live docs | Best GitHub source for comparing custom instructions, prompt files, custom agents, hooks, skills, subagents, and MCP across surfaces | High-level reference, not a design manual |
| urlAdding repository custom instructions for GitHub Copilotturn15view0 | urlGitHubhttps://github.com | Live docs | Primary source on repository-wide and path-specific instruction behavior and precedence | Instructions are advisory; docs do not frame them as enforcement |
| urlAbout hooksturn21view0 and urlHooks configurationturn21view2 | urlGitHubhttps://github.com | Live docs | Primary source on Copilot CLI / cloud-agent hooks, JSON event payloads, and deterministic tool controls | Current support is limited to specific Copilot surfaces |
| urlAbout GitHub Copilot cloud agentturn15view3 and urlConfigure the development environmentturn21view4 | urlGitHubhttps://github.com | Live docs | Defines the GitHub Actions-powered cloud-agent environment and why preinstall/setup steps matter for reliability | Vendor implementation details may change over time |
| urlResponsible use of GitHub Copilot code reviewturn23view1 | urlGitHubhttps://github.com | Live docs | Best official statement of code-review limitations, false positives, and need for human review | Discusses limitations, not workflow architecture |
| urlIndexing repositories for GitHub Copilotturn31view0 and urlAbout agentic memory for GitHub Copilotturn31view1 | urlGitHubhttps://github.com | Live docs | Strong evidence for codebase-aware retrieval, semantic search, and memory validated by citations | Retrieval support is not the same as guaranteed reasoning correctness |
| urlCustom instructions with AGENTS.mdturn38view1 | entity["organization","OpenAI","ai company"] / urlOpenAIhttps://openai.com | Live docs | Best primary source on `AGENTS.md` precedence and discovery in Codex | Guidance surface, not enforcement surface |
| urlAgent Skillsturn38view2, urlSubagentsturn38view6, and urlCreating multi-agent workflowsturn38view0 | urlOpenAIhttps://openai.com | Live docs | Best Codex sources for progressive disclosure, explicit subagents, MCP, and deterministic reviewable orchestration | Product docs, not controlled experiments |
| urlIntroducing Codexturn36search1 | urlOpenAIhttps://openai.com | 16 May 2025 | High-level product framing for cloud-based software-engineering agent workflows | Launch post; marketing bias expected |
| urlSWE-agent: Agent-Computer Interfaces Enable Automated Software Engineeringturn37search2 | John Yang et al. | 2024 | Strong research evidence that agent-computer interface design materially affects software-agent performance | Benchmark-centric; not a platform manual |
| urlAgentless: Demystifying LLM-based Software Engineering Agentsturn37search0 | Chunqiu Steven Xia et al. | 2024 | Strong research evidence that a fixed localization-repair-validation pipeline can match or beat more open-ended agent scaffolds | Focused on issue resolution benchmarks; not every enterprise workflow maps directly |

**Bottom line:** design complex AI coding skills as **governed workflows with typed artifacts, explicit state, deterministic validation, and bounded reasoning contracts**. Use natural language to specify intent and analysis prompts. Use hooks, runners, CI, and branch rules to make the workflow real. citeturn11view0turn21view0turn15view10turn38view0turn37search0turn37search2
