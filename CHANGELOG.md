# Changelog

All notable changes to research-pipeline.

## [v0.17.31] — 2026-05-14

### Fixed

- **Bug 1 (MEDIUM — research-pipeline/runners/subagent_contracts/paper_screener.yaml):
  Header comment `# Task: screen-candidates` did not match the manifest task ID `paper-screener`**
  The first two comment lines read `# Task: screen-candidates (paper_screener)`. The actual
  `task_id` field (line 6) was already `paper-screener` (correct), but the comment contradicted
  it. A sub-agent reading the contract for self-identification would see a task name
  (`screen-candidates`) that does not exist anywhere in `workflow_state.json`, causing confusion
  when interpreting delegation messages. Fixed by correcting the header comment to
  `# Task: paper-screener (paper_screener)`.

- **Bug 2 (MEDIUM — research-pipeline/runners/subagent_contracts/paper_analyzer.yaml):
  Header comment `# Task: paper-analysis` did not match the manifest task ID `paper-analyzer`**
  Same class of bug as Bug 1 above. The comment said `# Task: paper-analysis (paper_analyzer)`,
  while the `task_id` field was already `paper-analyzer` (correct). Fixed by correcting the
  header comment to `# Task: paper-analyzer (paper_analyzer)`.

- **Bug 3 (LOW — research-pipeline/runners/subagent_contracts/paper_synthesizer.yaml):
  Header comment `# Task: synthesis` did not match the manifest task ID `paper-synthesizer`**
  The comment said `# Task: synthesis (paper_synthesizer)`; the actual `task_id` field was
  already `paper-synthesizer` (correct). Fixed by updating the header comment to
  `# Task: paper-synthesizer (paper_synthesizer)`.

- **Bug 4 (MEDIUM — research-pipeline/runners/runner.py + hooks/resume-inject.sh):
  `_write_round_state()` wrote `current_round` but `round_state_template.json` and the
  fixed `iterative-synthesis.md` (v0.17.26) both use the field name `round`**
  `_write_round_state()` in `runners/runner.py` constructed the `round_state.json` dictionary
  with key `current_round`, but `round_state_template.json` (the authoritative schema) declares
  the field as `round`, and `iterative-synthesis.md` was corrected to use `round` in v0.17.26.
  In a multi-round session the LLM (following the fixed guide) writes `round_state.json` with the
  `round` field; the runner had already written `current_round` in the same file for the prior
  round. If the LLM replaced the whole file (not just patched it), `resume-inject.sh`'s lookup of
  `current_round` would find nothing and display `"?"` for the round number, breaking the
  context-injection hook. This was a latent regression introduced when v0.17.26 fixed
  `iterative-synthesis.md` but left the runner and hook unchanged.
  Fixed by:
  (a) Changing the key in `_write_round_state()` from `"current_round"` to `"round"` so the
  runner-written `round_state.json` matches the schema and the LLM-written version;
  (b) Changing `resume-inject.sh` to read `d.get("round", "?")` instead of
  `d.get("current_round", "?")` and updating the display label from `current_round:` to
  `round:`.

- **Bug 5 (MEDIUM — research-pipeline/runners/subagent_contracts/paper_analyzer.yaml):
  Input/output paths used `{cwd}` instead of `{run_dir}`, pointing outside the pipeline run directory**
  The `paper_analyzer.yaml` contract specified input paths as `{cwd}/convert/markdown/` and
  `{cwd}/screened.jsonl`, and output as `{cwd}/analysis/`. The runner context provides
  `run_dir` = `<cwd>/runs/<run_id>`, so all pipeline stage artifacts live under `{run_dir}`,
  not `{cwd}`. Using `{cwd}` would direct the sub-agent to look one level too high (the skill
  working directory rather than the specific pipeline run). Fixed by updating all paths to
  use `{run_dir}`: `{run_dir}/convert/markdown/`, `{run_dir}/screen/screened.jsonl`,
  `{run_dir}/analysis/`, and the completion criterion reference.

- **Bug 6 (LOW — research-pipeline/references/command-reference.md):
  Profile membership table and screen stage output filename were stale**
  The `standard` profile was described as adding `paper-screener, expand, enrich,
  analyze-claims, score-claims` to `quick`; the actual `standard` set in `manifest.json` is
  `expand, convert-fine, analyze-claims, score-claims, classify-gaps`. The `deep` profile
  description likewise listed tasks that belong to `standard`. Additionally the Screen stage
  output was listed as `screen/shortlist.json` instead of the correct `screen/screened.jsonl`.
  Fixed profile membership descriptions to match `manifest.json` and corrected the output
  filename.

- **Bug 7 (MEDIUM — research-pipeline/schemas/final_report.schema.json):
  Required section keys were stale and did not match `references/output-templates.md` headings**
  The schema required old section names (`background`, `key_findings`, `evidence_table`,
  `comparative_analysis`, `assumption_map`, `risk_register`, `conclusion`) that no longer exist
  in the current report template. The validator would accept reports with those legacy headings
  while rejecting reports written against the current `output-templates.md`. Fixed by aligning
  the required section keys with `output-templates.md`: `research_question`, `papers_reviewed`,
  `research_landscape`, `confidence_graded_findings`, `practical_recommendations`, `evidence_map`
  are now required; optional sections (`methodology_comparison`, `trade_off_analysis`,
  `points_of_agreement`, `points_of_contradiction`, `reproducibility_notes`, `future_directions`,
  `readiness_assessment`, `appendix_run_metadata`) are now listed as known boolean properties.

### Changed

- **research-pipeline/config.toml**: Added `analysis_model = "claude-opus-4.6"` under
  `[summarization]` so the model used by all LLM worker sub-agents (paper-screener,
  paper-analyzer, paper-synthesizer, gap-classifier, synthesis-reviewer) is documented and
  configurable in one place. Also expanded the default `[sources].enabled` list to include
  `semantic_scholar`, `openalex`, `dblp`, and `huggingface` alongside `arxiv`.

## [v0.17.30] — 2026-05-14

### Fixed

- **Bug 1 (BREAKING — daily-ai-intelligence/manifest.json + runners/runner.py): `review-ranked`
  had `optional: true` but no `trigger_condition`, causing it to always delegate as an
  `llm_reviewer` and block daily brief generation**
  The DAI runner skips optional tasks only when they have a non-empty `trigger_condition` AND
  the condition is not met (`if trigger and not _optional_trigger_met(...)`). `review-ranked`
  had `optional: true` but lacked `trigger_condition` entirely, so the guard was never entered.
  As an `llm_reviewer` task, the runner always delegated it and returned 0 immediately after
  `rank` completed — before `generate-daily` ever ran. The `failure_policy.note` correctly
  described it as "non-blocking for daily runs" but the code made it blocking in practice.
  Fixed by:
  (a) Adding `trigger_condition` to `review-ranked` in `manifest.json`;
  (b) Adding `"review-ranked": ["reviewer_requested"]` to `_optional_trigger_met` in
  `runners/runner.py`;
  (c) Adding a `--reviewer` CLI flag that sets `reviewer_requested` in the workflow context;
  (d) Adding `"reviewer_requested": ""` to `workflow_state_template.json` context;
  (e) Updating `references/workflow-steps.md` `[review-ranked]` section to document that the
  task only runs when `--reviewer` is passed, and that it is skipped automatically otherwise;
  (f) Updating `SKILL.md` Launch section to list `review-ranked` among optional tasks and
  document the `--reviewer` flag.

- **Bug 2 (MEDIUM — research-pipeline/references/sub-agents.md): Mermaid flowchart node C
  incorrectly listed `download, convert, extract, summarize` as running before
  `paper-screener` delegation**
  Node C read `"Deterministic stages run automatically (plan, search, screen, download,
  convert, extract, summarize)"`, implying all these stages run before the `paper-screener`
  LLM delegation. But per `manifest.json`, `download` depends on `paper-screener`, so
  `download`, `convert-rough`, `convert-fine`, `extract`, and `summarize` all run *after*
  `paper-screener` is accepted. An agent following the incorrect diagram might misunderstand
  the pipeline order and attempt to run download/convert before delegating the screener.
  Fixed by:
  (a) Changing node C to list only the pre-screener stages: `"(plan, verify-plan, search,
  screen)"`;
  (b) Adding new node G2 between G ("re-run runner.py") and H ("delegate paper-analyzer") to
  show the post-screener deterministic stages: `"Deterministic stages continue (expand,
  download, convert-rough, extract, summarize)"`.

### Fixed

- **Bug 1 (BREAKING — daily-ai-intelligence/runners/subagent_contracts/rank_reviewer.yaml):
  `verdict_schema` fields did not match `reviewer_result.schema.json`**
  The `verdict_schema` section described a JSON structure that failed schema validation:
  `reviewer_id` (wrong) → `reviewer_task_id` (required by schema); `task_id` (extra, non-normative)
  removed; `verdict: "accept | reject"` (wrong field name + wrong enum) →
  `status: "accepted | rejected | accepted_with_issues"`; `findings: list[str]` (wrong field name)
  → `issues: list[str]`; `target_artifact` (required) was absent — added as
  `"{workspace}/{date}/clusters/ranked.jsonl"`. This is the same class of bug as v0.17.27 Bug A
  (which fixed `synthesis_reviewer.yaml`) but was never applied to `rank_reviewer.yaml`. An agent
  following the old contract would write a verdict file that failed schema validation, breaking the
  optional reviewer gate for the DAI skill. Also updated `completion_criteria`
  ("verdict is 'accept' or 'reject'" → "status is 'accepted' or 'rejected'").

- **Bug 2 (MEDIUM — daily-ai-intelligence/SKILL.md Rule 3): Rule 3 still referenced stale
  `verdict: reject` field and value**
  `daily-ai-intelligence/SKILL.md` Rule 3 said "If the optional rank_reviewer returns
  `verdict: reject`… Do not override a `reject` verdict." `reviewer_result.schema.json`
  uses `status` (not `verdict`) and the rejection value is `"rejected"` (not `"reject"`).
  The `synthesis_reviewer.yaml` contract and `research-pipeline/SKILL.md` Rule 4 were both
  corrected in v0.17.27–v0.17.28, but the equivalent DAI `SKILL.md` Rule 3 was not updated.
  Fixed by updating Rule 3 to reference `status: "rejected"` and `rejected` throughout.

- **Bug 3 (MEDIUM — daily-ai-intelligence/references/workflow-steps.md): `[review-ranked]`
  section still referenced stale `verdict: reject`**
  The `[review-ranked]` task description said "On `verdict: reject`, manually reset `[rank]` to
  `pending`…". Same stale field as Bug 2. Fixed to "On `status: \"rejected\"`".

## [v0.17.28] — 2026-05-14

### Fixed

- **Bug 1 (BREAKING — runners/runner.py): `_write_round_state` filtered gaps by wrong field `gap_type` instead of `classification`**
  `research-pipeline/runners/runner.py` `_write_round_state` built the `open_gaps`
  list for `round_state.json` using `g.get("gap_type") != "OUT_OF_SCOPE"`.
  `gap_classification.schema.json` defines the field as `classification` (enum:
  ACADEMIC, ENGINEERING, OUT_OF_SCOPE) — `gap_type` was the stale field name
  corrected in the `gap_classifier.yaml` contract by v0.17.25, but the runner itself
  was never updated. Because `g.get("gap_type")` always returns `None` and
  `None != "OUT_OF_SCOPE"` is always `True`, every gap including OUT_OF_SCOPE ones
  was included in `open_gaps`. This caused `round_state.json` to misreport the open
  gap count, misleading the `resume-inject.sh` context injection into reporting
  inflated gap counts to the agent on every new prompt. Fixed by changing
  `g.get("gap_type")` to `g.get("classification")`.

- **Bug 2 (MEDIUM — SKILL.md Rule 4): Rule 4 still referenced stale `verdict: reject` field and value**
  `research-pipeline/SKILL.md` Rule 4 said "If a reviewer sub-agent returns
  `verdict: reject`… Do not override a `reject` verdict." `reviewer_result.schema.json`
  uses `status` (not `verdict`) and the rejection value is `"rejected"` (not `"reject"`).
  The `synthesis_reviewer.yaml` contract was corrected to use `status: rejected` in
  v0.17.27, but `SKILL.md` was not updated at the same time. An orchestrating agent
  reading Rule 4 would look for the wrong field (`verdict`) and wrong value (`reject`),
  potentially never triggering the rejection handler. Fixed by updating Rule 4 to
  reference `status: "rejected"` and `rejected` throughout.

## [v0.17.27] — 2026-05-14

### Fixed

- **Bug A (BREAKING — synthesis_reviewer.yaml): `verdict_schema` fields did not match `reviewer_result.schema.json`**
  The `verdict_schema` section described a JSON structure that failed schema validation:
  `reviewer_id` (wrong) → `reviewer_task_id` (required by schema); `task_id` (extra, non-normative) removed;
  `verdict: "accept | reject"` (wrong field name + wrong enum) → `status: "accepted | rejected | accepted_with_issues"`;
  `findings: list[str]` (wrong field name) → `issues: list[str]`; `target_artifact` (required) was absent.
  An agent following the old contract would write a file that failed schema validation against
  `reviewer_result.schema.json`. Also updated `completion_criteria` ("verdict is 'accept' or 'reject'"
  → "status is 'accepted' or 'rejected'") and the `note` at the bottom, which incorrectly stated
  "the orchestrator reads the verdict and decides" — in reality the agent must manually update
  `workflow_state.json` and re-run runner.py.
  Additionally, `review_dimensions.*.verdict_field` names had a naming inconsistency with the
  `verdict_schema.scores` keys: suffixes `_score` and `_ok` were removed for uniform naming
  (`faithfulness_score` → `faithfulness`, `coherence_score` → `coherence`,
  `gap_completeness_score` → `gap_completeness`, `citation_integrity_ok` → `citation_integrity`),
  and `rejection_triggers` were updated accordingly to reference `scores.faithfulness` and
  `scores.citation_integrity`.

- **Bug B (MEDIUM — references/sub-agents.md): paper-analyzer `Writes` line said "returned in agent
  output; optionally written to `analysis/`"**
  The `paper_analyzer.yaml` contract requires writing to `{run_dir}/analysis/` (mandatory
  `completion_criteria` and `status_update` step 1: run `tool_analyze_papers --collect` to validate
  files on disk). The sub-agents.md entry incorrectly described these writes as optional, potentially
  misleading an orchestrating agent into thinking the paper-analyzer sub-agent need not write files to
  disk. Fixed to clearly state the output path and that it is required. (Same class as v0.17.21 fix for
  paper-synthesizer `Writes` line.)

- **Bug C (MEDIUM — runners/subagent_contracts/paper_synthesizer.yaml): `evidence_requirements` used
  field name `gap_type` instead of `classification`**
  `synthesis_report.schema.json` defines gaps with a `classification` field (enum: ACADEMIC,
  ENGINEERING, OUT_OF_SCOPE). The contract said "Open gaps must each have a `gap_type`" — an agent
  following the contract would use the wrong field name, leaving `classification` absent. Fixed to
  reference `classification` with all three valid values.

- **Bug D (LOW — runners/subagent_contracts/paper_synthesizer.yaml): `gap_classification` table
  missing `OUT_OF_SCOPE` entry**
  `synthesis_report.schema.json` and `gap_classifier.yaml` (downstream) both support
  `OUT_OF_SCOPE` as a valid gap classification, but the `paper_synthesizer` contract's
  `gap_classification` table only listed ENGINEERING and ACADEMIC, preventing the synthesizer
  from emitting `OUT_OF_SCOPE` gaps. Fixed by adding `OUT_OF_SCOPE` to the table.
  (Bug C and D fixed together in one edit.)

## [v0.17.26] — 2026-05-14

### Fixed

- **Bug 1 (BREAKING — manifest.json): `paper-screener` depended on `["search"]` instead of `["screen"]`**
  The runner processes tasks in manifest order. With `depends_on: ["search"]`, both `paper-screener`
  and `screen` were ready at the same time. The runner delegated `paper-screener` first, paused, and
  when re-run `screen` executed and **overwrote** `screened.jsonl` with the BM25 result — discarding
  the LLM screener's improved shortlist entirely. Fixed by setting `depends_on: ["screen"]` so the LLM
  screener always runs after BM25 screening and only refines its output. Also set `phase: "screen"` and
  added `output.path: "runs/{run_id}/screen/screened.jsonl"` to the manifest entry, and replaced the
  invalid `failure_policy.fallback` key (not enforced by the runner) with a `note` field.

- **Bug 2 (BREAKING — manifest.json): `expand`, `quality`, `enrich`, `download` depended on `["screen"]`
  instead of `["paper-screener"]`**
  Even with Bug 1 fixed, if `paper-screener` was `delegated` (paused for LLM), downstream tasks still
  saw `screen` as accepted and ran immediately — processing the BM25 shortlist before the LLM screener
  could improve it. Fixed by setting all four tasks to `depends_on: ["paper-screener"]`. Since
  `skipped_by_policy ∈ READY_STATUSES`, this is transparent in `quick`/`standard` profiles where
  `paper-screener` is not included (it is immediately skipped, and downstream tasks proceed normally).

- **Bug 3 (MEDIUM — paper_screener.yaml): contract primary input was `search/candidates.jsonl`**
  The sub-agent contract listed `search/candidates.jsonl` as primary input, but `cmd_screen.py` writes
  BM25 scores to `screen/cheap_scores.jsonl` (a richer file containing scores + explanation fields),
  and `references/sub-agents.md` already documented `cheap_scores.jsonl` as the correct input. Fixed
  by updating the contract to read `screen/cheap_scores.jsonl` as primary, with `search/candidates.jsonl`
  as a fallback if BM25 scores are unavailable.

- **Bug 4 (MEDIUM — sub-agents.md): paper-screener `Writes` line said "returned in agent output"**
  The documentation stated that the LLM screener's output was "returned in agent output", contradicting
  the `paper_screener.yaml` contract which requires writing to `{run_dir}/screen/screened.jsonl` and
  uses `artifact_exists` as a completion criterion. Fixed by correcting both the `Reads` line
  (to `screen/cheap_scores.jsonl`) and the `Writes` line (to `{run_dir}/screen/screened.jsonl`).

- **Bug 5 (MEDIUM — iterative-synthesis.md): wrong field name, missing state-reset instruction,
  and undefined template variable in runner invocation (step 3)**
  Three errors in the per-round invocation instructions:
  1. "set `current_round = <N+1>`" — `workflow_state.json` uses `"round"`, not `"current_round"`.
     (`current_round` belongs to the separate `round_state.json` written by hooks.) This would cause
     the runner to start round 2 with `round: 1` still in state, corrupting round tracking.
  2. No instruction to reset task statuses to `pending`. The runner reads `workflow_state.json` and
     sees all tasks as `accepted` from round 1, making no progress on the new round. The correct
     preparation is to copy `workflow_state_template.json` (which has all tasks as `pending`) and
     then update `run_id`, `round`, `context.prior_paper_ids`, and `context.prior_gaps`.
  3. `--config CFG` — `CFG` is an undefined bare word. All other template variables in the docs use
     `{variable}` syntax. Fixed to `--config {config}`.
  Replaced the single "Before invoking…" sentence with a numbered 5-step preparation checklist that
  covers all necessary `workflow_state.json` fields before the runner is invoked.

## [v0.17.25] — 2026-05-14

### Fixed

- **Bug (sub-agents.md): Mermaid flowchart node still used `--topic` flag after v0.17.24 partial fix**
  `research-pipeline/references/sub-agents.md` Mermaid flowchart node A showed
  `python3 runner.py --topic TOPIC --profile deep`. `runner.py` declares `topic`
  as a **positional** argument (`parser.add_argument("topic", nargs="?", …)`), so
  `--topic` is unrecognized by argparse and raises "unrecognized arguments" at
  runtime. v0.17.23 fixed `python` → `python3` in this exact node; v0.17.24 then
  fixed the identical `--topic` issue in `iterative-synthesis.md` and stated "This
  pattern was already correct everywhere else" — but the `sub-agents.md` Mermaid
  node was still wrong. Fixed by removing `--topic` and using the positional form:
  `python3 runner.py TOPIC --profile deep`.

- **Bug (gap_classifier.yaml): `output_schema` field names contradicted `gap_classification.schema.json` — BREAKING**
  `research-pipeline/runners/subagent_contracts/gap_classifier.yaml` `output_schema`
  section described the `gaps.json` structure using field names that did not match the
  normative `schemas/gap_classification.schema.json`. An agent following the YAML
  contract would produce a `gaps.json` that fails schema validation because the
  required fields `id` and `classification` were absent. The specific mismatches:
  - `gap_id` → must be `id` (schema `required`)
  - `gap_type: ENGINEERING | ACADEMIC` → must be `classification: ACADEMIC | ENGINEERING | OUT_OF_SCOPE` (schema `required`; enum extended to include `OUT_OF_SCOPE`)
  - `search_queries: list[str]` → must be `suggested_search_query: str` (different name
    and type: list vs. single string)
  - `engineering_refs: list[str]` → not in schema; replaced with `resolution_notes: str`
    which is the schema-sanctioned field for per-gap notes
  - `priority` enum values `high | medium | low` → must be `HIGH | MEDIUM | LOW`
  - Top-level required fields `run_id`, `round`, and `convergence` were entirely absent
    from the `output_schema` illustration. The runner reads `convergence.should_continue`
    to decide whether to continue iterating; an agent omitting `convergence` causes the
    runner to silently treat convergence as `false` and stop after one round even when
    open ACADEMIC gaps remain.

  Additionally fixed in the same contract:
  - `description`: mentioned only ENGINEERING/ACADEMIC; added OUT_OF_SCOPE.
  - `classification_criteria`: was missing an OUT_OF_SCOPE entry; added it.
  - `completion_criteria`: used `search_query` (inconsistent with both the old
    `search_queries` and the correct `suggested_search_query`) and `engineering_ref`
    (not in schema); both corrected to match schema field names.
  - `instructions`: added step 4 explaining how to populate the `convergence` object,
    since this is a runner-observable required field that the agent must produce.


### Fixed

- **Bug (iterative-synthesis.md): `--topic` flag does not exist in runner.py — BREAKING**
  `research-pipeline/references/iterative-synthesis.md` showed the per-round
  gap-closure `runner.py` invocation using `--topic "<gap-specific topic>"`.
  However `runner.py` declares `topic` as a **positional** argument
  (`parser.add_argument("topic", nargs="?", …)`), so passing `--topic` would cause
  argparse to raise "unrecognized arguments". Fixed by removing the `--topic` flag
  and using the bare positional form: `runner.py "<gap-specific topic>" --profile …`.
  This pattern was already correct everywhere else in the skill documentation.

- **Bug (rank_reviewer.yaml): `forbidden_actions` referenced wrong filename `ranked_events.json`**
  `daily-ai-intelligence/runners/subagent_contracts/rank_reviewer.yaml` line 32
  said `Do NOT edit ranked_events.json`. The actual output file written by
  `brief_rank_events` is `ranked.jsonl` (manifest path:
  `{workspace}/{date}/clusters/ranked.jsonl`). The same stale name was fixed in
  `workflow-steps.md` in v0.17.21 but the contract file was overlooked. Fixed to
  `Do NOT edit ranked.jsonl`.

- **Bug (synthesis_reviewer.yaml): `forbidden_actions` referenced wrong filename `synthesis_report.json`**
  `research-pipeline/runners/subagent_contracts/synthesis_reviewer.yaml` line 34
  said `Do NOT edit synthesis_report.json or synthesis.md`. `synthesis_report.json`
  lives in `{run_dir}/summarize/` and is not one of the reviewer's input files;
  the reviewer's actual input is `{run_dir}/analysis/synthesis.json` (written by
  `paper-synthesizer`). The identical class of wrong filename was fixed in
  `paper_synthesizer.yaml` in v0.17.21 but this contract was missed. Fixed to
  `Do NOT edit synthesis.json or synthesis.md`.

- **Bug (manifest.json): `review-synthesis` missing `paper-synthesizer` dependency — race condition**
  `research-pipeline/manifest.json` task `review-synthesis` declared
  `"depends_on": ["report"]`. The synthesis reviewer reads
  `{run_dir}/analysis/synthesis.json` which is written by `paper-synthesizer`, not
  by `report`. Because `report` and `paper-synthesizer` are on parallel paths in
  the task DAG (both descend from `convert-rough` independently), the runner could
  delegate `review-synthesis` while `paper-synthesizer` was still in `delegated`
  state — causing the sub-agent to try to read a file that does not yet exist.
  Fixed by changing `"depends_on"` to `["paper-synthesizer", "report"]`, so
  `review-synthesis` only becomes ready after `paper-synthesizer` is accepted.
  In non-deep profiles (quick, standard) where neither `paper-synthesizer` nor
  `review-synthesis` is in scope, both receive `skipped_by_policy` immediately, so
  the extra dependency is safe.

## [v0.17.23] — 2026-05-14

### Fixed

- **Bug (DAI runner): `run_deterministic` lacked timeout — could hang forever**
  `daily-ai-intelligence/runners/runner.py` called `subprocess.run(…)` with no
  `timeout` argument and no `subprocess.TimeoutExpired` handler. If
  `validate-registry.sh` or `check_completion.sh` ever hung, the DAI runner would
  block indefinitely. The RP runner already had `timeout=300` with a proper
  `TimeoutExpired` catch. Fixed by mirroring the RP runner pattern: pass
  `timeout=300` to `subprocess.run` and wrap in `try/except subprocess.TimeoutExpired`
  returning `False, "command timed out after 300s: ..."`.

- **Bug (DAI manifest): `export-obsidian` had `on_failure: block` but is `optional: true`**
  `daily-ai-intelligence/manifest.json` task `export-obsidian` declared
  `"optional": true` but `"failure_policy": {"on_failure": "block", ...}`. While
  the DAI runner never actually evaluates the failure policy for MCP tool tasks
  (they are always auto-accepted), the policy was misleading and violated the
  invariant that optional tasks must use `on_failure: skip`. The identical pattern
  was fixed for the RP skill's `paper-synthesizer` in v0.17.21 but the DAI
  `export-obsidian` was missed. Fixed by changing `"on_failure"` from `"block"` to
  `"skip"` and renaming the explanatory key from `"message"` to `"note"` to match
  all other optional tasks.

- **Bug (iterative-synthesis.md + sub-agents.md): remaining bare `python` invocations**
  `v0.17.22` fixed SKILL.md and `workflow-steps.md` for both skills but missed two
  further locations where runner.py was invoked with bare `python`:
  - `research-pipeline/references/iterative-synthesis.md` — bash code block in the
    per-round procedure (step 3) used `python {skill_dir}/runners/runner.py`
  - `research-pipeline/references/sub-agents.md` — Mermaid flowchart node
    `A["You: python runner.py …"]`
  Both fixed to `python3`.

- **Chore (runner.py + check_completion.py docstrings): remaining bare `python` in Usage blocks**
  The module-level docstrings in both `research-pipeline/runners/runner.py` and
  `daily-ai-intelligence/runners/runner.py`, and `scripts/check_completion.py`,
  still showed bare `python` in their `Usage:` examples. These are read by agents
  consulting the source for CLI invocation hints. Fixed all to `python3`.

## [v0.17.22] — 2026-05-14

### Fixed

- **Bug (DAI runner): `print_llm_delegation` ignored manifest-declared contract path**
  `daily-ai-intelligence/runners/runner.py` always derived the contract filename from
  the task ID (`task_id.replace("-","_") + ".yaml"`), so for the `review-ranked` task
  it looked for `review_ranked.yaml` instead of the actual file `rank_reviewer.yaml`
  declared in `executor.contract`. Fixed by mirroring the RP runner: first check
  `task["executor"].get("contract", "")` and resolve relative to `SKILL_DIR`; fall
  back to the derived name only when no manifest path is declared.

- **Bug (DAI runner): `print_llm_delegation` never substituted context variables**
  The function took no `ctx` parameter and printed raw contract text, so the reviewer
  sub-agent saw literal `{workspace}/{date}/...` template placeholders instead of real
  paths. Fixed by adding a `ctx` parameter and iterating over its key/value pairs to
  replace placeholders in the contract text — matching the pattern already used in
  the RP runner. Updated the call site from `print_llm_delegation(task)` to
  `print_llm_delegation(task, ctx)`.

- **Bug (DAI stop-check.sh): hook blocked agent when no brief was run today**
  `daily-ai-intelligence/hooks/stop-check.sh` called `check_completion.sh` without
  `--workspace` / `--date`, so `check_completion.sh` exited 1 when
  `./workspace/briefing/<today>/` did not exist, causing the hook to exit 2 and block
  the agent in every project that had the hook installed globally — even when no brief
  had been started. The header comment said "The hook is a no-op when no brief workspace
  is found for today" but the code contradicted this. Fixed by adding a sentinel check
  after locating `$CHECK_SCRIPT`: if `./workspace/briefing/$(date -u +%F)` does not
  exist, exit 0 immediately — mirroring the RP `stop-check.sh` pattern.

- **Bug (SKILL.md + workflow-steps.md): bare `python` invocations of runner.py**
  Four locations still used bare `python` instead of `python3` for runner.py launch
  examples. `v0.17.21` fixed `manifest.json` and the `check_completion.py` debug line
  in `references/workflow-steps.md`, but missed the actual Launch blocks and the
  Profiles section. Fixed:
  - `research-pipeline/SKILL.md` Launch block
  - `daily-ai-intelligence/SKILL.md` Launch block
  - `research-pipeline/references/workflow-steps.md` Profiles section (two lines)
  - `daily-ai-intelligence/references/workflow-steps.md` core-pipeline section

## [v0.17.21] — 2026-05-14

### Fixed

- **Bug (workflow-steps.md RP): resume_context.json field names were wrong in debug block**
  `references/workflow-steps.md` resume-check section (prose description and the
  Python debug snippet) used stale key names that never matched what `resume-check.sh`
  actually writes. Updated four keys:
  - `ctx["resuming"]` → `ctx["resume"]`
  - `ctx["prior_report_path"]` → `ctx["snapshot"]`
  - `ctx["prior_arxiv_ids"]` → `ctx["prior_paper_ids"]`
  - `ctx["prior_gaps"]` → `ctx["open_gaps_raw"]`
  Also updated the prose description on line 27 ("with `prior_arxiv_ids` and
  `prior_gaps`" → "with `prior_paper_ids` and `open_gaps_raw`") to match.
  The ground-truth keys are defined in `scripts/resume-check.sh` (unchanged).

- **Bug (gap_classifier contract): completion_criteria referenced wrong primary source**
  `runners/subagent_contracts/gap_classifier.yaml` `completion_criteria` said
  "Every gap in `{run_dir}/analysis/synthesis.json`…" but that path is the
  *optional* deep-profile output from `paper-synthesizer`; the *primary* source
  is `{run_dir}/summarize/synthesis_report.json`. The `inputs` section already
  correctly named both files with the right primary/optional distinction — only
  `completion_criteria` was inconsistent. Updated the criterion to reference the
  primary source first and the alternative second, matching the `instructions`
  section logic.

- **Bug (workflow-steps.md DAI): incorrect claim that runner auto-re-queues [rank] on reviewer reject**
  `daily-ai-intelligence/references/workflow-steps.md` task `[review-ranked]`
  section said "On `verdict: reject`, the runner re-queues `[rank]`." The runner
  does NOT automatically reset any task status — that must be done manually by
  the agent. Updated to: "On `verdict: reject`, manually reset `[rank]` to
  `pending` in `workflow_state.json` and re-run the runner."

- **Bug (workflow-steps.md DAI): wrong artifact filename for ranked clusters**
  The `[poll]+[rank]+[generate-daily]` section listed
  `<WS>/<DATE>/clusters/clusters.jsonl` as the ranked-clusters artifact, but the
  DAI manifest `rank` task output path is `{workspace}/{date}/clusters/ranked.jsonl`.
  Updated to `ranked.jsonl`.

- **Bug (sub-agents.md): paper-synthesizer Outputs section described CLI tool outputs, not sub-agent outputs**
  The `## paper-synthesizer` section listed `synthesis_report.md`,
  `synthesis_report.json`, and `synthesis_traceability.json` as outputs — these
  are files written by the deterministic CLI `summarize` stage, not by the
  `paper-synthesizer` sub-agent. The sub-agent writes to
  `{run_dir}/analysis/synthesis.md` and `{run_dir}/analysis/synthesis.json`.
  Updated the `Outputs` field, the prompt template write instruction (was
  `/runs/<run_id>/summarize/` → `/runs/<run_id>/analysis/`), and the `Writes`
  line to reflect the correct paths.

- **Bug (paper_synthesizer contract): status_update referenced wrong output filename**
  `runners/subagent_contracts/paper_synthesizer.yaml` `status_update` said
  "Verify synthesis_report.json validates against the schema" but the actual
  output is `synthesis.json` (in `{run_dir}/analysis/`). Updated to
  "Verify synthesis.json validates against the schema".

- **Bug (DAI manifest): dossier task had wrong `type` field**
  `daily-ai-intelligence/manifest.json` `dossier` task declared
  `"type": "llm_worker_task"` but its executor is
  `"kind": "deterministic_mcp_tool"`. The `type` field is metadata — the runner
  uses `executor.kind` — but the inconsistency was misleading. Changed `type`
  to `"pipeline_stage"` to match all other optional MCP tasks (`feedback`,
  `export-obsidian`, etc.).



### Fixed

- **Bug (DAI runner): `task_ready` blocked on `skipped_by_policy` dependencies**
  `daily-ai-intelligence/runners/runner.py`: `task_ready()` only accepted `"accepted"`
  as a satisfying dependency status, causing `preferences` (which depends on optional
  `feedback`) to stay permanently `pending` when `feedback` was skipped. Added
  `READY_STATUSES = {"accepted", "skipped_by_policy"}` constant — mirroring the
  research-pipeline runner — and updated `task_ready()` to use `not in READY_STATUSES`.

- **Bug (research-pipeline manifest): `check-completion` used bare `python` not `python3`**
  `manifest.json` check-completion executor command used `python` which fails on
  Python-3-only systems. Changed to `python3` for consistency with all other skill
  scripts (`resume-check.sh`, `stop-check.sh`, `resume-inject.sh`, `validate-registry.sh`).
  Also fixed the debug invocation example in `references/workflow-steps.md`.

- **Bug (gap_classifier contract): `status_update` referenced wrong output filename**
  `runners/subagent_contracts/gap_classifier.yaml` `status_update` said
  `gap_classifications.json` but the actual output (per manifest and runner code) is
  `gaps.json`. This was a leftover from the partial fix in `0a3aaeb`. Updated to `gaps.json`.

- **Bug (workflow-steps.md): gap-closure section referenced wrong filename**
  `references/workflow-steps.md` gap-closure section said `read gap_classifications.json`
  but the file is `gaps.json`. Updated to `gaps.json`.

- **Bug (research-pipeline manifest): `paper-synthesizer` had conflicting `failure_policy`**
  `paper-synthesizer` is declared `optional: true` but had `"on_failure": "block"`.
  A task that is semantically optional must not block the workflow on failure. Changed
  `"on_failure": "block"` → `"on_failure": "skip"` with explanatory note.

- **Bug (research-pipeline manifest + contract): missing `paper_analysis.schema.json`**
  `manifest.json` output section for `paper-analyzer` and the `paper_analyzer.yaml`
  sub-agent contract both reference `schemas/paper_analysis.schema.json` which did not
  exist. Created the schema with all required fields from the analysis template:
  `paper_id`, `title`, `research_question`, `methodology`, `key_findings`
  (with `evidence_type` enum), `limitations`, `reproducibility`, `confidence_scores`,
  `raw_claims`.

- **Bug (sub-agents.md): stale `_analysis` filename convention**
  `references/sub-agents.md` still used the old underscore convention
  (`{arxiv_id}_analysis.json`) after `0a3aaeb` changed to dot-separator
  (`{arxiv_id}.analysis.json`). Updated both occurrences.


## [v0.17.14] — 2026-05-13

### Added
- **Final Daily AI Intelligence completeness audit** (Phases A–G):
  - All 63 tickets (Phases A–G) confirmed `audit_pass` via MCP `run_implementation_check`
    against `daily-ai-intelligence-implementation-plan.md` (260 satisfied, 0 violated)
  - `docs/daily-ai-intelligence/final-traceability-matrix.md` — 63-row feature-to-implementation map
  - `docs/daily-ai-intelligence/final-gap-register.md` — empty gap register (no gaps found)
  - `docs/daily-ai-intelligence/final-completeness-audit-report.md` — complete audit report
  - `phase-status.yaml` updated: `final_audit.status: complete`, `verdict: no_gaps_found`
- **Architecture compliance artifacts** in `.agent/artifacts/`:
  - `compliance_report.md` — overall partially_compliant (101 satisfied, 0 violated, 1 unknown)
  - `impl_check_daily_ai.json` — 260 satisfied, 0 violated from daily-AI-intelligence plan
  - `impl_check_implplan.json` — implementation-plan.md fully compliant (118 satisfied, 0 violated)


## [v0.17.13] — 2026-05-12

### Added
- **Multi-agent reliability structs** (Research Report Rec. 4):
  - `AgentDiversityConfig` — enforces ≥N model families across sub-agent pool
    with `validate_diversity()` returning warnings on insufficient diversity
  - `SubAgentBudget` — per-role token limits (max + target) with `SUB_AGENT_BUDGETS`
    defaults for paper-analyzer, synthesizer, screener, and report-generator
  - `AgentsConfig` — wrapper for diversity + budgets, exposed via `PipelineConfig.agents`
  - `PreCommitmentPolicy` enum (PARALLEL / SEQUENTIAL_BLIND / SEQUENTIAL_INFORMED)
    for controlling sub-agent dispatch isolation
- **Minority finding tracking** in synthesis:
  - `MinorityFinding` model with `finding`, `supporting_sources`, `contradicting_sources`,
    `evidence_quality`, `evaluation`, and `suppression_risk` fields
  - `SynthesisReport.minority_findings` and `SynthesisReport.consensus_confidence`
    populated by `_build_template_synthesis()` from `_detect_dissent()` output
- **Memory lifecycle hooks** on `MemoryManager`:
  - `consolidate()` — promotes episodic memories to semantic store via `EpisodeStore`
  - `between_stages(new_stage, consolidation_threshold=20)` — resets working memory
    and auto-triggers consolidation when episodic count exceeds threshold
  - Orchestrator now calls `memory.between_stages()` at all 7 stage boundaries

## [v0.17.12] — 2026-05-13

### Added
- **deep profile orchestration**: Wire `expand`, `quality`, `analyze_claims`, and
  `score_claims` stages into the `run` orchestrator pipeline for the deep profile.
  Previously these stages were no-ops (log-only TODO stubs) even when selected by
  `--profile deep`.
  - `expand`: auto-expands citation graph from shortlisted paper IDs via Semantic Scholar
  - `quality`: computes composite quality scores (citation impact, venue reputation, recency)
  - `analyze_claims`: decomposes paper summaries into atomic claims with evidence classification
  - `score_claims`: scores confidence for decomposed claims using multi-signal aggregation
- Stage verifiers for all 4 new deep-profile stages (`_verify_expand`, `_verify_quality`,
  `_verify_analyze_claims`, `_verify_score_claims`) added to `STAGE_VERIFIERS`
- Full resume support for all 4 new stages

## [v0.17.10] — 2026-05-13

### Added
- **GAP-005**: `--retry-failed` flag on `download` command (spec §6.2 dead-letter tracking)
  Reads existing `download_manifest.jsonl`, filters entries with `status='failed'`,
  re-attempts only those, and merges results back with successful entries.

## [v0.17.9] — 2026-05-13

### Fixed
- **GAP-002**: Align composite quality DEFAULT_WEIGHTS with spec §5
  (citation: 0.35, venue: 0.25, author: 0.25, recency: 0.15, reproducibility: 0.0)
- Add `reproducibility_weight` to `QualityConfig` and propagate to all callers
  (`cmd_quality.py`, `mcp_server/tools.py`)

### Added
- **GAP-003**: New MCP tool `get_venue_tier` — look up CORE venue tier and score
- **GAP-004**: New MCP tool `compute_semantic_scores` — SPECTER2 semantic similarity
  scores for screened candidates (spec §3.5)
- New `GetVenueTierInput` and `ComputeSemanticScoresInput` schemas in `mcp_server/schemas.py`

## [v0.17.8] — 2026-05-12

### Fixed

- Corrected stale environment variable name in `docs/architecture.md`:
  `ARXIV_PAPER_PIPELINE_CONFIG` → `RESEARCH_PIPELINE_CONFIG` (the
  implementation has used `RESEARCH_PIPELINE_CONFIG` since v0.3.0; the
  documentation lagged behind).

### Changed

- Updated `astral-sh/setup-uv` from `@v7` to `@v8` in all GitHub Actions
  workflows (`ci.yml`, `docs.yml`, `daily-brief.yml`, `publish.yml`).

## [v0.17.0] — 2026-04-28

### Added

- Added a parallel daily AI intelligence briefing pipeline under
  `research_pipeline.briefing` with governed source registry loading, stable
  event and cluster IDs, JSONL artifacts, telemetry, exact deduplication,
  deterministic ranking, report validation, and replayable workflow state.
- Added `research-pipeline brief ...` commands for polling sources, ranking
  events, generating and validating daily briefs, running the full workflow,
  recording feedback, reviewing topic aliases, exporting Obsidian notes,
  generating dossiers, computing preference adjustments, resuming runs,
  comparing source-expanded runs, and weekly synthesis.
- Added Phase A-G briefing surfaces: GitHub releases, RSS/Atom, manual,
  Hacker News, Hugging Face papers, and arXiv-style source adapters; topic
  memory/fatigue; explicit feedback and reversible preferences; Obsidian
  daily/topic/source notes; hot-topic dossiers; MCP `brief_*` tools and
  `briefings://` resources.
- Added a bundled `daily-ai-intelligence` skill with command, source-policy,
  report-template, feedback-loop, troubleshooting, and evaluation references.
- Added offline briefing fixtures and unit tests for the briefing workflow,
  source expansion gates, feedback conflicts, alias review, dossier
  generation, replay, MCP resources, and report quality.

### Changed

- `research-pipeline setup` now discovers and installs all bundled skills
  instead of only the academic `research-pipeline` skill.
- Daily briefing reports now suppress filler items, select the strongest
  evidence from duplicate clusters, label factual evidence, expose dynamic
  novelty/confidence/action fields, and mark reports as validated after
  deterministic validation passes.

## [v0.16.2] — 2026-04-27

### Changed

- Moved the MCP server into the packaged source tree at
  `src/research_pipeline/mcp_server/` so it is included in installed wheels.
- Added the packaged MCP CLI surface:
  `research-pipeline mcp serve` and `research-pipeline mcp config`.
- Made packaged skill and agent data the canonical setup source, removing the
  duplicate `.github/skills/` and `.github/agents/` copies that could drift.
- Extended `research-pipeline setup` to install a reusable MCP config snippet
  at `~/.config/research-pipeline/mcp.json` alongside skills and agents.
- Updated human and AI-agent documentation for the packaged MCP, skill, and
  agent layout.

## [v0.16.1] — 2026-04-22

### Changed

- **Skill refined per Anthropic's Skill-Building Guide** (bumped skill
  metadata version 1.9.0 → 1.10.0):
  - Stronger YAML `description` with explicit positive and negative
    trigger phrases (e.g., "resume a prior research report", "fill
    research gaps"; explicit redirects for general web search,
    one-off PDF conversion, and requirements analysis).
  - Added `license: MIT` and `compatibility` frontmatter fields.
  - Added **`## When To Trigger`** section listing trigger phrases
    and out-of-scope requests.
  - Added **`## Examples`** section with four concrete user-prompt →
    action pairs (fresh review, resume prior report, system-building,
    out-of-scope redirect).
  - Reformatted `## Critical Rules` into numbered, scannable items.
- README, AGENTS.md, and `.github/copilot-instructions.md` updated to
  document resume-on-top, 4-round gap-closure, and required report
  formatting (Contents, Round History, Mermaid, LaTeX).

## [v0.16.0] — 2026-04-20

### Added

- **Unified Horizon Metric (UHM)** — closes gap A3-5 from the Deep Research
  Report. Combines difficulty-weighted normalized score, horizon efficiency,
  stability (UltraHorizon token-entropy trend), and Pass[k] reliability into
  a single scalar in `[0, 1]` via geometric mean + reliability gate.
  See `src/research_pipeline/evaluation/horizon.py`.
  - New CLI: `research-pipeline horizon --score ... --achieved ... --target ...`
  - New MCP tool: `tool_horizon_metric`.
- **Recall / Reasoning / Presentation (RRP) diagnostic** — operationalizes
  the DeepResearch Bench II finding (Theme 16) that Information Recall is
  the dominant bottleneck (<50% of expert rubrics satisfied) while
  Presentation is usually near-saturated. Decomposes a synthesis report
  into three axes and identifies the bottleneck.
  See `src/research_pipeline/evaluation/recall_diagnostic.py`.
  - New CLI: `research-pipeline rrp --report <md> --shortlist <json>`
  - New MCP tool: `tool_rrp_diagnostic`.
- 30 new unit and CLI integration tests for the metrics above.

### Changed

- MCP server now registers 53 tools (was 51).

## [v0.14.4] — 2026-04-19

### Added

- Step 1 structured per-paper extraction records with typed statements,
  evidence snippets, confidence labels, uncertainty notes, and quality scores.
- Step 2 design-neutral cross-paper synthesis with taxonomy, evidence matrix,
  recurring patterns, assumption map, contradiction map, evidence-strength map,
  operational implications, risk register, and traceability appendix.
- `research-pipeline summarize --step extraction|synthesis|all` for running
  the structured stages independently.
- `structured_synthesis` report template and validation support for the new
  synthesis report shape.

### Changed

- `summarize` now writes rich artifacts under `summarize/extractions/` and
  `summarize/synthesis_report.*` while preserving legacy `*.summary.json` and
  `synthesis.json` projections.
- Bundled AI skill and human docs now describe the structured extraction and
  synthesis workflow.

## [v0.14.3] — 2026-04-18

### Added

- 64 new unit tests covering 16 previously untested CLI handlers and scholar source
- CI coverage threshold raised from 80% to 85%

### Fixed

- Remove deprecated PEP 639 `License :: OSI Approved :: MIT License` classifier
  (`license = "MIT"` SPDX identifier is sufficient)

### Changed

- Coverage: 80% → 86% (3430 tests, 15816 statements, only `__main__.py` at 0%)

## [v0.14.2] — 2026-04-18

### Added

- 98 new unit tests across 7 files covering 20+ previously untested modules
- CI coverage threshold raised from 75% to 80%
- CI coverage report XML artifact upload (Python 3.12)
- README badges: CI status, mypy, ruff
- GitHub repo description and 10 topics
- PyPI classifiers: Scientific/Engineering, MIT License

### Changed

- Development Status classifier upgraded: Alpha → Beta
- Coverage: 77% → 80% (3362 tests, 15816 statements)

## [v0.14.1] — 2026-04-18

### Fixed

- Fix all 241 mypy errors across 59 source files (now 0 errors in 211 files)
- Fix real bugs found via type checking: non-existent `config.runs_dir`,
  `resolve_workspace`, `LLMProvider.complete()`, wrong `get_stage_dir` args
- Fix variable type conflicts from shadowing in 5 modules
- Enforce mypy in CI (removed `|| true` fallback)

### Added

- PEP 561 `py.typed` marker for downstream type checking
- CLI smoke tests (22 subcommands verified via `--help`)
- Coverage threshold `--cov-fail-under=70` in CI
- Expanded ruff rules: A, C4, PT, RUF groups

## [v0.14.0] — 2026-04-18

### Changed

- Replace Black + isort with Ruff format/check (single toolchain)
- Modernize pre-commit hooks: add bandit, validate-pyproject, remove legacy hooks

### Added

- pip-audit and pip-licenses for security & license auditing
- vulture for dead code detection
- hypothesis for property-based testing (36 tests)
- Security & License Audit CI job
- Python 3.13 added to CI test matrix

## [v0.13.52] — 2026-04-18

### Added

- MCP tool wrappers for 6 new commands (cite-context, cluster, export-bibtex,
  eval-log, feedback, export-html) with integration tests (C2+C3)

## [v0.13.51] — 2026-04-18

### Added

- CHANGELOG.md auto-generated from git history (C5)

## [v0.13.50] — 2026-04-18

### Documentation

- update user-guide with 6 new CLI commands (v0.13.44-49)

## [v0.13.49] — 2026-04-18

### Added

- watch mode for topics (B8) — periodic new paper monitoring

## [v0.13.48] — 2026-04-18

### Added

- citation context extraction (B7) — in-text citation sentence extraction

## [v0.13.47] — 2026-04-18

### Added

- abstract enrichment pipeline (B6) — DOI + title-based S2 lookup

## [v0.13.46] — 2026-04-18

### Added

- paper similarity clustering (B5) — TF-IDF + agglomerative clustering

## [v0.13.45] — 2026-04-18

### Added

- report template system with 4 built-in formats (B4)

## [v0.13.44] — 2026-04-18

### Added

- BibTeX export from candidate records (B2)

## [v0.13.43] — 2026-04-18

### Fixed

- sync black/ruff versions in pre-commit, use pre-commit in CI, track skill config.toml

## [v0.13.42] — 2026-04-18

### Fixed

- set black target-version py312, add jinja2 to dev deps

## [v0.13.41] — 2026-04-18

### CI

- add GitHub Actions CI workflow (lint, test, typecheck)

## [v0.13.40] — 2026-04-18

### Added

- full multi-source parallel search + incremental runs via global index

## [v0.13.39] — 2026-04-17

### Added

- wire 5 standalone modules into pipeline flow (Tier A integration)

## [v0.13.38] — 2026-04-17

### Added

- add 7-Dimension Coherence Evaluation framework

## [v0.13.37] — 2026-04-17

### Added

- add Scientific KG Benchmark framework

## [v0.13.36] — 2026-04-17

### Added

- add RL query reformulation with Thompson sampling bandit

## [v0.13.35] — 2026-04-17

### Added

- add adaptive difficulty routing (v0.13.35)

## [v0.13.34] — 2026-04-17

### Added

- add multi-model consensus engine (v0.13.34)

## [v0.13.33] — 2026-04-17

### Added

- add query-typed retrieval stopping profiles (v0.13.33)

## [v0.13.32] — 2026-04-17

### Added

- forward citation traversal with budget-aware stopping

## [v0.13.31] — 2026-04-17

### Added

- pre-commitment protocol for conformity bias elimination

## [v0.13.30] — 2026-04-17

### Added

- retention regularization — drift detection and score penalty

## [v0.13.29] — 2026-04-17

### Added

- add full environment snapshot capture at stage boundaries (v0.13.29)

## [v0.13.28] — 2026-04-17

### Added

- add graduated rubric scoring with 4-level grades (v0.13.28)

## [v0.13.27] — 2026-04-17

### Added

- add hash-pinned tool definitions for MCP integrity (v0.13.27)

## [v0.13.26] — 2026-04-17

### Added

- add length normalization for LLM responses (v0.13.26)

## [v0.13.25] — 2026-04-17

### Added

- claim-level citation accuracy scoring (v0.13.25)

## [v0.13.24] — 2026-04-17

### Added

- non-destructive versioned memory with rollback (v0.13.24)

## [v0.13.23] — 2026-04-17

### Added

- failure taxonomy logging with JSONL persistence (v0.13.23)

## [v0.13.22] — 2026-04-17

### Added

- structured output enforcement for LLM responses (v0.13.22)

## [v0.13.21] — 2026-04-17

### Added

- segment-level memory entries with token-aware splitting

## [v0.13.20] — 2026-04-17

### Added

- Q2D query augmentation with domain synonym expansion

## [v0.13.19] — 2026-04-17

### Added

- citation budget stopping criteria for BFS expansion

## [v0.13.18] — 2026-04-17

### Added

- query noise removal with academic boilerplate filtering

## [v0.13.17] — 2026-04-17

### Added

- heuristic dissent preservation in template-mode synthesis

## [v0.13.16] — 2026-04-17

### Added

- true MMR diversity with Jaccard document similarity in screening

## [v0.13.15] — 2026-04-17

### Added

- add 4-layer confidence architecture (C4)

## [v0.13.14] — 2026-04-17

### Added

- add query-adaptive retrieval stopping criteria (C3)

## [v0.13.13] — 2026-04-17

### Added

- add KG quality evaluation framework (5-dimension, 3-layer)

## [v0.13.12] — 2026-04-17

### Added

- add Case-Based Reasoning (CBR) for strategy reuse

## [v0.13.11] — 2026-04-17

### Added

- add Pass@k + Pass[k] dual metrics evaluation (B6)

## [v0.13.10] — 2026-04-17

### Added

- add epistemic blinding audits (B5)

## [v0.13.9] — 2026-04-17

### Added

- memory consolidation engine (B4)

## [v0.13.8] — 2026-04-16

### Added

- multi-session coherence evaluation (B3)

## [v0.13.7] — 2026-04-16

### Added

- human-in-the-loop approval gates (B2)

## [v0.13.6] — 2026-04-16

### Added

- phase-aware model routing (B1)

## [v0.13.5] — 2026-04-16

### Fixed

- correct config access and synthesis filename in aggregate/export-html

## [v0.13.4] — 2026-04-16

### Added

- add HTML report export with Jinja2 templates (A5)

## [v0.13.3] — 2026-04-16

### Added

- bidirectional citation snowball with budget-aware stopping (A4)

## [v0.13.2] — 2026-04-16

### Added

- evidence-only aggregation (A3)

## [v0.13.1] — 2026-04-16

### Added

- three-channel eval logging (A2)

## [v0.13.0] — 2026-04-16

### Added

- add user feedback loop for screening weight adjustment (v0.13.0)

## [v0.12.14] — 2026-04-16

### Added

- add MCP zero-trust security (T3-6)

## [v0.12.13] — 2026-04-16

### Added

- add multi-agent architecture (T3-5)

## [v0.12.12] — 2026-04-15

### Added

- add tiered page dispatch (T3-4, v0.12.12)

## [v0.12.11] — 2026-04-15

### Added

- add self-improving retrieval (T3-3, v0.12.11)

## [v0.12.10] — 2026-04-15

### Added

- schema-grounded evaluation with per-stage validation

## [v0.12.9] — 2026-04-15

### Added

- content security gates with classification and taint tracking

## [v0.12.8] — 2026-04-15

### Added

- three-tier memory architecture (working/episodic/semantic)

## [v0.12.7] — 2026-04-15

### Added

- THINK→EXECUTE→REFLECT iterative gap-filling loop

## [v0.12.6] — 2026-04-15

### Chore

- bump pyproject.toml version to 0.12.6

## [v0.12.5] — 2026-04-15

### Added

- per-claim confidence scoring with multi-signal aggregation

## [v0.12.4] — 2026-04-15

### Chore

- bump pyproject.toml version to 0.12.4

## [v0.12.3] — 2026-04-15

### Added

- claim decomposition with evidence taxonomy

## [v0.12.2] — 2026-04-15

### Added

- MinerU (magic-pdf) conversion backend

## [v0.12.1] — 2026-04-15

### Added

- cross-encoder passage reranking for chunk retrieval

## [v0.12.0] — 2026-04-15

### Added

- Tier 1 enhancements — FACT verification, export formats, query refinement, structured evidence, enhanced comparison

## [v0.11.0] — 2026-04-15

### Added

- Batch 3 (O-T) — LLM providers, screening judge, summarization, diversity, RACE scoring

## [v0.10.0] — 2026-04-14

### Added

- Batch 2 enhancements (I-N) — safety gate, BFS expansion, sanitization, depth gate, checkpoints, hybrid retrieval

## [v0.9.0] — 2026-04-14

### Added

- v0.9.0 — audit logging, backward-preference citation, Q2D query augmentation, FTS5 index search, bibliography extraction, tool integrity hashing

## [v0.8.1] — 2026-04-14

### Chore

- bump version to 0.8.1

## [v0.8.0] — 2026-04-14

### Added

- add P1-P3 quality improvements (v0.8.0)

## [v0.7.1] — 2026-04-14

### Added

- enhanced report template with confidence levels and structured agent output schemas

## [v0.7.0] — 2026-04-14

### Chore

- bump version to 0.7.0

## [v0.6.0] — 2026-04-14

### Chore

- bump version to 0.6.0

## [v0.5.0] — 2026-04-14

### Chore

- bump version to 0.5.0

## [v0.4.0] — 2026-04-08

### Added

- v0.4.0 — merge diverged fixes + new features

## [v0.3.0] — 2026-04-05

### Documentation

- update all documentation for v0.3.0 release

## [v0.2.0] — 2026-04-05

### Added

- multi-backend PDF conversion with registry pattern

## [v0.1.0] — 2026-04-05

### Chore

- prepare v0.1.0 release for PyPI
