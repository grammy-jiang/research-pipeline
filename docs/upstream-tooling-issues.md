# Upstream Tooling Issues — Bug Reports for `llm-sca-tooling`

## 1. Document Control

| Field | Value |
|-------|-------|
| Document | Upstream tooling issue drafts |
| Status | *Filing-ready — paste each section into the upstream tracker* |
| Discovered | 2026-05-17 by the `audit` skill running against `code-intelligence` MCP server `3.2.4` (`llm-sca-tooling 0.3.4`) |
| Discovery audit | [audit-deep-research-compliance-2026-05-17.md](audit-deep-research-compliance-2026-05-17.md) §5 |
| Reproduction repo | `research-pipeline` (any large Python repo with a comprehensive spec doc) |

These are the four tooling gaps that prevented the `audit` skill's
`implementation-check` workflow from satisfying its own evidence
requirements during the 2026-05-17 audit. Each section is structured as
a self-contained issue body you can paste into the upstream tracker.

---

## 2. Issue 1 — `get_relevant_files` MCP tool is missing in v3.2.4

### Title

`MCP server 3.2.4: get_relevant_files tool is no longer exposed via tools/list`

### Summary

The `code-intelligence` MCP server at version 3.2.4 does not enumerate
`get_relevant_files` in its `tools/list` response. The `audit` skill's
`implementation-check` workflow specifies this tool as the canonical
way to look up clause-level evidence:

> ALL file lookups MUST use `get_relevant_files` MCP tool (not grep/bash/view)

Without it, any clause flagged `unknown` by `run_implementation_check`
cannot be investigated through the prescribed evidence-gathering path,
breaking the workflow's "every clause finding must cite `file:line`
from MCP tool responses" contract.

### Reproduction

```bash
printf '%s\n%s\n%s\n' \
  '{"jsonrpc":"2.0","method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"x","version":"1"}},"id":1}' \
  '{"jsonrpc":"2.0","method":"notifications/initialized"}' \
  '{"jsonrpc":"2.0","method":"tools/list","params":{},"id":2}' \
  | llm-sca-tooling mcp serve --transport stdio \
  | jq '.result.tools[].name' 2>/dev/null \
  | grep -i relevant_files

# Expected: get_relevant_files listed
# Actual: empty (tool absent)
```

### Currently exposed tools (17)

`register_repo`, `graph_build`, `graph_update`, `plugin_reload`,
`run_static_analysis`, `run_eval_suite`, `run_operational_review`,
`run_readiness_audit`, `run_patch_review`, `run_sast_repair`,
`run_issue_resolution`, `run_implementation_check`, `memory_compact`,
`task_status`, `task_result`, `task_cancel`, `task_list`.

### Expected behavior

Either:
- Restore `get_relevant_files` as an MCP tool that accepts `{"query": "<clause text>"}` and returns ranked file:line evidence from the graph, **or**
- Update the `audit` skill documentation to point to the replacement
  workflow (and provide a working equivalent in the current API).

### Severity

**Blocks** the `implementation-check` workflow's per-clause evidence
requirements when unknowns exist. Aggregate verdict can still be
produced; individual classification cannot.

### Environment

| Field | Value |
|-------|-------|
| `llm-sca-tooling` CLI version | 0.3.4 |
| MCP server name/version | `code-intelligence` / `3.2.4` |
| Discovered against repo | `research-pipeline` (git SHA `caf5edb6`) |

---

## 3. Issue 2 — Resource handlers missing for `matrix://`, `intent-graph://`, `spec://`, `trace://`

### Title

`MCP server 3.2.4: per-run resource URIs returned by run_implementation_check have no handlers`

### Summary

`run_implementation_check` returns these reference URIs in its report
body:

```json
{
  "spec_document_ref": "spec://spec:ec1188cd",
  "intent_graph_ref": "intent-graph://intent:spec:ec1188cd:1671fde7",
  "clause_verdict_matrix_ref": "matrix://impl-check:16119ff3",
  "session_trace_manifest_ref": "trace://impl-check:16119ff3"
}
```

But `resources/read` on each of these URIs returns:

```json
{
  "code": -32603,
  "message": "Internal error",
  "data": "No resource handler for 'matrix://impl-check:16119ff3'"
}
```

Similarly for `intent-graph://`, `spec://`, `trace://`. The
`code-intelligence://runs/<run_id>` URI also returns `Run not found`
for `impl-check:` run IDs, even when the run completed seconds before.

### Reproduction

Run any `run_implementation_check`, then attempt to read the URIs it
returned. All four fail with the same error pattern.

### Expected behavior

Each reference URI should be readable as a JSON resource, exposing at
minimum:

- `matrix://<run_id>` → the clause × evidence verdict matrix, including each clause's text
- `intent-graph://<doc_id>` → the parsed intent graph (clauses, dependencies, taxonomies)
- `spec://<doc_id>` → the canonical spec document the engine ingested
- `trace://<run_id>` → the per-tool-call trace
- `code-intelligence://runs/<run_id>` → the workflow run record (currently restricted to certain run types)

### Severity

**Blocks** per-clause evidence retrieval and the workflow's audit-trail
requirement. The `audit` skill describes these URIs as expected to
return JSON; the absence is undocumented.

### Environment

Same as Issue 1.

---

## 4. Issue 3 — `governance/.../manifest-state` falsely reports root markdown files missing

### Title

`MCP server 3.2.4: manifest-state resource reports AGENTS.md/CLAUDE.md/copilot-instructions.md missing when they exist`

### Summary

The resource

```
code-intelligence://governance/repo:8ff002e647ce33968a571786/manifest-state
```

returns:

```json
{
  "agents_md_present": false,
  "claude_md_present": false,
  "copilot_instructions_present": false,
  "drift_findings": [{"artefact": "AGENTS.md", "state": "missing"}],
  "harness_stage": "S2"
}
```

Direct filesystem verification of the same repository contradicts every
field:

| File | `test -f` |
|------|-----------|
| `AGENTS.md` | present (385 lines) |
| `CLAUDE.md` | present |
| `.github/copilot-instructions.md` | present |

Furthermore, `run_readiness_audit` (which runs a different scanner on
the same repo at the same time) reports `harness_stage: S3` and
`drift_findings: []`, in direct conflict with the `manifest-state`
resource.

### Reproduction

```bash
# 1. Register the repo
... register_repo with repo_path=/path/to/repo containing AGENTS.md

# 2. Read manifest-state
... resources/read code-intelligence://governance/<repo_id>/manifest-state

# Observe: agents_md_present=false despite AGENTS.md being in the working tree.
```

### Likely root cause

The `manifest-state` scanner appears to be reading from stale
graph-build evidence rather than re-checking the working tree, or it is
applying an exclusion pattern that filters root-level markdown files.
The `code-intelligence://build-evidence/<repo>` resource correctly
includes Makefile, uv.lock, pyproject.toml, so the build-evidence
scanner is not affected — only manifest-state is.

### Severity

**High** for governance use: the drift signal from this resource is
currently untrustworthy. Any consumer that gates on `drift_findings`
will see false positives and may erroneously refuse to proceed.

### Suggested fix

Either:
- Have `manifest-state` re-check the working tree on read (not rely on cached graph evidence), **or**
- Ensure `graph_build` indexes root-level markdown files (AGENTS.md, CLAUDE.md) consistently and that `manifest-state` reads from the indexed set.

### Environment

Same as Issue 1.

---

## 5. Issue 4 — `run_static_analysis` with `analyser=bandit` times out via MCP

### Title

`MCP server 3.2.4: run_static_analysis with bandit emits ANALYSER_TIMEOUT and returns empty alert list`

### Summary

Calling `run_static_analysis` with `analyser=bandit` on a moderately
sized Python repository returns:

```json
{
  "run_id": "sarif-run:...",
  "status": "completed",
  "alert_count": 0,
  "rule_count": 0,
  "new_critical_high_count": 0,
  "diagnostics": [
    {"message": "usage: bandit [-h] [-r] ..."},
    {"message": "ANALYSER_TIMEOUT: bandit JSON fallback timed out"}
  ]
}
```

The `status` field says `completed` and the alert count is 0, but the
diagnostics reveal that bandit actually timed out without producing
real results. Consumers that check only `status` and `alert_count`
will mistakenly conclude the repository is SAST-clean when in fact no
scan ran to completion.

### Reproduction

```bash
... tools/call run_static_analysis with {"repo": "<absolute path>", "analyser": "bandit"}
# Diagnostics will contain ANALYSER_TIMEOUT for any non-trivial repo.
```

### Severity

**High**. A `completed` status with no findings is the most dangerous
possible outcome for a SAST gate — it silently presents an
unverified-clean result.

### Suggested fix

- Expose a `timeout_seconds` argument on `run_static_analysis` so callers can raise it for large repos.
- Set `status` to `failed` (not `completed`) when `ANALYSER_TIMEOUT` is in the diagnostics.
- Optionally: stream bandit's intermediate JSON output rather than waiting for the full result, to keep memory low and avoid the JSON-fallback timeout path.

### Environment

Same as Issue 1; reproduced against a repo with ~1,429 source files.

---

## 6. Verification After Fixes

Once an upstream release ships fixes for any of the above, re-run the
audit:

```bash
# Re-run the implementation-check workflow against the deep-research design
# (one-shot Python MCP client keeps the server alive across the whole DAG)
python3 /tmp/audit_client.py
# Expected: matching resources/read calls succeed, manifest-state reports
# AGENTS.md present, bandit returns real findings or fails loudly.
```

Then update [`audit-deep-research-compliance-2026-05-17.md`](audit-deep-research-compliance-2026-05-17.md)
§6 to close the corresponding action item, and append a new audit
report with the post-fix verdict.
