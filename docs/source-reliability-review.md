# Source and workflow reliability repair review

Date: 2026-09-14

## Release preparation update

The user approved the existing-test and root-example maintenance, publishing a
new release and updating the local installation. Version 0.33.0 includes those
changes, MCP SDK <2 compatibility, and CI external-network blocking. The 12
formerly failing cases now pass. Full local Python 3.12 verification is green:
4,852 passed, 1 skipped, coverage 84.86% (job a085efc88473).

Release CI additionally exposed six vulnerable locked dependencies. Their
runtime/development floors and lock entries were updated without adding audit
exceptions. In the resulting isolated environment, 4,852 unit tests pass
(1 skipped, job d024db14dfa0); the existing vulnerability policy, license gate
and strict mypy pass. See RP-024 in the issue ledger for the original CI evidence.

PR #193 review identified additional release regressions. The repair now uses
portable inter-process locks, carries configured pacing/credentials and sparsity
thresholds into all affected call paths, and enforces Contents, Round History,
Mermaid and LaTeX as mandatory workflow publication checks. MCP report validation
is correctly marked as mutating, and the publication helper uses the runner's
Python interpreter. Nine new regressions reproduced the review findings before
the fixes and pass afterward. Watch fixtures now supply a real explicit config
and verify a 45-second override, while retaining all original behavior assertions.
Independent workflow use then reproduced a valid worker Mermaid closing fence
being corrupted by appended citations. The renderer now separates citations from
multiline content; strict publication checks reject unterminated Mermaid blocks.
Original failures are archived and recorded as RP-029. Legacy confidence and
opaque-ID quality diagnostics remain a recorded limitation (RP-030).
The final guarded local suite passed: 4,863 passed, 1 skipped, coverage 84.87%
(job 4e46c84a54ba). Fresh independent quick/deep completion evidence is recorded
in the issue ledger before publishing the tag.

The sections below preserve the original review checkpoint and its historical
test failures. Its "pending approval" and "not installed" statements describe
that earlier checkpoint; release/install completion is recorded separately in
the msgloom issue ledger.

## Review status

Implemented in /home/grammy-jiang/Projects/research-pipeline-reliability on branch
fix/source-workflow-reliability, based on b4d480e349d77a604edfc40ed32ddaa34f84eb90.
This is a review branch, not an installed release. The original checkout and its
pre-existing MCP dependency pin in pyproject.toml and uv.lock are preserved.
No new msgloom research round was started during this repair.

The original installed CLI is 0.32.0. The bundled workflow manifest changes from
2.0.0 to 2.1.0; the distribution version has not been bumped. The installed CLI,
installed MCP server package, installed Skill, and existing runtime configuration
have not been replaced.

## Repair map

| Recorded issue | Implementation in this branch | Verification or limit |
| --- | --- | --- |
| RP-001: OpenAlex dates | Normalize compact and ISO date input to provider dates. | Offline compact/year/ISO cases. |
| RP-002: existence-only gates | Enforce schemas, nonempty output, evidence, per-paper coverage, worker receipts, reviewer decisions and mandatory gates. | Negative fixtures and independent quick/deep execution. |
| RP-003: all-source failures exit zero | Share structured acquisition outcomes between CLI and MCP; distinguish valid empty, partial and failed results. | Failed CLI exits nonzero; failed MCP returns failure with coverage. |
| RP-004: query-plan schema mismatch | Use canonical topic_raw and aligned runtime schemas; CLI/MCP share query planning. | Schema and workflow tests. |
| RP-005: configured sources overridden | Remove manifest's unconditional source=all. Pass configuration through CLI/MCP. | Offline configured-source checks. |
| RP-006: overstated coverage | Record planned and attempted source queries, applied date windows, status and partial candidates. Date fallback applies only to successful sparse results from capable providers. | Query/fallback/resume tests. Feed/SDK limitations remain explicit. |
| RP-007: report paths disagree | Use one draft path and explicit synthesis input; publish the verified draft to the final path. | Real offline CLI report/validate/publish execution. |
| RP-008: final-attempt sleep | Stop after the terminal arXiv failure without sleeping. | Mocked timeout/429 tests. |
| RP-009: publication before validation | Require review and validation before atomic publication; completion requires matching validation. | Rejection and changed-content fixtures fail closed. |
| RP-010: deep output not rendered | Deep synthesis emits the renderer's CrossPaperSynthesisRecord contract; report explicitly consumes it. | Distinct deep-content marker appeared in rendered report during independent verification. |
| RP-011: instruction origin misattributed | Clarify instruction provenance in the Skill and supervisory reporting guidance. | Historical clarification retained; future narration still requires review. |
| RP-012: Retry-After shortened | Parse numeric and HTTP-date values; apply server delay after jitter; persist HTTP 429 cooldown. | Mocked deadlines and jitter extremes. |
| RP-013: permanent errors retried | Do not retry ordinary permanent 4xx or invalid JSON; preserve bounded transient retries. | Mocked 400/429/transient failures. |
| RP-014: independent client budgets | Coordinate provider requests across local processes with POSIX locks and persisted last-start/cooldown state. | Shared-client/process tests; host/streaming limits below. |
| RP-015: missing request evidence | Emit credential-safe attempt ID, host, dispatch/response timing, status, recognized content type and cooldown deadline. | Synthetic HTTP events/redaction; no raw response bodies or credentials retained. |
| ENV-001 / ENV-002 | Preserve external rate-limit and DBLP response failures as unresolved environment observations. | No successful recovery probe; blocking cause and duration remain unknown. |
| CFG-001 | Document task-specific dates, coverage and model provenance; all provider defaults in code and bundled Skill use 30 seconds. | Root example and existing configuration can still override defaults; see approval boundary. |

The 30-second interval is the user's conservative default, not a claim about a
provider's required rate or an unblocking guarantee. A 429 with no usable
Retry-After pauses the source for 15 minutes. This is a local fallback policy;
a valid server deadline is a lower bound, and no automatic recovery claim is made.

## Additional defects found and repaired

- Resume previously replaced or misrepresented saved search coverage. Resume now
  validates saved candidates and coverage without issuing new search requests.
- Scholar failures could appear as successful empty results; SDK and SerpAPI
  failures now surface as failed/unavailable outcomes. Scholar candidates retain
  source provenance and stable identifiers.
- A later arXiv page failure could discard already acquired papers. Partial
  candidates now survive, raw query/window files have distinct paths, and a
  non-Atom response cannot be accepted as an empty scholarly feed.
- MCP screening and downstream download paths disagreed about shortlist format.
  The canonical RelevanceDecision format is now shared. LLM screening requires
  actual score, rationale and evidence; untouched heuristic output cannot pass.
- Plan failures before a run ID was captured were not reliably recorded. Exit
  receipts, explicit ID checks, resolved workspace arguments and bounded retries
  now preserve failed work and prevent acceptance of the wrong run.
- Resume checking renamed away prior final reports and could lose prior gaps.
  Unique snapshots preserve the published report; prior paper IDs and gaps enter
  workflow context. Shell interpolation of gap text has been removed.
- Validation could write a failed result yet leave the CLI successful. Failure
  now propagates through CLI/MCP and report publication checks the validated hash.
- Worker contracts referred to unavailable conversion paths or the wrong synthesis
  shape. Analyzer, synthesizer, reviewer and gap classifier contracts now agree
  with manifest artifacts. Unknown synthesis paper IDs are rejected.
- The workflow performs one research cycle. The parent evaluates remaining gaps
  and prepares a separate state for another bounded cycle. It does not claim to
  automatically launch or complete additional research rounds.

## Verification

| Check | Result |
| --- | --- |
| New regression files, with external network blocked | 37 passed in 1.09 seconds, job a88a47190029. |
| Full unit suite with the external-network guard | 4,839 passed, 12 failed, 1 skipped in 80.82 seconds, job 67f75eaa0e3d, before the final focused regression additions. |
| Ruff and pre-commit | Full checks passed, including all staged new files; pre-commit job 2d0f20da862b. |
| Strict mypy | Final success across 322 source files, job a66f600c37fa. |
| Independent workflow execution | Quick and deep flows completed using actual offline CLI plan/report/validate/publish/completion commands and synthetic acquisition/worker evidence. |
| Independent failure checks | Rejected reviews, stale hashes, failed plan receipts, wrong run IDs, changed plans and unknown corpus IDs blocked; workspace paths with spaces and preserved prior reports verified. |
| Final required stop-on-first-failure unit check | 1,135 passed, then the known stale CLI dedup mock failed; job b90327862438. External-network guard enabled. |
| Skill structure | Frontmatter, JSON Schemas, manifest, state template, TOML and worker YAML checks passed, job 4556cb7bb294. |
| Dependency lock | jsonschema becomes an explicit dependency; locked package names and versions are unchanged. Offline uv regeneration also normalizes platform markers. |

These are offline contract and workflow checks, not evidence of academic quality,
real LLM screening accuracy, cloud-provider availability, or recovery from 429.
The full suite is not green. Its 12 failures and the precise proposed maintenance
are listed in source-reliability-test-maintenance.md. Existing test files have
not been modified.

Reproduce the focused suite without external requests:

~~~sh
PYTHONPATH=/home/grammy-jiang/Projects/research-pipeline-reliability/tests:/home/grammy-jiang/Projects/research-pipeline-reliability/src \
uv run --no-sync --project /home/grammy-jiang/Projects/research-pipeline \
pytest -p test_offline_guard \
  tests/unit/test_source_reliability.py \
  tests/unit/test_request_budget.py \
  tests/unit/test_source_search.py \
  tests/unit/test_source_review_regressions.py \
  tests/unit/test_workflow_reliability.py -q
~~~

The opt-in tests/test_offline_guard.py plugin blocks external Requests sends,
DNS resolution and socket connections. It allows localhost and isolates test
request-budget files from real persisted cooldowns. Live-provider checks must
remain separate; never clear a real cooldown to make tests pass.

## Test incident: TEST-001

The first broad legacy unit run was incorrectly launched before a global network
guard was installed. Some older tests mocked DOI lookup but left title fallback
unmocked. One actual Semantic Scholar request is confirmed by persisted budget
state: dispatch at 2026-09-14T03:30:48.368081Z and a subsequent HTTP 429 cooldown
through 2026-09-14T03:45:50.751279Z. Subsequent calls using that budget were blocked.

This was an assistant test-isolation error and a pre-existing test-fixture gap.
It is not evidence that this call caused the historical service restriction.
The full unguarded suite had no complete wire trace, so this is one confirmed
request, not an asserted exhaustive count of all external attempts.

The guarded run additionally exposed an old Mistral OCR test that assumed an
absent SDK, although the backend now uses direct HTTP. The guard stopped this
request and two incompletely mocked enrichment cases before external dispatch.
No recovery probe was performed. The actual service cooldown was left intact.

## Remaining boundaries and follow-up

1. AGENTS.md requires explicit approval to edit existing tests. The 12 failures
   require updates in five existing test files; no assertions have been silently
   removed or skipped.
2. Root config.example.toml lies outside AGENTS.md HC2's agent-authored path
   allowlist. It still contains the old seven provider interval examples.
   The proposed change sets all seven to 30.0 seconds, matching the code and Skill.
3. Until these gates are resolved, do not install or claim the running pipeline
   has the repair. Installation must preserve the original checkout's existing
   MCP <2 dependency pin and the user's configuration, then refresh the bundled
   Skill and verify the effective runtime settings.
4. POSIX locking coordinates processes sharing a local state directory, not
   multiple hosts or all traffic behind a shared public IP. Injected custom
   sessions remain the caller's responsibility. Nonstreaming requests hold a
   provider lock through the response; streamed bodies are not fully serialized.
5. The scholarly SDK can hide internal requests. Its high-level operations share
   a budget, but this does not prove exact internal HTTP cadence. Arbitrary
   third-party retry loops and unrelated applications remain outside this budget.
6. Provider status, ban scope and recovery remain unknown. Do not rotate
   identities, proxies or budgets to evade a service restriction.
