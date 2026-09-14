# Approved maintenance of existing tests and root example

Date: 2026-09-14. Status: approved and applied for v0.33.0.

The user authorized the listed changes, release and local installation. All 12
affected cases pass with external requests blocked. The complete local Python
3.12 suite passes: 4,852 passed, 1 skipped, coverage 84.86% (job a085efc88473).
The CI unit-test command now enables the same external-network guard.

The original 12 full-suite failures came from outdated fixture assumptions,
intentional behavior changes, and missing network isolation. The fixes preserve their
original behavioral purpose and add assertions for the changed public contract,
rather than remove tests, skip them, or restore unsafe production behavior.

AGENTS.md, Testing conventions: "Never modify existing tests without explicit
approval." AGENTS.md HC2 also excludes root config.example.toml from the allowed
agent-authored paths. The user's subsequent instruction to commit, publish a release and update the
installation authorizes this previously proposed maintenance scope.

## Exact existing test scope

| File | Affected test(s) | Applied change |
| --- | --- | --- |
| tests/unit/test_cmd_cli_handlers_batch2.py | TestCmdSearch.test_happy_path_with_topic | Use a valid mocked provider for the happy path; separately assert unknown-source failure and recorded unavailable coverage. Replace stale helper mocks with the shared engine's contract and concrete config values. |
| tests/unit/test_cmd_cli_handlers_batch2.py | TestCmdSearch.test_zero_yield_source_warns_and_strict_exits | Model a successful empty result, assert the warning and persisted empty coverage, and retain strict-mode nonzero exit. |
| tests/unit/test_cmd_cli_handlers_batch2.py | TestCmdSearch.test_happy_path_with_existing_plan | Use a real CandidateRecord fixture and verify the loaded plan, persisted candidate and coverage; avoid obsolete CLI-local dedup mock assumptions. |
| tests/unit/test_cmd_cli_handlers_batch2.py | TestCmdSearch.test_search_import_error_handled | Retain a mocked ImportError; assert unavailable coverage and nonzero exit for all-source failure, with no transport request. |
| tests/unit/test_cmd_cli_handlers_batch3.py | TestCmdPlan.test_run_plan_basic | Patch cleanup/augmentation where the shared query planner looks them up, and retain canonical plan-content assertions. |
| tests/unit/test_cmd_cli_handlers_batch3.py | TestCmdPlan.test_run_plan_stop_words_removed | Remove unused obsolete CLI mocks or retarget them; retain the real stop-word behavior assertions. |
| tests/unit/test_cmd_cli_handlers_batch3.py | TestCmdSearch.test_run_search_with_topic | Replace the stale CLI-local dedup mock; retain real candidate-file and plan assertions with a mocked source. |
| tests/unit/test_cmd_cli_handlers_batch3.py | TestArxivClient.test_fetch_page_cache_hit | Supply a numeric limiter interval or concrete limiter. Retain the cached response assertion and assert that transport is not called. |
| tests/unit/test_config.py | TestLoadConfig.test_defaults_without_file | Assert the user-requested 30-second default, including the other provider defaults; keep custom override tests unchanged. |
| tests/unit/test_enrichment.py | TestEnrichCandidates.test_multiple_candidates_partial_enrichment | Mock both DOI and title fallback, explicitly return no title match for the second candidate, and retain partial-enrichment assertions. |
| tests/unit/test_enrichment.py | TestEnrichCandidates.test_s2_api_key_set_in_session | Patch the SourceSession factory used by enrichment and mock title fallback; retain the API-key header assertion with synthetic values. |
| tests/unit/test_conversion_online_backends.py | TestMistralOcrBackend.test_convert_import_error | Replace the obsolete absent-SDK assumption with a mocked HTTP failure, rename the test accordingly, and assert failure reporting without sending the fixture PDF externally. |

## Release-review follow-up

The release's configuration propagation repairs also require concrete sparsity
values in the batch3 config fixture, a configured interval in the enrichment
session-factory assertion, and a real config file for watch command fixtures.
The watch search test additionally verifies that both the HTTP session and
limiter receive the configured 45-second interval. No behavioral assertion was
removed. The missing-config fixture failed after 4,734 tests passed in the
guarded suite (job b4150b7fdb9d); the watch and new review regressions then passed
together (25 passed, job ec352289d971).

## Root example changes

| Key | Existing seconds | Applied seconds |
| --- | --- | --- |
| arxiv.min_interval_seconds | 5.0 | 30.0 |
| sources.scholar_min_interval | 10.0 | 30.0 |
| sources.serpapi_min_interval | 5.0 | 30.0 |
| sources.semantic_scholar_min_interval | 1.0 | 30.0 |
| sources.openalex_min_interval | 0.1 | 30.0 |
| sources.dblp_min_interval | 1.0 | 30.0 |
| sources.huggingface_min_interval | 0.5 | 30.0 |

## Completion gate

Run the affected cases with the external-network guard first, followed by the
required full unit suite with the same guard. Run formatting, lint, strict mypy
and pre-commit after the final edits. Keep the guard enabled even when all known
missing mocks have been fixed.

Do not install while this gate is red. Once it passes, preserve the existing
MCP dependency pin and effective user configuration during installation,
refresh the installed Skill, and verify the running CLI/MCP/Skill against the
reviewed source. A successful offline installation check still does not justify
claiming provider recovery or automatically restarting research.
