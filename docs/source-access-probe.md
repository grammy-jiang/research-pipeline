# Public source access probe

Use this diagnostic to check whether the Raspberry Pi (or whichever machine runs
the command) can currently use a public search API. It does not start a research
run or use provider API keys.

```bash
research-pipeline probe-sources
research-pipeline probe-sources --source openalex
research-pipeline probe-sources --source arxiv --json --output access-check.json
```

From a source checkout, prefix the command with `uv run --no-sync`.
Sources are `all` (default), `arxiv`, `semantic_scholar`, `dblp`, and `openalex`.
An optional `--config PATH` loads the normal pipeline configuration; only request
pacing affects the probes. The usual configuration discovery applies when omitted.
The output file's parent must already exist.

## Request behavior

- One anonymous GET at most per selected source, requesting at most one result
  for the fixed query `email`. No automatic retries or redirect following.
- Requests are sequential, with at least 30 seconds between the completion of an
  attempt and the next attempt to another source. The existing shared per-source
  budget enforces at least 30 seconds, or the configured longer source interval.
- An active shared cooldown returns `cooldown` immediately for that source,
  without a network request. HTTP 429 updates the same persistent cooldown used
  by research, honoring `Retry-After` or the existing 15-minute fallback.
- The probe does not erase/reset cooldowns. Repeated invocations and research
  share per-source budgets. Different hosts need their existing shared coordinator.
- Connect/read timeouts are 10/15 seconds; these are transport inactivity limits,
  not a strict wall-clock deadline. Response inspection is capped at 64 KiB of
  decoded content. Sessions and response bodies are closed.
- All four eligible sources take at least 90 seconds plus network and existing
  budget waits. Select a single source for a shorter check. This is an on-demand
  diagnostic, not a background polling loop.
- Provider keys, contact details and implicit netrc credentials are not sent.
  Normal proxy/environment routing remains in effect. The report includes fixed
  endpoints, timestamps and observations, never raw bodies, redirects or errors.

## Reading the result

| Status | Observation |
| --- | --- |
| `ok` | Expected search JSON/Atom was received; valid empty results also count. |
| `cooldown` | Skipped locally because the shared retry deadline is still active. |
| `rate_limited` | HTTP 429 received; the source must remain paused. |
| `bot_challenge` | Recognized HTML robot/human verification, including HTTP 200 pages. |
| `auth_required` | HTTP 401 or an explicit authentication requirement in HTTP 403. |
| `forbidden` | Other HTTP 403; the response alone does not establish the cause. |
| `redirect` | HTTP redirect observed and deliberately not followed. |
| `http_error` | Another unsuccessful HTTP status. |
| `unexpected_response` | Malformed, oversized or unexpected search content. |
| `dns_error` | A DNS-resolution exception was identified in the exception chain. |
| `tls_error` | A TLS/certificate exception was identified. |
| `timeout` | A transport timeout, possibly while reading an HTTP response. |
| `connection_error` | Another transport failure; no more specific cause established. |

HTTP receipt and API usability are distinct. `received_http_response=true` with
`status=bot_challenge` means a server answered, but usable search was not obtained.
`request_sent=true` records an attempted network request, not proof that the
origin received it. The retained `http_status` may be 200 even when body reading
timed out. `body_truncated` means the inspection size limit was reached.

`cooldown_until` is the earliest locally allowed retry time in UTC, **not** a
prediction that the provider will unblock access then. HTTP 429 cannot distinguish
an IP limit, shared quota, account restriction or other provider policy. An `ok`
result establishes current access only to the tested public search endpoint;
it does not establish website/PDF access or validate authenticated API quotas.

Exit codes are 0 when all selected endpoints are usable, 1 for any unavailable or
skipped source, and 2 for invalid input or an internal/configuration/output error.
`--json` emits a `ProbeReport` to stdout; progress uses stderr. `--output` saves the
same JSON, including unavailable observations, for comparison with a later check.

## MCP

The `diagnostics` toolset exposes `tool_probe_sources`:

```json
{"source": "openalex", "config_path": ""}
```

It returns the same report in `artifacts.probe` and sends awaited progress
notifications during the check. `success=true` means the diagnostic completed;
inspect `artifacts.probe.all_available` and each source's `status` for usability.
The tool is annotated as networked and state-mutating because it updates shared
request budgets. It has no workspace/run argument and cannot reset a cooldown.
