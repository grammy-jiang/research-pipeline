# Phase F Spec — Source Expansion

## Preconditions

A01-A12, B01-B08, C01-C08, D01-D08, and E01-E08 are `audit_pass`; phases A/B/C/D/E are complete; F is in_progress.

## Purpose

Increase coverage without increasing noise.

## Scope

- source enablement evaluation
- side-by-side report comparison
- arXiv/Hugging Face paper events
- OpenAlex/Semantic Scholar/Crossref enrichment
- Hacker News discussion source
- targeted Reddit source
- targeted Bluesky / AT Protocol source
- X/Twitter official API policy-gated source stub
- YouTube / podcast weekly-context source
- offline e2e tests for source expansion

## Non-goals

No browser scraping, no full social firehose, no automatic source expansion, no UI, no MCP/skill hardening, no behavioral tracking.

## Source enablement rule

A source can be enabled only if:

1. source registry entry exists
2. access method is sanctioned
3. retention policy is defined
4. rate-limit/cadence policy is defined
5. offline fixtures exist
6. source-specific parser tests pass
7. side-by-side report comparison passes
8. report length/noise does not increase unacceptably
9. explicit feedback or source-quality metric improves

## Default posture

All new Phase F sources are disabled by default.

Primary artifacts and implementation sources outrank social/discussion sources.

Social-only clusters normally go to weekly synthesis unless corroborated by primary artifacts.

## X/Twitter rule

X/Twitter is official API only, disabled by default, strict budget only, and must require corroboration before daily inclusion.

## Completion

F01-F09 audit_pass; new sources remain governed; side-by-side comparison works; disabled-by-default behavior enforced; no scraping; A-E tests still pass.
