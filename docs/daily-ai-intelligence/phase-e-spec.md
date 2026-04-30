# Phase E Spec — Hot-topic Dossiers

## Preconditions

A01-A12, B01-B08, C01-C08, and D01-D08 are `audit_pass`; phases A/B/C/D are complete; E is in_progress.

## Purpose

Support deeper reading for one selected topic without bloating the daily brief.

## Scope

- `TopicDossier`
- single-topic dossier template
- manual CLI `brief dossier --cluster <ID>`
- primary-artifact requirement
- evidence timeline from cluster events and topic memory
- prior context
- `What changed`
- `Why it matters technically`
- `What to try / watch / ignore`
- factuality labels
- validation of required sections, evidence URLs, and length
- offline dossier e2e tests

## Factuality labels

- `supported_fact`
- `inference`
- `speculation_or_watch_item`

Every important claim must map to evidence URL or be explicitly labeled inference/speculation.

## Non-goals

No automatic dossier generation by default, no scheduler, no social-source expansion, no MCP expansion, no UI, no general literature review, no raw source dump summarization.

## Dossier gate

A dossier can be generated only when:

- selected cluster exists
- primary artifact exists
- evidence URLs exist
- dossier stays focused on one topic
- report length is within configured limit
- validator passes

## Completion

E01-E08 audit_pass; manual dossier command works; missing primary artifact rejected; evidence timeline works; claims labeled; validation catches missing sections/evidence/overlong dossiers; A-D tests still pass.
