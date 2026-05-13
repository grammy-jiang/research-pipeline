# Briefing Replay & Diagnosis (Phase G)

This document describes how to replay or diagnose a daily briefing run using
the Phase G harness primitives:

- ``BriefingWorkflowState`` — persisted state machine for one run.
- Stage verifiers (``research_pipeline.briefing.workflow_verification``) —
  per-stage structural checks.
- Telemetry — ``workspace/briefings/<date>/telemetry.jsonl``.

## State model

A briefing run is described by ``BriefingWorkflowState``:

| Field             | Meaning                                              |
|-------------------|------------------------------------------------------|
| ``run_date``      | ISO date for the run (e.g. ``2026-04-20``).          |
| ``current_stage`` | One of ``planned``, ``polled``, ``ranked``, ``generated``, ``validated``, ``archived``, ``failed``. |
| ``completed_stages`` | Tuple of stages successfully completed (insertion order). |
| ``artifact_paths``  | Mapping of stage name → relative artifact path.    |
| ``last_error``      | Last error message if the run failed; empty otherwise. |

State is persisted at ``<run_root>/workflow_state.json`` and loaded with
``load_workflow_state`` / written with ``save_workflow_state`` /
``advance_workflow_state``.

## Stage verifiers

For each stage there is a structural verifier that returns a
``StageVerification`` with ``ok`` and an issues tuple:

| Stage       | Verifier            | Checks                                    |
|-------------|---------------------|-------------------------------------------|
| planned     | ``verify_planned``  | ``workflow_state.json`` parses.           |
| polled      | ``verify_polled``   | ``polled.jsonl`` has at least one record. |
| ranked      | ``verify_ranked``   | ``ranked.jsonl`` parses.                  |
| generated   | ``verify_generated``| ``daily.md`` non-empty.                   |
| validated   | ``verify_validated``| ``validation.json`` parses.               |
| archived    | ``verify_archived`` | ``archived.json`` marker present.         |

Use ``verify_completed_stages(state.completed_stages, run_root)`` to verify
every stage the workflow claims completed.

## Replay procedure

1. Locate the run root: ``workspace/briefings/<date>/``.
2. Load state with ``load_workflow_state(run_root, run_date)``.
3. Inspect ``current_stage`` and ``last_error``.
4. Run ``verify_completed_stages(state.completed_stages, run_root)`` to find
   any stage whose artifact has drifted or been deleted since the original run.
5. Read ``telemetry.jsonl`` for cognitive/operational/contextual signals
   (``stage`` field) corresponding to the failure point.
6. Re-run only the failing stage by invoking the matching ``brief_*`` MCP tool
   or CLI command — every stage is idempotent.

## Diagnosis checklist

- ``current_stage == "failed"`` and ``last_error`` non-empty → triage cause.
- ``completed_stages`` truncated → re-run from first missing stage.
- A verifier returns ``ok=False`` while ``current_stage`` is later than that
  stage → artifact drift; rebuild from the previous good state.
- No telemetry events for a stage marked complete → operational logging gap;
  re-run the stage with a clean workspace.

## Tools to use

- CLI: ``research-pipeline brief run --registry config.toml`` re-runs the
  full workflow; individual ``brief poll`` / ``brief rank`` / ``brief
  generate-daily`` / ``brief validate`` commands re-run a single stage.
- MCP: namespaced ``brief_*`` tools mirror the CLI behavior; resources expose
  the artifact tree (``daily.md``, ``ranked_clusters.jsonl``, ``telemetry``,
  ``validation``, ``workflow_state``).
- Governance: ``research_pipeline.briefing.tool_governance.policy_for(name)``
  documents whether a tool is networked, deterministic, and source-allowlisted.
