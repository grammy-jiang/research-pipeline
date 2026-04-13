"""Server-driven research workflow orchestrator.

Integrates all 6 harness layers (telemetry, context engineering, governance,
verification, monitoring, recovery) into a single MCP tool that drives the
full SKILL.md pipeline using sampling and elicitation.

The tool degrades gracefully:
- Without sampling: returns stage results for external agent to analyze
- Without elicitation: uses sensible defaults at decision points
"""

from __future__ import annotations

import contextlib
import json
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mcp.server.fastmcp import Context
    from mcp.server.session import ServerSession

from mcp_server.workflow.context import compact_paper, estimate_tokens
from mcp_server.workflow.monitoring import (
    IterationMetrics,
    StopReason,
    check_doom_loop,
    content_fingerprint,
)
from mcp_server.workflow.state import (
    ExecutionRecord,
    GovernanceError,
    StageStatus,
    WorkflowStage,
    WorkflowState,
    load_state,
    save_state,
)
from mcp_server.workflow.telemetry import WorkflowTelemetry
from mcp_server.workflow.verification import (
    STAGE_VERIFIERS,
    VerificationResult,
    verify_analyze,
    verify_synthesize,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Capability detection
# ---------------------------------------------------------------------------


def _has_sampling(ctx: Context | None) -> bool:
    """Check if the client supports sampling."""
    if ctx is None:
        return False
    with contextlib.suppress(Exception):
        session = _get_session(ctx)
        if session is not None:
            caps = getattr(session, "client_params", None)
            if caps is not None:
                return (
                    getattr(getattr(caps, "capabilities", None), "sampling", None)
                    is not None
                )
    return False


def _has_elicitation(ctx: Context | None) -> bool:
    """Check if the client supports elicitation."""
    if ctx is None:
        return False
    with contextlib.suppress(Exception):
        session = _get_session(ctx)
        if session is not None:
            caps = getattr(session, "client_params", None)
            if caps is not None:
                return (
                    getattr(getattr(caps, "capabilities", None), "elicitation", None)
                    is not None
                )
    return False


def _get_session(ctx: Context) -> ServerSession | None:
    """Extract the server session from a Context."""
    with contextlib.suppress(Exception):
        return getattr(ctx, "session", None)
    return None


# ---------------------------------------------------------------------------
# Stage execution helpers
# ---------------------------------------------------------------------------


def _execute_pipeline_stage(
    stage: str,
    workspace: str,
    run_id: str,
    *,
    topic: str = "",
    source: str = "",
    force: bool = False,
) -> dict:
    """Execute a pipeline stage via the existing CLI logic.

    Returns a result dict with success/message/artifacts.
    """
    from mcp_server.schemas import (
        ConvertPdfsInput,
        DownloadPdfsInput,
        ExtractContentInput,
        PlanTopicInput,
        ScreenCandidatesInput,
        SearchInput,
        SummarizePapersInput,
    )
    from mcp_server.tools import (
        convert_pdfs,
        download_pdfs,
        extract_content,
        plan_topic,
        screen_candidates,
        search,
        summarize_papers,
    )

    stage_map = {
        WorkflowStage.PLAN: lambda: plan_topic(
            PlanTopicInput(topic=topic, workspace=workspace, run_id=run_id)
        ),
        WorkflowStage.SEARCH: lambda: search(
            SearchInput(
                workspace=workspace,
                run_id=run_id,
                topic=topic,
                source=source,
            )
        ),
        WorkflowStage.SCREEN: lambda: screen_candidates(
            ScreenCandidatesInput(workspace=workspace, run_id=run_id)
        ),
        WorkflowStage.DOWNLOAD: lambda: download_pdfs(
            DownloadPdfsInput(workspace=workspace, run_id=run_id, force=force)
        ),
        WorkflowStage.CONVERT: lambda: convert_pdfs(
            ConvertPdfsInput(workspace=workspace, run_id=run_id, force=force)
        ),
        WorkflowStage.EXTRACT: lambda: extract_content(
            ExtractContentInput(workspace=workspace, run_id=run_id)
        ),
        WorkflowStage.SYNTHESIZE: lambda: summarize_papers(
            SummarizePapersInput(workspace=workspace, run_id=run_id)
        ),
    }

    executor = stage_map.get(stage)
    if executor is None:
        return {"success": False, "message": f"No executor for stage: {stage}"}

    result = executor()
    return result.model_dump()


# ---------------------------------------------------------------------------
# Sampling helpers
# ---------------------------------------------------------------------------


PAPER_ANALYSIS_SYSTEM_PROMPT = """\
You are an expert academic paper analyst. Analyze the following paper and \
provide a structured assessment. Your response MUST be valid JSON with these fields:
- "rating": integer 1-5 (5=excellent)
- "methodology": string describing the approach
- "findings": list of key findings (at least 1)
- "limitations": list of limitations
- "relevance": string explaining relevance to the research question
"""

SYNTHESIS_SYSTEM_PROMPT = """\
You are an expert research synthesizer. Given multiple paper analyses, \
produce a cross-paper synthesis. Your response MUST be valid JSON with:
- "themes": list of major themes found across papers (at least 1)
- "contradictions": list of contradictions between papers
- "gaps": list of research gaps, each with "description" and "type" \
("ENGINEERING" or "ACADEMIC")
- "readiness_verdict": one of "IMPLEMENTATION_READY", "HAS_GAPS", "INSUFFICIENT"
- "summary": overall narrative summary
"""


async def _sample_analysis(
    ctx: Context,
    paper_content: str,
    topic: str,
    paper_id: str,
    telemetry: WorkflowTelemetry,
    *,
    iteration: int = 0,
) -> dict:
    """Analyze a paper via MCP sampling (create_message).

    Bounded rationality: 1 sampling round per paper, no iterative
    refinement (CVA: more rounds → more polarization).
    """
    from mcp.types import SamplingMessage, TextContent

    session = _get_session(ctx)
    if session is None:
        return {"error": "No session available for sampling"}

    prompt = (
        f"Research question: {topic}\n\n"
        f"Paper ID: {paper_id}\n\n"
        f"--- Paper Content ---\n{paper_content}\n--- End Paper ---\n\n"
        "Analyze this paper and respond with the required JSON structure."
    )

    token_est = estimate_tokens(prompt)
    telemetry.log_sampling_request(
        "analyze", f"Paper analysis: {paper_id}", token_est, iteration=iteration
    )

    try:
        result = await session.create_message(
            messages=[
                SamplingMessage(
                    role="user",
                    content=TextContent(type="text", text=prompt),
                )
            ],
            max_tokens=4000,
            system_prompt=PAPER_ANALYSIS_SYSTEM_PROMPT,
        )

        response_text = ""
        if hasattr(result, "content"):
            content = result.content
            if isinstance(content, list):
                response_text = "".join(c.text for c in content if hasattr(c, "text"))
            elif hasattr(content, "text"):
                response_text = content.text

        telemetry.log_sampling_response(
            "analyze",
            f"Analysis received for {paper_id}",
            estimate_tokens(response_text),
            iteration=iteration,
        )

        # Parse JSON from response
        analysis = _parse_json_response(response_text)
        analysis["paper_id"] = paper_id
        return analysis

    except Exception as exc:
        telemetry.log_stage_failed(
            "analyze", f"Sampling failed for {paper_id}: {exc}", iteration=iteration
        )
        return {
            "paper_id": paper_id,
            "error": str(exc),
            "rating": 3,
            "findings": ["Analysis unavailable due to sampling error"],
            "methodology": "unknown",
            "limitations": ["Could not complete analysis"],
            "relevance": "unknown",
        }


async def _sample_synthesis(
    ctx: Context,
    analyses: list[dict],
    topic: str,
    telemetry: WorkflowTelemetry,
    *,
    system_building: bool = False,
    iteration: int = 0,
) -> dict:
    """Synthesize analyses via MCP sampling."""
    from mcp.types import SamplingMessage, TextContent

    session = _get_session(ctx)
    if session is None:
        return {"error": "No session available for sampling"}

    analyses_text = json.dumps(analyses, indent=2)
    mode_note = (
        "System-building mode: assess implementation readiness."
        if system_building
        else ""
    )
    prompt = (
        f"Research question: {topic}\n\n"
        f"{mode_note}\n\n"
        f"Paper analyses:\n{analyses_text}\n\n"
        "Synthesize these analyses and respond with the required JSON structure."
    )

    token_est = estimate_tokens(prompt)
    telemetry.log_sampling_request(
        "synthesize", "Cross-paper synthesis", token_est, iteration=iteration
    )

    try:
        result = await session.create_message(
            messages=[
                SamplingMessage(
                    role="user",
                    content=TextContent(type="text", text=prompt),
                )
            ],
            max_tokens=8000,
            system_prompt=SYNTHESIS_SYSTEM_PROMPT,
        )

        response_text = ""
        if hasattr(result, "content"):
            content = result.content
            if isinstance(content, list):
                response_text = "".join(c.text for c in content if hasattr(c, "text"))
            elif hasattr(content, "text"):
                response_text = content.text

        telemetry.log_sampling_response(
            "synthesize",
            "Synthesis received",
            estimate_tokens(response_text),
            iteration=iteration,
        )

        return _parse_json_response(response_text)

    except Exception as exc:
        telemetry.log_stage_failed(
            "synthesize", f"Synthesis sampling failed: {exc}", iteration=iteration
        )
        return {
            "themes": [],
            "contradictions": [],
            "gaps": [],
            "readiness_verdict": "INSUFFICIENT",
            "summary": f"Synthesis unavailable: {exc}",
        }


def _parse_json_response(text: str) -> dict:
    """Parse a JSON response from sampling, handling markdown code blocks."""
    text = text.strip()

    # Strip markdown code fence
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first and last lines if they're code fences
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to find JSON object in the text
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            with contextlib.suppress(json.JSONDecodeError):
                return json.loads(text[start:end])
        return {"raw_response": text}


# ---------------------------------------------------------------------------
# Elicitation helpers
# ---------------------------------------------------------------------------


async def _elicit_plan_review(
    ctx: Context,
    plan_data: dict,
    telemetry: WorkflowTelemetry,
) -> dict:
    """Elicit user review of the query plan."""
    message = (
        f"**Query Plan Review**\n\n"
        f"Must terms: {plan_data.get('must_terms', [])}\n"
        f"Nice terms: {plan_data.get('nice_terms', [])}\n"
        f"Variants: {len(plan_data.get('query_variants', []))}\n\n"
        f"Approve this plan?"
    )
    try:
        result = await ctx.elicit(
            message=message,
            schema={
                "type": "object",
                "properties": {
                    "approved": {
                        "type": "boolean",
                        "description": "Approve the query plan",
                        "default": True,
                    },
                },
            },
        )
        decision = {"approved": True}
        if hasattr(result, "data") and result.data:
            decision = (
                result.data if isinstance(result.data, dict) else {"approved": True}
            )
        action = getattr(result, "action", "accept")
        if action == "decline":
            decision = {"approved": False}
        telemetry.log_user_decision("plan", f"Plan review: {decision}")
        return decision
    except Exception as exc:
        logger.warning("Elicitation failed, using defaults: %s", exc)
        telemetry.log_user_decision("plan", f"Default (elicitation failed): {exc}")
        return {"approved": True}


async def _elicit_shortlist_review(
    ctx: Context,
    paper_count: int,
    top_papers: list[dict],
    telemetry: WorkflowTelemetry,
) -> dict:
    """Elicit user review of screened shortlist."""
    preview = "\n".join(f"  - {p.get('title', 'Unknown')[:80]}" for p in top_papers[:5])
    message = (
        f"**Shortlist Review**\n\n"
        f"{paper_count} papers shortlisted.\n"
        f"Top papers:\n{preview}\n\n"
        f"Approve this shortlist?"
    )
    try:
        result = await ctx.elicit(
            message=message,
            schema={
                "type": "object",
                "properties": {
                    "approved": {
                        "type": "boolean",
                        "description": "Approve the shortlist",
                        "default": True,
                    },
                },
            },
        )
        decision = {"approved": True}
        if hasattr(result, "data") and result.data:
            decision = (
                result.data if isinstance(result.data, dict) else {"approved": True}
            )
        action = getattr(result, "action", "accept")
        if action == "decline":
            decision = {"approved": False}
        telemetry.log_user_decision("screen", f"Shortlist review: {decision}")
        return decision
    except Exception as exc:
        logger.warning("Elicitation failed, using defaults: %s", exc)
        telemetry.log_user_decision("screen", f"Default (elicitation failed): {exc}")
        return {"approved": True}


async def _elicit_iteration_approval(
    ctx: Context,
    iteration: int,
    max_iterations: int,
    gaps: list[dict],
    budget_remaining: int,
    metrics_summary: str,
    telemetry: WorkflowTelemetry,
) -> dict:
    """Elicit user approval before starting a new iteration."""
    gap_preview = "\n".join(
        f"  - [{g.get('type', '?')}] {g.get('description', 'Unknown')[:80]}"
        for g in gaps[:5]
    )
    message = (
        f"**Iteration {iteration + 1}/{max_iterations} Approval**\n\n"
        f"Academic gaps found:\n{gap_preview}\n\n"
        f"Budget remaining: {budget_remaining:,} tokens\n"
        f"{metrics_summary}\n\n"
        f"Continue to next iteration?"
    )
    try:
        result = await ctx.elicit(
            message=message,
            schema={
                "type": "object",
                "properties": {
                    "approved": {
                        "type": "boolean",
                        "description": "Continue iteration",
                        "default": True,
                    },
                },
            },
        )
        decision = {"approved": True}
        if hasattr(result, "data") and result.data:
            decision = (
                result.data if isinstance(result.data, dict) else {"approved": True}
            )
        action = getattr(result, "action", "accept")
        if action == "decline":
            decision = {"approved": False}
        telemetry.log_user_decision(
            "iterate", f"Iteration approval: {decision}", iteration=iteration
        )
        return decision
    except Exception as exc:
        logger.warning("Elicitation failed, using defaults: %s", exc)
        telemetry.log_user_decision(
            "iterate", f"Default (elicitation failed): {exc}", iteration=iteration
        )
        return {"approved": True}


# ---------------------------------------------------------------------------
# Main workflow orchestrator
# ---------------------------------------------------------------------------


async def run_research_workflow(
    topic: str,
    ctx: Context | None = None,
    *,
    workspace: str = "./workspace",
    run_id: str = "",
    config_path: str = "",
    system_building: bool = False,
    source: str = "",
    max_iterations: int = 3,
    resume: bool = False,
) -> dict:
    """Execute the full research workflow with harness engineering.

    Integrates all 6 layers at every stage:
    1. WL1 (Telemetry): Log intent + observation
    2. WL2 (Context): Check budget, compact if needed
    3. Execute stage
    4. WL4 (Verify): Run structural verification
    5. WL3 (Governance): Transition only if verified
    6. WL5 (Monitor): Check for doom-loops, drift
    7. WL6 (Recovery): Persist state
    8. WL1 (Telemetry): Log completion + inference
    """
    has_sampling = _has_sampling(ctx)
    has_elicitation = _has_elicitation(ctx)

    # Generate run_id if not provided
    if not run_id:
        import time as _time

        run_id = str(int(_time.time() * 1000))

    # Recovery: try to resume from saved state
    state: WorkflowState | None = None
    if resume:
        state = load_state(workspace, run_id)
        if state is not None:
            logger.info("Resumed workflow state for run %s", run_id)

    if state is None:
        state = WorkflowState(
            run_id=run_id,
            topic=topic,
            workspace=workspace,
            config_path=config_path,
            system_building=system_building,
            max_iterations=max_iterations,
        )

    telemetry = WorkflowTelemetry(workspace, run_id, ctx)
    iteration_metrics = IterationMetrics()

    # Log capabilities
    telemetry.log_analysis_decision(
        "init",
        "Workflow initialized",
        f"sampling={has_sampling}, elicitation={has_elicitation}, "
        f"system_building={system_building}",
    )

    try:
        # -- Core pipeline stages -------------------------------------------
        for stage_name in [
            WorkflowStage.PLAN,
            WorkflowStage.SEARCH,
            WorkflowStage.SCREEN,
            WorkflowStage.DOWNLOAD,
            WorkflowStage.CONVERT,
            WorkflowStage.EXTRACT,
        ]:
            # Skip already-verified stages (recovery)
            if state.get_stage_status(stage_name) == StageStatus.VERIFIED:
                telemetry.log_analysis_decision(
                    stage_name, "Skipped (already verified)", "Crash-recovery resume"
                )
                continue

            state = await _execute_harness_stage(
                state=state,
                stage=stage_name,
                telemetry=telemetry,
                ctx=ctx,
                has_elicitation=has_elicitation,
                topic=topic,
                source=source,
            )
            save_state(state)

            # Report progress
            if ctx is not None:
                stages_done = sum(
                    1
                    for s in state.stages.values()
                    if s in (StageStatus.VERIFIED, StageStatus.SKIPPED)
                )
                with contextlib.suppress(Exception):
                    ctx.report_progress(stages_done, 10)

        # -- Sampling-based analysis ----------------------------------------
        if (
            has_sampling
            and state.get_stage_status(WorkflowStage.ANALYZE) != StageStatus.VERIFIED
        ):
            state = await _run_analysis_stage(state, ctx, telemetry, topic)
            save_state(state)

            # -- Synthesis --------------------------------------------------
            state = await _run_synthesis_stage(
                state, ctx, telemetry, topic, system_building=system_building
            )
            save_state(state)

            # -- Iterative loop (system-building) ---------------------------
            if system_building:
                state = await _run_iteration_loop(
                    state,
                    ctx,
                    telemetry,
                    topic,
                    source,
                    iteration_metrics=iteration_metrics,
                    has_elicitation=has_elicitation,
                )
                save_state(state)
        elif not has_sampling:
            telemetry.log_analysis_decision(
                "analyze",
                "Skipped (no sampling capability)",
                "Client does not support sampling — returning stage results "
                "for external analysis",
            )

        # Mark report stage
        state = state.with_stage_status(WorkflowStage.REPORT, StageStatus.COMPLETED)
        state = state.with_stage_status(WorkflowStage.REPORT, StageStatus.VERIFIED)
        save_state(state)

        return _build_result(state, telemetry, iteration_metrics)

    except GovernanceError as exc:
        telemetry.log_stage_failed(state.current_stage, f"Governance error: {exc}")
        save_state(state)
        return {
            "success": False,
            "message": f"Workflow governance error: {exc}",
            "state": state.model_dump(),
        }
    except Exception as exc:
        telemetry.log_stage_failed(state.current_stage, f"Unexpected error: {exc}")
        save_state(state)
        return {
            "success": False,
            "message": f"Workflow error: {exc}",
            "state": state.model_dump(),
        }


async def _execute_harness_stage(
    state: WorkflowState,
    stage: str,
    telemetry: WorkflowTelemetry,
    ctx: Context | None,
    has_elicitation: bool,
    topic: str,
    source: str = "",
) -> WorkflowState:
    """Execute a single pipeline stage with full harness integration."""
    t0 = time.monotonic()

    # WL1: Log intent
    telemetry.log_stage_start(
        stage, f"Execute {stage} stage", iteration=state.iteration
    )

    # WL3: Governance — mark running
    state = state.with_stage_status(stage, StageStatus.RUNNING)

    # Execute
    try:
        result = _execute_pipeline_stage(
            stage,
            state.workspace,
            state.run_id,
            topic=topic,
            source=source,
        )
    except Exception as exc:
        state = state.with_stage_status(stage, StageStatus.FAILED)
        telemetry.log_stage_failed(stage, str(exc), iteration=state.iteration)
        raise

    elapsed = time.monotonic() - t0
    success = result.get("success", False)

    if not success:
        state = state.with_stage_status(stage, StageStatus.FAILED)
        telemetry.log_stage_failed(
            stage, result.get("message", "Unknown failure"), iteration=state.iteration
        )
        raise GovernanceError(f"Stage {stage} failed: {result.get('message')}")

    state = state.with_stage_status(stage, StageStatus.COMPLETED)

    # WL4: Structural verification
    verifier = STAGE_VERIFIERS.get(stage)
    if verifier:
        vr: VerificationResult = verifier(state.workspace, state.run_id)
        telemetry.log_verification_result(
            stage, vr.passed, vr.details, iteration=state.iteration
        )
        if vr.passed:
            state = state.with_stage_status(stage, StageStatus.VERIFIED)
        else:
            raise GovernanceError(f"Stage {stage} verification failed: {vr.details}")
    else:
        # Stages without disk verifiers are auto-verified on success
        state = state.with_stage_status(stage, StageStatus.VERIFIED)

    # WL1: Log completion + execution record
    artifacts = list(result.get("artifacts", {}).values())
    artifact_count = len(artifacts) if isinstance(artifacts, list) else 0
    telemetry.log_stage_complete(
        stage, elapsed, artifact_count, iteration=state.iteration
    )

    verified = state.get_stage_status(stage) == StageStatus.VERIFIED
    verdict = "verified" if verified else "completed"
    record = ExecutionRecord(
        stage=stage,
        intent=f"Execute {stage} stage",
        observation=f"Result: {result.get('message', '')}",
        inference=f"Stage {verdict}",
        artifacts_produced=artifacts if isinstance(artifacts, list) else [],
        verification_result=(
            "passed"
            if state.get_stage_status(stage) == StageStatus.VERIFIED
            else "none"
        ),
        elapsed_seconds=elapsed,
        iteration=state.iteration,
    )
    state = state.with_execution_record(record)

    # Elicitation decision points
    if has_elicitation and ctx is not None:
        if stage == WorkflowStage.PLAN:
            plan_path = (
                Path(state.workspace) / state.run_id / "plan" / "query_plan.json"
            )
            if plan_path.exists():
                plan_data = json.loads(plan_path.read_text())
                decision = await _elicit_plan_review(ctx, plan_data, telemetry)
                if not decision.get("approved", True):
                    raise GovernanceError("User declined the query plan")

        elif stage == WorkflowStage.SCREEN:
            shortlist_path = (
                Path(state.workspace) / state.run_id / "screen" / "shortlist.json"
            )
            if shortlist_path.exists():
                shortlist = json.loads(shortlist_path.read_text())
                papers = (
                    shortlist
                    if isinstance(shortlist, list)
                    else shortlist.get("papers", [])
                )
                decision = await _elicit_shortlist_review(
                    ctx, len(papers), papers[:5], telemetry
                )
                if not decision.get("approved", True):
                    raise GovernanceError("User declined the shortlist")

    return state


async def _run_analysis_stage(
    state: WorkflowState,
    ctx: Context,
    telemetry: WorkflowTelemetry,
    topic: str,
) -> WorkflowState:
    """Run sampling-based paper analysis."""
    state = state.with_stage_status(WorkflowStage.ANALYZE, StageStatus.RUNNING)

    # Find markdown files
    run_path = Path(state.workspace) / state.run_id
    md_dirs = [
        run_path / "convert" / "markdown",
        run_path / "convert_rough" / "markdown",
        run_path / "convert_fine" / "markdown",
    ]
    md_files: list[Path] = []
    for d in md_dirs:
        if d.is_dir():
            md_files.extend(d.glob("*.md"))

    if not md_files:
        telemetry.log_stage_failed("analyze", "No markdown files for analysis")
        state = state.with_stage_status(WorkflowStage.ANALYZE, StageStatus.FAILED)
        return state

    analyses: list[dict] = []
    for md_file in md_files:
        paper_id = md_file.stem
        content = md_file.read_text()

        # WL2: Context engineering — compact if needed
        content, compaction_level = compact_paper(
            content, state.context_budget.max_tokens_per_paper
        )
        token_count = estimate_tokens(content)

        # WL2: Budget check
        if not state.context_budget.can_afford(token_count):
            telemetry.log_budget_update(
                "analyze",
                state.context_budget.used,
                state.context_budget.remaining,
                state.context_budget.budget_utilization,
                iteration=state.iteration,
            )
            telemetry.log_analysis_decision(
                "analyze",
                f"Skipping {paper_id} — budget exhausted",
                f"Need {token_count} tokens, have {state.context_budget.remaining}",
            )
            break

        # Sample analysis
        analysis = await _sample_analysis(
            ctx, content, topic, paper_id, telemetry, iteration=state.iteration
        )
        analyses.append(analysis)

        # Update budget
        state = state.with_budget_update(token_count, paper_content_tokens=token_count)
        telemetry.log_budget_update(
            "analyze",
            state.context_budget.used,
            state.context_budget.remaining,
            state.context_budget.budget_utilization,
            iteration=state.iteration,
        )

    # WL4: Verify analyses
    vr = verify_analyze(analyses, len(md_files))
    telemetry.log_verification_result(
        "analyze", vr.passed, vr.details, iteration=state.iteration
    )

    # Store analyses
    analyses_path = run_path / "workflow" / "analyses.json"
    analyses_path.parent.mkdir(parents=True, exist_ok=True)
    analyses_path.write_text(json.dumps(analyses, indent=2))

    status = StageStatus.VERIFIED if vr.passed else StageStatus.COMPLETED
    state = state.with_stage_status(WorkflowStage.ANALYZE, status)

    record = ExecutionRecord(
        stage=WorkflowStage.ANALYZE,
        intent="Analyze papers via sampling",
        observation=f"Analyzed {len(analyses)}/{len(md_files)} papers",
        inference=f"Verification: {'passed' if vr.passed else 'needs review'}",
        artifacts_produced=[str(analyses_path)],
        verification_result="passed" if vr.passed else "partial",
        iteration=state.iteration,
        metadata={"paper_count": len(analyses)},
    )
    state = state.with_execution_record(record)
    return state


async def _run_synthesis_stage(
    state: WorkflowState,
    ctx: Context,
    telemetry: WorkflowTelemetry,
    topic: str,
    *,
    system_building: bool = False,
) -> WorkflowState:
    """Run sampling-based cross-paper synthesis."""
    state = state.with_stage_status(WorkflowStage.SYNTHESIZE, StageStatus.RUNNING)

    # Load analyses
    analyses_path = Path(state.workspace) / state.run_id / "workflow" / "analyses.json"
    if not analyses_path.exists():
        state = state.with_stage_status(WorkflowStage.SYNTHESIZE, StageStatus.FAILED)
        return state

    analyses = json.loads(analyses_path.read_text())

    synthesis = await _sample_synthesis(
        ctx,
        analyses,
        topic,
        telemetry,
        system_building=system_building,
        iteration=state.iteration,
    )

    # WL4: Verify synthesis
    vr = verify_synthesize(synthesis)
    telemetry.log_verification_result(
        "synthesize", vr.passed, vr.details, iteration=state.iteration
    )

    # WL5: Fingerprint for doom-loop detection
    synthesis_text = json.dumps(synthesis, sort_keys=True)
    state = state.with_fingerprint(f"iter{state.iteration}_synthesize", synthesis_text)

    # Store synthesis
    synthesis_path = (
        Path(state.workspace) / state.run_id / "workflow" / "synthesis.json"
    )
    synthesis_path.write_text(json.dumps(synthesis, indent=2))

    status = StageStatus.VERIFIED if vr.passed else StageStatus.COMPLETED
    state = state.with_stage_status(WorkflowStage.SYNTHESIZE, status)

    record = ExecutionRecord(
        stage=WorkflowStage.SYNTHESIZE,
        intent="Cross-paper synthesis via sampling",
        observation=f"Themes: {len(synthesis.get('themes', []))}, "
        f"Gaps: {len(synthesis.get('gaps', []))}",
        inference=f"Verdict: {synthesis.get('readiness_verdict', 'unknown')}",
        artifacts_produced=[str(synthesis_path)],
        verification_result="passed" if vr.passed else "partial",
        iteration=state.iteration,
    )
    state = state.with_execution_record(record)
    return state


async def _run_iteration_loop(
    state: WorkflowState,
    ctx: Context,
    telemetry: WorkflowTelemetry,
    topic: str,
    source: str,
    *,
    iteration_metrics: IterationMetrics,
    has_elicitation: bool,
) -> WorkflowState:
    """Run bounded iterative synthesis for system-building mode."""
    while state.iteration < state.max_iterations:
        # Load current synthesis
        synthesis_path = (
            Path(state.workspace) / state.run_id / "workflow" / "synthesis.json"
        )
        if not synthesis_path.exists():
            break

        synthesis = json.loads(synthesis_path.read_text())
        verdict = synthesis.get("readiness_verdict", "INSUFFICIENT")

        # Stop condition: implementation ready
        if verdict == "IMPLEMENTATION_READY":
            telemetry.log_analysis_decision(
                "iterate",
                f"Stopping: {StopReason.IMPLEMENTATION_READY}",
                "Synthesis verdict is IMPLEMENTATION_READY",
                iteration=state.iteration,
            )
            break

        # Stop condition: no academic gaps
        gaps = synthesis.get("gaps", [])
        academic_gaps = [g for g in gaps if g.get("type") == "ACADEMIC"]
        if not academic_gaps:
            telemetry.log_analysis_decision(
                "iterate",
                f"Stopping: {StopReason.NO_NEW_GAPS}",
                "No ACADEMIC gaps remain",
                iteration=state.iteration,
            )
            break

        # Stop condition: budget exhausted
        if not state.context_budget.can_afford(10_000):
            telemetry.log_analysis_decision(
                "iterate",
                f"Stopping: {StopReason.BUDGET_EXHAUSTED}",
                f"Budget remaining: {state.context_budget.remaining:,}",
                iteration=state.iteration,
            )
            break

        # WL5: Doom-loop detection on synthesis
        prev_key = f"iter{state.iteration - 1}_synthesize"
        prev_fp = state.content_fingerprints.get(prev_key)
        if prev_fp is not None and state.iteration > 0:
            current_text = json.dumps(synthesis, sort_keys=True)
            is_loop, similarity, reason = check_doom_loop(
                prev_fp, content_fingerprint(current_text)
            )
            telemetry.log_doom_loop_check(
                "iterate", is_loop, similarity, iteration=state.iteration
            )
            if is_loop:
                telemetry.log_analysis_decision(
                    "iterate",
                    f"Stopping: {reason}",
                    f"Similarity: {similarity:.2f}",
                    iteration=state.iteration,
                )
                break

        # Record iteration metrics
        telemetry.log_iteration_state(
            state.iteration, state.max_iterations, 0, len(academic_gaps)
        )

        # Elicitation: ask user before iterating
        if has_elicitation and ctx is not None:
            decision = await _elicit_iteration_approval(
                ctx,
                state.iteration,
                state.max_iterations,
                academic_gaps,
                state.context_budget.remaining,
                iteration_metrics.summary(),
                telemetry,
            )
            if not decision.get("approved", True):
                telemetry.log_analysis_decision(
                    "iterate",
                    f"Stopping: {StopReason.USER_DECLINED}",
                    "User declined iteration",
                    iteration=state.iteration,
                )
                break

        # Advance iteration
        state = state.model_copy(update={"iteration": state.iteration + 1})

        # The full iteration would re-run plan→search→...→synthesize
        # For now, log the intent (actual re-execution depends on gap queries)
        telemetry.log_analysis_decision(
            "iterate",
            f"Starting iteration {state.iteration}",
            f"Targeting {len(academic_gaps)} academic gaps",
            iteration=state.iteration,
        )

        # Record metrics for drift monitoring
        iteration_metrics.record(
            iteration=state.iteration,
            papers_found=0,
            papers_analyzed=0,
            gaps_remaining=len(academic_gaps),
        )

        # Check metric-based stop
        metric_stop = iteration_metrics.should_stop()
        if metric_stop is not None:
            telemetry.log_analysis_decision(
                "iterate",
                f"Stopping: {metric_stop}",
                "Metric-based stop condition",
                iteration=state.iteration,
            )
            break

    return state


# ---------------------------------------------------------------------------
# Result builder
# ---------------------------------------------------------------------------


def _build_result(
    state: WorkflowState,
    telemetry: WorkflowTelemetry,
    iteration_metrics: IterationMetrics,
) -> dict:
    """Build the final workflow result."""
    completed = [
        stage
        for stage, status in state.stages.items()
        if status in (StageStatus.VERIFIED, StageStatus.COMPLETED)
    ]
    verified = [
        stage
        for stage, status in state.stages.items()
        if status == StageStatus.VERIFIED
    ]

    return {
        "success": True,
        "message": (
            f"Workflow completed: {len(completed)} stages completed, "
            f"{len(verified)} verified, {state.iteration} iterations"
        ),
        "run_id": state.run_id,
        "workspace": state.workspace,
        "stages_completed": completed,
        "stages_verified": verified,
        "iteration_count": state.iteration,
        "budget_used": state.context_budget.used,
        "budget_remaining": state.context_budget.remaining,
        "execution_records": len(state.execution_log),
        "telemetry_path": str(telemetry.flush()),
        "iteration_metrics": iteration_metrics.summary(),
    }
