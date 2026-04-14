"""Cross-paper synthesis: agreements, disagreements, open questions."""

from __future__ import annotations

import logging

from research_pipeline.llm.base import LLMProvider
from research_pipeline.models.summary import (
    PaperSummary,
    SynthesisAgreement,
    SynthesisDisagreement,
    SynthesisReport,
)

logger = logging.getLogger(__name__)

_SYNTHESIS_PROMPT = """\
You are an expert research synthesizer. Analyze the following paper summaries
and identify agreements, disagreements, and open questions across papers.

Research topic: {topic}

Paper summaries:
{papers_text}

Respond with a JSON object containing:
- "agreements": a list of objects, each with:
    - "claim": string describing the agreed-upon finding
    - "supporting_papers": list of arXiv IDs that support this claim
- "disagreements": a list of objects, each with:
    - "topic": string describing the contested topic
    - "positions": object mapping arXiv ID to that paper's position
- "open_questions": a list of strings describing unresolved questions
"""


def _build_synthesis_prompt(
    summaries: list[PaperSummary],
    topic: str,
) -> str:
    """Build the LLM prompt for cross-paper synthesis.

    Args:
        summaries: Individual paper summaries.
        topic: Research topic.

    Returns:
        Formatted prompt string.
    """
    parts: list[str] = []
    for s in summaries:
        findings_str = "; ".join(s.findings) if s.findings else "(none)"
        parts.append(
            f"- [{s.arxiv_id}] {s.title}\n"
            f"  Objective: {s.objective}\n"
            f"  Findings: {findings_str}"
        )
    return _SYNTHESIS_PROMPT.format(
        topic=topic,
        papers_text="\n".join(parts),
    )


def _parse_llm_synthesis_response(
    response: dict,  # type: ignore[type-arg]
    summaries: list[PaperSummary],
    topic: str,
) -> SynthesisReport:
    """Parse LLM response dict into a SynthesisReport.

    Args:
        response: Raw LLM response dict.
        summaries: Paper summaries included in the synthesis.
        topic: Research topic.

    Returns:
        SynthesisReport constructed from LLM output.

    Raises:
        KeyError: If required fields are missing.
        TypeError: If field types are wrong.
    """
    agreements = [
        SynthesisAgreement(
            claim=str(a["claim"]),
            supporting_papers=[str(p) for p in a["supporting_papers"]],
        )
        for a in response["agreements"]
    ]
    disagreements = [
        SynthesisDisagreement(
            topic=str(d["topic"]),
            positions={str(k): str(v) for k, v in d["positions"].items()},
        )
        for d in response["disagreements"]
    ]
    open_questions = [str(q) for q in response["open_questions"]]

    return SynthesisReport(
        topic=topic,
        paper_count=len(summaries),
        agreements=agreements,
        disagreements=disagreements,
        open_questions=open_questions,
        paper_summaries=summaries,
    )


def _build_template_synthesis(
    summaries: list[PaperSummary],
    topic: str,
) -> SynthesisReport:
    """Build an improved template-mode synthesis without an LLM.

    Collects all findings from papers into agreement-style entries and
    aggregates limitations, providing a richer output than a bare stub.

    Args:
        summaries: Individual paper summaries.
        topic: Research topic.

    Returns:
        SynthesisReport with aggregated findings and limitations.
    """
    agreements: list[SynthesisAgreement] = []
    for s in summaries:
        for finding in s.findings:
            agreements.append(
                SynthesisAgreement(
                    claim=finding,
                    supporting_papers=[s.arxiv_id],
                )
            )

    all_limitations: list[str] = []
    for s in summaries:
        for lim in s.limitations:
            all_limitations.append(f"[{s.arxiv_id}] {lim}")

    open_questions = [
        "Cross-paper synthesis requires LLM for detailed "
        "agreement/disagreement analysis.",
    ]
    if all_limitations:
        open_questions.append(
            f"Combined limitations across {len(summaries)} papers: "
            + "; ".join(all_limitations)
        )

    return SynthesisReport(
        topic=topic,
        paper_count=len(summaries),
        agreements=agreements,
        disagreements=[],
        open_questions=open_questions,
        paper_summaries=summaries,
    )


def synthesize(
    summaries: list[PaperSummary],
    topic: str,
    llm_provider: LLMProvider | None = None,
) -> SynthesisReport:
    """Produce a cross-paper synthesis report.

    When *llm_provider* is given, the LLM identifies agreements,
    disagreements, and open questions across papers.  Falls back to
    improved template mode on failure or when no provider is supplied.

    Args:
        summaries: List of per-paper summaries.
        topic: Research topic.
        llm_provider: Optional LLM provider for synthesis.

    Returns:
        SynthesisReport aggregating all paper summaries.
    """
    # --- LLM mode ---
    if llm_provider is not None:
        try:
            prompt = _build_synthesis_prompt(summaries, topic)
            response = llm_provider.call(prompt, schema_id="synthesis", temperature=0.0)
            report = _parse_llm_synthesis_response(response, summaries, topic)
            logger.info(
                "Generated LLM synthesis for %d papers on topic: %s",
                len(summaries),
                topic,
            )
            return report
        except Exception as exc:
            logger.warning(
                "LLM synthesis failed, falling back to template: %s",
                exc,
            )

    # --- Template mode (improved) ---
    report = _build_template_synthesis(summaries, topic)

    logger.info(
        "Synthesized %d papers on topic: %s (template mode)",
        len(summaries),
        topic,
    )
    return report
