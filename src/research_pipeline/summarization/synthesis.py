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


# --- Heuristic dissent detection ---

# Pairs of opposing terms that signal potential disagreement
_OPPOSITION_PAIRS: list[tuple[str, str]] = [
    ("outperform", "underperform"),
    ("effective", "ineffective"),
    ("better", "worse"),
    ("improve", "degrade"),
    ("increase", "decrease"),
    ("superior", "inferior"),
    ("significant", "insignificant"),
    ("benefit", "drawback"),
    ("advantage", "disadvantage"),
    ("success", "failure"),
    ("robust", "fragile"),
    ("scalable", "unscalable"),
    ("efficient", "inefficient"),
    ("accurate", "inaccurate"),
    ("reliable", "unreliable"),
    ("sufficient", "insufficient"),
    ("support", "contradict"),
    ("confirm", "refute"),
    ("enable", "hinder"),
    ("positive", "negative"),
]

# Additional negation patterns
_NEGATION_PREFIXES = ("not ", "no ", "does not ", "cannot ", "fails to ", "unable to ")


def _extract_topic_tokens(finding: str) -> set[str]:
    """Extract content-bearing tokens from a finding for topic matching.

    Args:
        finding: A finding string from a paper summary.

    Returns:
        Set of lowercase tokens (length > 3 to filter noise).
    """
    return {t for t in finding.lower().split() if len(t) > 3}


def _findings_share_topic(finding_a: str, finding_b: str) -> bool:
    """Check if two findings discuss a similar topic via token overlap.

    Uses Jaccard similarity with a threshold to determine if two
    findings are about the same topic.

    Args:
        finding_a: First finding text.
        finding_b: Second finding text.

    Returns:
        True if the findings share sufficient topical overlap.
    """
    tokens_a = _extract_topic_tokens(finding_a)
    tokens_b = _extract_topic_tokens(finding_b)
    if not tokens_a or not tokens_b:
        return False
    intersection = len(tokens_a & tokens_b)
    union = len(tokens_a | tokens_b)
    jaccard = intersection / union if union > 0 else 0.0
    return jaccard >= 0.15


def _has_opposition(text_a: str, text_b: str) -> str | None:
    """Check if two texts express opposing sentiments about the same topic.

    Detects opposition through:
    1. Lexical opposition pairs (e.g., "effective" vs "ineffective")
    2. Negation patterns (e.g., "improves X" vs "does not improve X")

    Args:
        text_a: First finding text.
        text_b: Second finding text.

    Returns:
        Description of the opposition found, or None if no opposition.
    """
    lower_a = text_a.lower()
    lower_b = text_b.lower()

    # Check opposition pairs
    for pos, neg in _OPPOSITION_PAIRS:
        if (pos in lower_a and neg in lower_b) or (neg in lower_a and pos in lower_b):
            return f"opposing terms: '{pos}' vs '{neg}'"

    # Check negation patterns: one has a verb, the other negates it
    for prefix in _NEGATION_PREFIXES:
        for term in _extract_topic_tokens(text_a):
            if term in lower_b and prefix + term in lower_a:
                return f"negation detected: '{prefix}{term}'"
            if term in lower_a and prefix + term in lower_b:
                return f"negation detected: '{prefix}{term}'"

    return None


def _detect_dissent(
    summaries: list[PaperSummary],
) -> list[SynthesisDisagreement]:
    """Detect potential disagreements across paper findings.

    Compares findings from different papers pairwise. When two findings
    share topical overlap but express opposing sentiments, they are
    flagged as a potential disagreement.

    Args:
        summaries: Individual paper summaries.

    Returns:
        List of detected disagreements.
    """
    if len(summaries) < 2:
        return []

    # Collect (arxiv_id, finding) pairs
    paper_findings: list[tuple[str, str]] = []
    for s in summaries:
        for f in s.findings:
            paper_findings.append((s.arxiv_id, f))

    # Track seen pairs to avoid duplicates
    seen_pairs: set[tuple[str, str]] = set()
    disagreements: list[SynthesisDisagreement] = []

    for i, (id_a, finding_a) in enumerate(paper_findings):
        for j in range(i + 1, len(paper_findings)):
            id_b, finding_b = paper_findings[j]

            # Skip findings from the same paper
            if id_a == id_b:
                continue

            # Skip if we've already compared these two papers on this topic
            pair_key = (min(id_a, id_b), max(id_a, id_b))
            if pair_key in seen_pairs:
                continue

            # Check topical overlap first
            if not _findings_share_topic(finding_a, finding_b):
                continue

            # Check for opposition
            opposition = _has_opposition(finding_a, finding_b)
            if opposition:
                seen_pairs.add(pair_key)
                # Build a topic summary from shared tokens
                shared = _extract_topic_tokens(finding_a) & _extract_topic_tokens(
                    finding_b
                )
                topic = " ".join(sorted(shared)[:5]) if shared else "methodology"

                disagreements.append(
                    SynthesisDisagreement(
                        topic=f"Potential disagreement on: {topic} ({opposition})",
                        positions={
                            id_a: finding_a,
                            id_b: finding_b,
                        },
                    )
                )

    return disagreements


def _build_template_synthesis(
    summaries: list[PaperSummary],
    topic: str,
) -> SynthesisReport:
    """Build an improved template-mode synthesis without an LLM.

    Collects all findings from papers into agreement-style entries,
    detects potential disagreements using heuristic lexical opposition,
    and aggregates limitations.

    Args:
        summaries: Individual paper summaries.
        topic: Research topic.

    Returns:
        SynthesisReport with aggregated findings, detected disagreements,
        and limitations.
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

    # Detect dissenting views across papers
    disagreements = _detect_dissent(summaries)
    if disagreements:
        logger.info(
            "Detected %d potential disagreement(s) across %d papers",
            len(disagreements),
            len(summaries),
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
        disagreements=disagreements,
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
