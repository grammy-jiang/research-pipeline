"""Cross-paper synthesis: agreements, disagreements, open questions."""

import logging

from research_pipeline.models.summary import PaperSummary, SynthesisReport

logger = logging.getLogger(__name__)


def synthesize(
    summaries: list[PaperSummary],
    topic: str,
) -> SynthesisReport:
    """Produce a cross-paper synthesis report.

    In template mode (no LLM), collects findings and flags uncertainties.

    Args:
        summaries: List of per-paper summaries.
        topic: Research topic.

    Returns:
        SynthesisReport aggregating all paper summaries.
    """
    report = SynthesisReport(
        topic=topic,
        paper_count=len(summaries),
        agreements=[],
        disagreements=[],
        open_questions=[
            "Cross-paper synthesis requires LLM. "
            "Enable LLM for detailed agreement/disagreement analysis.",
        ],
        paper_summaries=summaries,
    )

    logger.info(
        "Synthesized %d papers on topic: %s",
        len(summaries),
        topic,
    )
    return report
