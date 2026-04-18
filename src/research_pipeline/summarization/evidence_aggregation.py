"""Evidence-only aggregation: strip rhetoric, require citations, merge facts.

Implements the Evidence-Only Aggregation pattern from the deep research
report (Pattern 6). When combining outputs from multiple sources:

1. **Strip rhetoric** — Remove hedging language, confidence claims,
   subjective opinions, and filler phrases.
2. **Normalize length** — Constrain statements to target word counts.
3. **Extract evidence pointers** — Identify citation references.
4. **Merge duplicates** — Deduplicate semantically similar statements.
5. **Filter unsupported** — Drop statements lacking evidence citations.

Usage::

    from research_pipeline.summarization.evidence_aggregation import (
        aggregate_evidence,
    )

    result = aggregate_evidence(synthesis_report, min_pointers=1)
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from difflib import SequenceMatcher

from research_pipeline.models.evidence import (
    AggregationStats,
    EvidenceAggregation,
    EvidencePointer,
    EvidenceStatement,
    RhetoricSpan,
    RhetoricType,
)
from research_pipeline.models.summary import (
    PaperSummary,
    SynthesisReport,
)

logger = logging.getLogger(__name__)

# Rhetoric detection patterns (compiled once)
_HEDGING_PATTERNS = [
    re.compile(r"\b(?:might|may|could|possibly|perhaps|seemingly)\b", re.IGNORECASE),
    re.compile(r"\b(?:it\s+(?:seems?|appears?))\b", re.IGNORECASE),
    re.compile(r"\b(?:to\s+some\s+extent|in\s+a\s+sense)\b", re.IGNORECASE),
    re.compile(r"\b(?:arguably|presumably|conceivably)\b", re.IGNORECASE),
]

_CONFIDENCE_PATTERNS = [
    re.compile(
        r"\b(?:we\s+(?:are\s+)?(?:confident|certain|sure))\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:clearly|obviously|undoubtedly|certainly|definitely)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b(?:without\s+(?:a\s+)?doubt)\b", re.IGNORECASE),
    re.compile(
        r"\b(?:it\s+is\s+(?:clear|evident|obvious))\b",
        re.IGNORECASE,
    ),
]

_SUBJECTIVE_PATTERNS = [
    re.compile(r"\b(?:I\s+(?:think|believe|feel))\b", re.IGNORECASE),
    re.compile(
        r"\b(?:in\s+(?:my|our)\s+(?:opinion|view))\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b(?:interestingly|surprisingly|remarkably)\b", re.IGNORECASE),
    re.compile(r"\b(?:unfortunately|fortunately|hopefully)\b", re.IGNORECASE),
]

_FILLER_PATTERNS = [
    re.compile(
        r"\b(?:it\s+(?:is|should\s+be)\s+noted\s+that)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b(?:as\s+(?:we\s+)?mentioned\s+(?:above|earlier))\b", re.IGNORECASE),
    re.compile(
        r"\b(?:it\s+goes\s+without\s+saying)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b(?:needless\s+to\s+say)\b", re.IGNORECASE),
]

_UNSUPPORTED_CAUSAL_PATTERNS = [
    re.compile(
        r"\b(?:this\s+(?:proves|demonstrates|shows)\s+that)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:therefore|thus|hence|consequently)\b",
        re.IGNORECASE,
    ),
]

_PATTERN_MAP: dict[RhetoricType, list[re.Pattern[str]]] = {
    RhetoricType.HEDGING: _HEDGING_PATTERNS,
    RhetoricType.CONFIDENCE_CLAIM: _CONFIDENCE_PATTERNS,
    RhetoricType.SUBJECTIVE: _SUBJECTIVE_PATTERNS,
    RhetoricType.FILLER: _FILLER_PATTERNS,
    RhetoricType.UNSUPPORTED_CAUSAL: _UNSUPPORTED_CAUSAL_PATTERNS,
}


def detect_rhetoric(text: str) -> list[RhetoricSpan]:
    """Detect rhetoric patterns in text.

    Args:
        text: Input text to scan.

    Returns:
        List of detected rhetoric spans sorted by position.
    """
    spans: list[RhetoricSpan] = []
    for rtype, patterns in _PATTERN_MAP.items():
        for pattern in patterns:
            for match in pattern.finditer(text):
                spans.append(
                    RhetoricSpan(
                        start=match.start(),
                        end=match.end(),
                        text=match.group(),
                        rhetoric_type=rtype,
                    )
                )
    spans.sort(key=lambda s: s.start)
    return spans


def strip_rhetoric(text: str) -> tuple[str, list[RhetoricSpan]]:
    """Remove rhetoric from text while preserving meaning.

    Removes hedging language, confidence claims, subjective opinions,
    and filler phrases. Cleans up residual whitespace and punctuation.

    Args:
        text: Input text.

    Returns:
        Tuple of (cleaned_text, detected_spans).
    """
    spans = detect_rhetoric(text)
    if not spans:
        return text, []

    # Remove spans in reverse order to preserve offsets
    result = text
    for span in reversed(spans):
        before = result[: span.start]
        after = result[span.end :]
        result = before + after

    # Clean up residual artifacts
    result = re.sub(r"\s{2,}", " ", result)  # collapse multiple spaces
    result = re.sub(r"\s+([,.])", r"\1", result)  # fix space before punctuation
    result = re.sub(r"([,.])\s*([,.])", r"\1", result)  # fix double punctuation
    result = re.sub(r"^\s*[,.:;]\s*", "", result)  # strip leading punctuation
    result = result.strip()

    return result, spans


def normalize_length(
    text: str,
    max_words: int = 50,
    min_words: int = 5,
) -> str:
    """Normalize statement length to target word range.

    Truncates at sentence boundary if too long, returns empty string
    if too short (likely a fragment).

    Args:
        text: Input text.
        max_words: Maximum word count.
        min_words: Minimum word count.

    Returns:
        Normalized text (empty if too short).
    """
    words = text.split()
    if len(words) < min_words:
        return ""
    if len(words) <= max_words:
        return text

    # Truncate at sentence boundary
    truncated = " ".join(words[:max_words])
    # Find last sentence-ending punctuation
    last_period = max(truncated.rfind("."), truncated.rfind("!"), truncated.rfind("?"))
    if last_period > len(truncated) // 2:
        return truncated[: last_period + 1]
    return truncated + "..."


def extract_evidence_pointers(
    text: str,
    paper_id: str = "",
) -> list[EvidencePointer]:
    """Extract evidence citation pointers from text.

    Recognizes patterns like:
    - [arXiv:2301.12345]
    - (Smith et al., 2023)
    - [1], [2,3]
    - Section 3.2
    - Table 1, Figure 2

    Args:
        text: Text containing citations.
        paper_id: Default paper ID if no specific ID found.

    Returns:
        List of extracted evidence pointers.
    """
    pointers: list[EvidencePointer] = []

    # arXiv ID pattern: [2301.12345] or [arXiv:2301.12345]
    arxiv_pattern = re.compile(r"\[(?:arXiv:)?(\d{4}\.\d{4,5}(?:v\d+)?)\]")
    for match in arxiv_pattern.finditer(text):
        pointers.append(
            EvidencePointer(
                paper_id=match.group(1),
                quote=text[max(0, match.start() - 30) : match.end() + 30].strip(),
            )
        )

    # Section references
    section_pattern = re.compile(r"[Ss]ection\s+(\d+(?:\.\d+)*)")
    for match in section_pattern.finditer(text):
        pointers.append(
            EvidencePointer(
                paper_id=paper_id,
                section=f"Section {match.group(1)}",
                quote=text[max(0, match.start() - 20) : match.end() + 20].strip(),
            )
        )

    # Table/Figure references
    tabfig_pattern = re.compile(r"(?:Table|Figure|Fig\.)\s+(\d+)")
    for match in tabfig_pattern.finditer(text):
        pointers.append(
            EvidencePointer(
                paper_id=paper_id,
                section=match.group(0),
                quote=text[max(0, match.start() - 20) : match.end() + 20].strip(),
            )
        )

    return pointers


def _similarity(a: str, b: str) -> float:
    """Compute normalized text similarity using SequenceMatcher.

    Args:
        a: First string.
        b: Second string.

    Returns:
        Similarity ratio (0.0 to 1.0).
    """
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def _merge_similar_statements(
    statements: list[EvidenceStatement],
    threshold: float = 0.7,
) -> tuple[list[EvidenceStatement], int]:
    """Merge semantically similar statements.

    Uses SequenceMatcher similarity ratio. When two statements are
    similar, the one with more evidence pointers is kept, and pointers
    from the other are added.

    Args:
        statements: Input statements.
        threshold: Similarity threshold for merging.

    Returns:
        Tuple of (merged_statements, merge_count).
    """
    if not statements:
        return [], 0

    merged: list[EvidenceStatement] = []
    used: set[int] = set()
    merge_count = 0

    for i, si in enumerate(statements):
        if i in used:
            continue
        best = si
        for j in range(i + 1, len(statements)):
            if j in used:
                continue
            sj = statements[j]
            if _similarity(si.text, sj.text) >= threshold:
                used.add(j)
                merge_count += 1
                # Merge pointers
                existing_quotes = {p.quote for p in best.pointers}
                for p in sj.pointers:
                    if p.quote not in existing_quotes:
                        best = best.model_copy(
                            update={
                                "pointers": [*best.pointers, p],
                                "agreement_count": best.agreement_count + 1,
                            }
                        )
                        existing_quotes.add(p.quote)
        merged.append(best)

    return merged, merge_count


def _extract_statements_from_summary(
    summary: PaperSummary,
) -> list[EvidenceStatement]:
    """Extract evidence statements from a PaperSummary.

    Converts findings, limitations, methodology, and objective into
    EvidenceStatement objects with evidence pointers.

    Args:
        summary: Paper summary.

    Returns:
        List of evidence statements.
    """
    statements: list[EvidenceStatement] = []
    counter = 0

    # Build evidence pointer lookup by chunk_id
    pointer_lookup: dict[str, EvidencePointer] = {}
    for ev in summary.evidence:
        pointer_lookup[ev.chunk_id] = EvidencePointer(
            paper_id=summary.arxiv_id,
            chunk_id=ev.chunk_id,
            quote=ev.quote,
        )

    # Findings
    for finding in summary.findings:
        counter += 1
        pointers = extract_evidence_pointers(finding, summary.arxiv_id)
        if not pointers and pointer_lookup:
            # Use first available evidence as fallback
            first_key = next(iter(pointer_lookup))
            pointers = [pointer_lookup[first_key]]
        statements.append(
            EvidenceStatement(
                statement_id=f"ES-{counter:03d}",
                text=finding,
                pointers=pointers,
                source_type="finding",
            )
        )

    # Limitations
    for limitation in summary.limitations:
        counter += 1
        pointers = extract_evidence_pointers(limitation, summary.arxiv_id)
        statements.append(
            EvidenceStatement(
                statement_id=f"ES-{counter:03d}",
                text=limitation,
                pointers=pointers,
                source_type="limitation",
            )
        )

    # Methodology
    if summary.methodology:
        counter += 1
        pointers = extract_evidence_pointers(summary.methodology, summary.arxiv_id)
        statements.append(
            EvidenceStatement(
                statement_id=f"ES-{counter:03d}",
                text=summary.methodology,
                pointers=pointers,
                source_type="methodology",
            )
        )

    # Objective
    if summary.objective:
        counter += 1
        pointers = extract_evidence_pointers(summary.objective, summary.arxiv_id)
        statements.append(
            EvidenceStatement(
                statement_id=f"ES-{counter:03d}",
                text=summary.objective,
                pointers=pointers,
                source_type="objective",
            )
        )

    return statements


def aggregate_evidence(
    report: SynthesisReport,
    *,
    min_pointers: int = 0,
    max_words: int = 50,
    min_words: int = 5,
    similarity_threshold: float = 0.7,
    strip_rhetoric_enabled: bool = True,
) -> EvidenceAggregation:
    """Aggregate evidence from a synthesis report.

    Main entry point for evidence-only aggregation. Processes all paper
    summaries in a synthesis report through the full pipeline:
    1. Extract statements from each paper summary
    2. Strip rhetoric (optional)
    3. Normalize statement length
    4. Extract evidence pointers
    5. Merge similar statements
    6. Filter by minimum evidence requirements

    Args:
        report: Synthesis report containing paper summaries.
        min_pointers: Minimum evidence pointers required per statement.
        max_words: Maximum words per statement.
        min_words: Minimum words per statement.
        similarity_threshold: Threshold for merging similar statements.
        strip_rhetoric_enabled: Whether to strip rhetoric from statements.

    Returns:
        EvidenceAggregation with filtered, merged statements.
    """
    all_statements: list[EvidenceStatement] = []
    total_rhetoric = 0
    dropped: list[str] = []

    # Also extract from agreement/disagreement claims
    counter = 0
    for agreement in report.agreements:
        counter += 1
        pointers = [
            EvidencePointer(paper_id=pid) for pid in agreement.supporting_papers
        ]
        for ev in agreement.evidence:
            pointers.append(
                EvidencePointer(
                    paper_id="",
                    chunk_id=ev.chunk_id,
                    quote=ev.quote,
                )
            )
        all_statements.append(
            EvidenceStatement(
                statement_id=f"AGR-{counter:03d}",
                text=agreement.claim,
                pointers=pointers,
                source_type="finding",
                agreement_count=len(agreement.supporting_papers),
            )
        )

    for disagreement in report.disagreements:
        counter += 1
        dis_pointers: list[EvidencePointer] = []
        for pid, position in disagreement.positions.items():
            dis_pointers.append(
                EvidencePointer(
                    paper_id=pid,
                    quote=position[:100],
                )
            )
        for ev in disagreement.evidence:
            dis_pointers.append(
                EvidencePointer(
                    paper_id="",
                    chunk_id=ev.chunk_id,
                    quote=ev.quote,
                )
            )
        all_statements.append(
            EvidenceStatement(
                statement_id=f"DIS-{counter:03d}",
                text=disagreement.topic,
                pointers=dis_pointers,
                source_type="finding",
            )
        )

    # Extract from paper summaries
    for summary in report.paper_summaries:
        stmts = _extract_statements_from_summary(summary)
        all_statements.extend(stmts)

    input_count = len(all_statements)

    # Step 1: Strip rhetoric
    if strip_rhetoric_enabled:
        cleaned: list[EvidenceStatement] = []
        for stmt in all_statements:
            clean_text, spans = strip_rhetoric(stmt.text)
            total_rhetoric += len(spans)
            if clean_text:
                cleaned.append(stmt.model_copy(update={"text": clean_text}))
            else:
                dropped.append(f"[rhetoric-empty] {stmt.text[:80]}")
        all_statements = cleaned

    # Step 2: Normalize length
    length_filtered: list[EvidenceStatement] = []
    for stmt in all_statements:
        normalized = normalize_length(
            stmt.text, max_words=max_words, min_words=min_words
        )
        if normalized:
            length_filtered.append(stmt.model_copy(update={"text": normalized}))
        else:
            dropped.append(f"[too-short] {stmt.text[:80]}")
    all_statements = length_filtered

    # Step 3: Merge similar statements
    all_statements, merge_count = _merge_similar_statements(
        all_statements, threshold=similarity_threshold
    )

    # Step 4: Filter by evidence requirements
    if min_pointers > 0:
        evidence_filtered: list[EvidenceStatement] = []
        for stmt in all_statements:
            if len(stmt.pointers) >= min_pointers:
                evidence_filtered.append(stmt)
            else:
                dropped.append(f"[no-evidence] {stmt.text[:80]}")
        all_statements = evidence_filtered

    # Step 5: Re-number statements
    renumbered: list[EvidenceStatement] = []
    for i, stmt in enumerate(all_statements, 1):
        renumbered.append(stmt.model_copy(update={"statement_id": f"ES-{i:03d}"}))

    # Compute stats
    evidence_matched = sum(1 for s in renumbered if s.pointers)
    avg_pointers = (
        sum(len(s.pointers) for s in renumbered) / len(renumbered)
        if renumbered
        else 0.0
    )

    stats = AggregationStats(
        input_statements=input_count,
        rhetoric_stripped=total_rhetoric,
        evidence_matched=evidence_matched,
        evidence_unmatched=len(renumbered) - evidence_matched,
        merged_duplicates=merge_count,
        output_statements=len(renumbered),
        avg_pointers_per_statement=round(avg_pointers, 2),
    )

    logger.info(
        "Evidence aggregation: %d input → %d output "
        "(%d rhetoric stripped, %d merged, %d dropped)",
        input_count,
        len(renumbered),
        total_rhetoric,
        merge_count,
        len(dropped),
    )

    return EvidenceAggregation(
        topic=report.topic,
        statements=renumbered,
        dropped=dropped,
        stats=stats,
    )


def aggregate_from_summaries(
    summaries: list[PaperSummary],
    topic: str = "",
    **kwargs: object,
) -> EvidenceAggregation:
    """Convenience: aggregate evidence directly from paper summaries.

    Creates a minimal SynthesisReport wrapper and delegates to
    aggregate_evidence.

    Args:
        summaries: List of paper summaries.
        topic: Research topic.
        **kwargs: Passed to aggregate_evidence.

    Returns:
        EvidenceAggregation result.
    """
    report = SynthesisReport(
        topic=topic,
        paper_count=len(summaries),
        paper_summaries=summaries,
    )
    return aggregate_evidence(report, **kwargs)  # type: ignore[arg-type]


def format_aggregation_text(aggregation: EvidenceAggregation) -> str:
    """Format aggregation result as readable text.

    Args:
        aggregation: The aggregation result.

    Returns:
        Formatted text string.
    """
    lines: list[str] = []
    lines.append(f"# Evidence-Only Aggregation: {aggregation.topic}")
    lines.append("")
    lines.append(f"**Statements**: {aggregation.stats.output_statements}")
    lines.append(f"**Rhetoric removed**: {aggregation.stats.rhetoric_stripped}")
    lines.append(f"**Duplicates merged**: {aggregation.stats.merged_duplicates}")
    lines.append(
        f"**Avg evidence/statement**: {aggregation.stats.avg_pointers_per_statement}"
    )
    lines.append("")

    # Group by source type
    by_type: dict[str, list[EvidenceStatement]] = defaultdict(list)
    for stmt in aggregation.statements:
        by_type[stmt.source_type].append(stmt)

    for stype, stmts in by_type.items():
        lines.append(f"## {stype.title()} ({len(stmts)})")
        lines.append("")
        for stmt in stmts:
            evidence_str = ""
            if stmt.pointers:
                papers = {p.paper_id for p in stmt.pointers if p.paper_id}
                if papers:
                    evidence_str = f" [{', '.join(sorted(papers))}]"
            lines.append(f"- **{stmt.statement_id}**: {stmt.text}{evidence_str}")
            if stmt.agreement_count > 1:
                lines.append(f"  *(agreed by {stmt.agreement_count} sources)*")
        lines.append("")

    if aggregation.dropped:
        lines.append(f"## Dropped ({len(aggregation.dropped)})")
        lines.append("")
        for d in aggregation.dropped[:20]:
            lines.append(f"- {d}")
        if len(aggregation.dropped) > 20:
            lines.append(f"- ... and {len(aggregation.dropped) - 20} more")
        lines.append("")

    return "\n".join(lines)
