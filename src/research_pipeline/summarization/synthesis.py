"""Cross-paper synthesis: agreements, disagreements, open questions."""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from collections.abc import Iterable, Sequence

from research_pipeline.llm.base import LLMProvider
from research_pipeline.models.summary import (
    AssumptionRecord,
    ConfidenceLevel,
    ContradictionRecord,
    CrossPaperSynthesisRecord,
    ExtractedStatement,
    PaperExtractionRecord,
    PaperSummary,
    ReusableMechanism,
    SynthesisAgreement,
    SynthesisDisagreement,
    SynthesisFinding,
    SynthesisQuality,
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


def _statements_for(
    record: PaperExtractionRecord,
    categories: Iterable[str],
) -> list[ExtractedStatement]:
    """Collect statements from selected extraction categories."""
    statements: list[ExtractedStatement] = []
    for category in categories:
        statements.extend(getattr(record, category))
    return [s for s in statements if s.statement and s.statement != "not_reported"]


def _normalize_statement_key(statement: str) -> str:
    """Build a rough deterministic grouping key for a statement."""
    lowered = re.sub(r"[^a-z0-9\s]", " ", statement.lower())
    tokens = [
        token
        for token in lowered.split()
        if len(token) > 3
        and token
        not in {
            "paper",
            "method",
            "model",
            "results",
            "using",
            "based",
            "shows",
            "requires",
        }
    ]
    return " ".join(tokens[:6]) or lowered[:80]


def _confidence_from_support(
    paper_count: int,
    statements: Sequence[ExtractedStatement],
) -> ConfidenceLevel:
    """Grade synthesized confidence from support count and Step 1 confidence."""
    if paper_count >= 3 and all(
        s.confidence != ConfidenceLevel.LOW for s in statements
    ):
        return ConfidenceLevel.HIGH
    if paper_count >= 2:
        return ConfidenceLevel.MEDIUM
    return ConfidenceLevel.LOW


def _first_words(text: str, count: int = 8) -> str:
    """Return a compact label from the first words in text."""
    words = text.split()
    return " ".join(words[:count]) + ("..." if len(words) > count else "")


def _paper_lookup(records: Sequence[PaperExtractionRecord]) -> dict[str, str]:
    """Map paper IDs to titles."""
    return {record.paper_id: record.title for record in records}


def _build_corpus(records: Sequence[PaperExtractionRecord]) -> list[dict[str, str]]:
    """Build corpus metadata for synthesis output."""
    return [
        {
            "paper_id": record.paper_id,
            "version": record.version,
            "title": record.title,
            "year": record.year,
            "venue": record.venue,
        }
        for record in records
    ]


def _join_statements(statements: Sequence[ExtractedStatement]) -> str:
    """Join statement text for matrix cells."""
    return "; ".join(s.statement for s in statements) or "not_reported"


def _build_evidence_matrix(
    records: Sequence[PaperExtractionRecord],
) -> list[dict[str, str]]:
    """Build the paper-by-attribute evidence matrix."""
    rows: list[dict[str, str]] = []
    for record in records:
        rows.append(
            {
                "paper_id": record.paper_id,
                "title": record.title,
                "methods": _join_statements(record.methods),
                "datasets": _join_statements(record.datasets),
                "results": _join_statements(record.results),
                "assumptions": _join_statements(record.assumptions),
                "limitations": _join_statements(record.limitations),
                "reusable_mechanisms": _join_statements(record.reusable_mechanisms),
                "operational_characteristics": _join_statements(
                    record.operational_characteristics
                ),
            }
        )
    return rows


def _taxonomy_label(statement: str) -> str:
    """Assign a deterministic coarse taxonomy label."""
    lowered = statement.lower()
    labels = {
        "attention/transformer": ("attention", "transformer"),
        "retrieval/search": ("retrieval", "search", "index"),
        "graph/knowledge": ("graph", "knowledge"),
        "optimization/scheduling": ("optimization", "schedule", "scheduler"),
        "agent/workflow": ("agent", "workflow", "planner"),
        "security/privacy": ("privacy", "security", "encryption"),
        "evaluation/benchmark": ("benchmark", "evaluation", "dataset"),
    }
    for label, terms in labels.items():
        if any(term in lowered for term in terms):
            return label
    return _first_words(statement, 5).lower()


def _group_statements(
    records: Sequence[PaperExtractionRecord],
    categories: Iterable[str],
    *,
    taxonomy: bool = False,
) -> dict[str, list[tuple[PaperExtractionRecord, ExtractedStatement]]]:
    """Group selected statements across papers."""
    groups: defaultdict[str, list[tuple[PaperExtractionRecord, ExtractedStatement]]] = (
        defaultdict(list)
    )
    for record in records:
        for statement in _statements_for(record, categories):
            key = (
                _taxonomy_label(statement.statement)
                if taxonomy
                else _normalize_statement_key(statement.statement)
            )
            groups[key].append((record, statement))
    return dict(groups)


def _finding_from_group(
    finding_id: str,
    finding_type: str,
    label: str,
    items: Sequence[tuple[PaperExtractionRecord, ExtractedStatement]],
) -> SynthesisFinding:
    """Build a synthesis finding from grouped statements."""
    papers = sorted({record.paper_id for record, _ in items})
    evidence_ids = sorted(
        {
            evidence_id
            for _record, statement in items
            for evidence_id in statement.evidence_ids
        }
    )
    statements = [statement for _, statement in items]
    examples = "; ".join(_first_words(s.statement, 12) for s in statements[:3])
    return SynthesisFinding(
        finding_id=finding_id,
        finding=f"{label}: {examples}",
        finding_type=finding_type,
        supporting_papers=papers,
        evidence_ids=evidence_ids,
        confidence=_confidence_from_support(len(papers), statements),
        interpretation_notes=(
            "Grouped from Step 1 extraction records; review source statements for "
            "fine-grained wording."
        ),
    )


def _build_taxonomy(
    records: Sequence[PaperExtractionRecord],
) -> list[SynthesisFinding]:
    """Build a taxonomy of approaches from methods and contributions."""
    groups = _group_statements(records, ("methods", "contributions"), taxonomy=True)
    findings: list[SynthesisFinding] = []
    for i, (label, items) in enumerate(sorted(groups.items()), start=1):
        findings.append(_finding_from_group(f"TAX-{i:03d}", "taxonomy", label, items))
    return findings


def _build_recurring_patterns(
    records: Sequence[PaperExtractionRecord],
) -> list[SynthesisFinding]:
    """Identify recurring mechanisms or patterns across papers."""
    groups = _group_statements(
        records,
        ("reusable_mechanisms", "methods", "contributions"),
    )
    findings: list[SynthesisFinding] = []
    counter = 1
    for label, items in sorted(groups.items()):
        papers = {record.paper_id for record, _ in items}
        if len(papers) < 2:
            continue
        findings.append(
            _finding_from_group(f"PAT-{counter:03d}", "recurring_pattern", label, items)
        )
        counter += 1
    return findings


def _build_assumption_map(
    records: Sequence[PaperExtractionRecord],
) -> list[AssumptionRecord]:
    """Consolidate assumptions across papers."""
    groups = _group_statements(records, ("assumptions", "scale_assumptions"))
    assumptions: list[AssumptionRecord] = []
    for i, (label, items) in enumerate(sorted(groups.items()), start=1):
        papers = sorted({record.paper_id for record, _ in items})
        evidence_ids = sorted(
            {
                evidence_id
                for _record, statement in items
                for evidence_id in statement.evidence_ids
            }
        )
        assumptions.append(
            AssumptionRecord(
                assumption_id=f"ASM-{i:03d}",
                assumption=items[0][1].statement if items else label,
                source_papers=papers,
                evidence_ids=evidence_ids,
                scope="; ".join(sorted({statement.category for _, statement in items})),
                risk_if_false=(
                    "Downstream design choices based on this assumption may not hold "
                    "outside the paper conditions."
                ),
            )
        )
    return assumptions


def _build_contradiction_map(
    records: Sequence[PaperExtractionRecord],
) -> list[ContradictionRecord]:
    """Detect lexical contradictions across extracted statements."""
    candidates: list[tuple[PaperExtractionRecord, ExtractedStatement]] = []
    for record in records:
        candidates.extend(
            (record, statement)
            for statement in _statements_for(
                record,
                (
                    "results",
                    "limitations",
                    "assumptions",
                    "operational_characteristics",
                ),
            )
        )

    contradictions: list[ContradictionRecord] = []
    seen: set[tuple[str, str]] = set()
    for i, (record_a, statement_a) in enumerate(candidates):
        for record_b, statement_b in candidates[i + 1 :]:
            if record_a.paper_id == record_b.paper_id:
                continue
            pair_key = (
                min(record_a.paper_id, record_b.paper_id),
                max(record_a.paper_id, record_b.paper_id),
            )
            if pair_key in seen:
                continue
            if not _findings_share_topic(statement_a.statement, statement_b.statement):
                continue
            opposition = _has_opposition(statement_a.statement, statement_b.statement)
            if opposition is None:
                continue
            seen.add(pair_key)
            contradictions.append(
                ContradictionRecord(
                    contradiction_id=f"CON-{len(contradictions) + 1:03d}",
                    topic=f"Potential conflict ({opposition})",
                    positions={
                        record_a.paper_id: statement_a.statement,
                        record_b.paper_id: statement_b.statement,
                    },
                    source_papers=sorted([record_a.paper_id, record_b.paper_id]),
                    evidence_ids=sorted(
                        {*statement_a.evidence_ids, *statement_b.evidence_ids}
                    ),
                    severity=ConfidenceLevel.MEDIUM,
                )
            )
    return contradictions


def _build_evidence_strength_map(
    records: Sequence[PaperExtractionRecord],
    patterns: Sequence[SynthesisFinding],
) -> list[SynthesisFinding]:
    """Build confidence-graded findings from result/contribution groups."""
    findings = list(patterns)
    groups = _group_statements(records, ("results", "contributions"))
    counter = len(findings) + 1
    for label, items in sorted(groups.items()):
        if not items:
            continue
        findings.append(
            _finding_from_group(f"EVS-{counter:03d}", "evidence_strength", label, items)
        )
        counter += 1
    return findings


def _build_operational_implications(
    records: Sequence[PaperExtractionRecord],
) -> list[SynthesisFinding]:
    """Translate operational statements into conditional implications."""
    categories = (
        "operational_characteristics",
        "scale_assumptions",
        "hardware_requirements",
        "software_tools",
        "cost_drivers",
        "security_privacy",
        "reliability",
        "observability_needs",
    )
    findings: list[SynthesisFinding] = []
    counter = 1
    for record in records:
        for statement in _statements_for(record, categories):
            findings.append(
                SynthesisFinding(
                    finding_id=f"OPI-{counter:03d}",
                    finding=(
                        f"If this constraint is relevant, account for: "
                        f"{statement.statement}"
                    ),
                    finding_type="operational_implication",
                    supporting_papers=[record.paper_id],
                    evidence_ids=list(statement.evidence_ids),
                    confidence=statement.confidence,
                    interpretation_notes=(
                        "Conditional implication; not an architecture choice."
                    ),
                )
            )
            counter += 1
    return findings


def _build_production_readiness(
    records: Sequence[PaperExtractionRecord],
) -> list[SynthesisFinding]:
    """Build coarse production-readiness notes from evaluation evidence."""
    findings: list[SynthesisFinding] = []
    for i, record in enumerate(records, start=1):
        evaluation = _statements_for(record, ("evaluation", "results"))
        readiness = "theoretical or not reported"
        confidence = ConfidenceLevel.LOW
        if record.results and record.evaluation:
            readiness = "empirically evaluated in the paper context"
            confidence = ConfidenceLevel.MEDIUM
        findings.append(
            SynthesisFinding(
                finding_id=f"PRD-{i:03d}",
                finding=f"{record.paper_id}: {readiness}",
                finding_type="production_readiness",
                supporting_papers=[record.paper_id],
                evidence_ids=sorted(
                    {eid for statement in evaluation for eid in statement.evidence_ids}
                ),
                confidence=confidence,
                limitations=[
                    statement.statement
                    for statement in record.limitations
                    if statement.statement != "not_reported"
                ],
            )
        )
    return findings


def _build_reusable_mechanisms(
    records: Sequence[PaperExtractionRecord],
) -> list[ReusableMechanism]:
    """Build reusable mechanism inventory."""
    mechanisms: list[ReusableMechanism] = []
    counter = 1
    for record in records:
        statements = record.reusable_mechanisms or record.methods[:1]
        for statement in statements:
            if statement.statement == "not_reported":
                continue
            mechanisms.append(
                ReusableMechanism(
                    mechanism_id=f"MECH-{counter:03d}",
                    name=_first_words(statement.statement, 5),
                    description=statement.statement,
                    source_papers=[record.paper_id],
                    evidence_ids=list(statement.evidence_ids),
                    generality=(
                        record.generality[0].statement
                        if record.generality
                        and record.generality[0].statement != "not_reported"
                        else ""
                    ),
                    known_constraints=[
                        limitation.statement
                        for limitation in record.limitations
                        if limitation.statement != "not_reported"
                    ],
                )
            )
            counter += 1
    return mechanisms


def _build_design_implications(
    patterns: Sequence[SynthesisFinding],
    operational: Sequence[SynthesisFinding],
) -> list[SynthesisFinding]:
    """Build conditional design implications without choosing an architecture."""
    implications: list[SynthesisFinding] = []
    counter = 1
    for source in [*patterns, *operational]:
        implications.append(
            SynthesisFinding(
                finding_id=f"DES-{counter:03d}",
                finding=(
                    "If the target requirements match this evidence context, "
                    f"consider the trade-off described by: {source.finding}"
                ),
                finding_type="design_implication",
                supporting_papers=list(source.supporting_papers),
                evidence_ids=list(source.evidence_ids),
                confidence=source.confidence,
                interpretation_notes=(
                    "Conditional implication only; no implementation selected."
                ),
            )
        )
        counter += 1
    return implications


def _build_risk_register(
    records: Sequence[PaperExtractionRecord],
    assumptions: Sequence[AssumptionRecord],
    contradictions: Sequence[ContradictionRecord],
) -> list[SynthesisFinding]:
    """Build risks from limitations, assumptions, and contradictions."""
    risks: list[SynthesisFinding] = []
    counter = 1
    for record in records:
        for limitation in _statements_for(record, ("limitations",)):
            risks.append(
                SynthesisFinding(
                    finding_id=f"RSK-{counter:03d}",
                    finding=f"Limitation risk: {limitation.statement}",
                    finding_type="risk",
                    supporting_papers=[record.paper_id],
                    evidence_ids=list(limitation.evidence_ids),
                    confidence=limitation.confidence,
                )
            )
            counter += 1
    for assumption in assumptions:
        risks.append(
            SynthesisFinding(
                finding_id=f"RSK-{counter:03d}",
                finding=f"Assumption risk: {assumption.risk_if_false}",
                finding_type="risk",
                supporting_papers=list(assumption.source_papers),
                evidence_ids=list(assumption.evidence_ids),
                confidence=ConfidenceLevel.MEDIUM,
            )
        )
        counter += 1
    for contradiction in contradictions:
        risks.append(
            SynthesisFinding(
                finding_id=f"RSK-{counter:03d}",
                finding=f"Contradiction risk: {contradiction.topic}",
                finding_type="risk",
                supporting_papers=list(contradiction.source_papers),
                evidence_ids=list(contradiction.evidence_ids),
                confidence=contradiction.severity,
            )
        )
        counter += 1
    return risks


def _build_unresolved_questions(
    records: Sequence[PaperExtractionRecord],
    contradictions: Sequence[ContradictionRecord],
) -> list[str]:
    """Build open questions from Step 1 uncertainties and Step 2 conflicts."""
    questions: list[str] = []
    for record in records:
        for uncertainty in record.uncertainties:
            questions.append(f"[{record.paper_id}] {uncertainty}")
        for field in record.quality.missing_critical_fields:
            questions.append(f"[{record.paper_id}] Missing critical field: {field}")
    for contradiction in contradictions:
        questions.append(f"How should downstream design handle {contradiction.topic}?")
    return questions or ["No unresolved questions identified from Step 1 records."]


def _build_traceability(
    findings: Sequence[SynthesisFinding],
    assumptions: Sequence[AssumptionRecord],
    contradictions: Sequence[ContradictionRecord],
    mechanisms: Sequence[ReusableMechanism],
) -> list[dict[str, str]]:
    """Build a flat traceability appendix."""
    rows: list[dict[str, str]] = []
    for finding in findings:
        rows.append(
            {
                "item_id": finding.finding_id,
                "item_type": finding.finding_type,
                "statement": finding.finding,
                "papers": ", ".join(finding.supporting_papers),
                "evidence_ids": ", ".join(finding.evidence_ids),
                "confidence": finding.confidence.value,
            }
        )
    for assumption in assumptions:
        rows.append(
            {
                "item_id": assumption.assumption_id,
                "item_type": "assumption",
                "statement": assumption.assumption,
                "papers": ", ".join(assumption.source_papers),
                "evidence_ids": ", ".join(assumption.evidence_ids),
                "confidence": ConfidenceLevel.MEDIUM.value,
            }
        )
    for contradiction in contradictions:
        rows.append(
            {
                "item_id": contradiction.contradiction_id,
                "item_type": "contradiction",
                "statement": contradiction.topic,
                "papers": ", ".join(contradiction.source_papers),
                "evidence_ids": ", ".join(contradiction.evidence_ids),
                "confidence": contradiction.severity.value,
            }
        )
    for mechanism in mechanisms:
        rows.append(
            {
                "item_id": mechanism.mechanism_id,
                "item_type": "reusable_mechanism",
                "statement": mechanism.description,
                "papers": ", ".join(mechanism.source_papers),
                "evidence_ids": ", ".join(mechanism.evidence_ids),
                "confidence": ConfidenceLevel.MEDIUM.value,
            }
        )
    return rows


_PRESCRIPTIVE_RE = re.compile(
    r"\b(must use|should use|best solution|we will use|choose|clearly dominates)\b",
    re.IGNORECASE,
)


def _score_synthesis_quality(
    records: Sequence[PaperExtractionRecord],
    synthesis: CrossPaperSynthesisRecord,
) -> SynthesisQuality:
    """Compute deterministic quality checks for structured synthesis."""
    covered = {
        paper_id
        for row in synthesis.traceability_appendix
        for paper_id in row.get("papers", "").split(", ")
        if paper_id
    }
    coverage = len(covered) / max(len(records), 1)
    traced = [
        row for row in synthesis.traceability_appendix if row.get("evidence_ids", "")
    ]
    traceability = len(traced) / max(len(synthesis.traceability_appendix), 1)
    report_text = " ".join(
        row.get("statement", "") for row in synthesis.traceability_appendix
    )
    neutrality = 0.0 if _PRESCRIPTIVE_RE.search(report_text) else 1.0
    contradiction_coverage = 1.0 if synthesis.contradiction_map else 0.8

    warnings: list[str] = []
    if coverage < 1.0:
        warnings.append("Not every paper appears in traceability appendix")
    if traceability < 0.8:
        warnings.append("Low evidence-ID coverage in traceability appendix")
    if neutrality < 1.0:
        warnings.append("Potential architecture-prescriptive language detected")

    return SynthesisQuality(
        coverage_score=round(coverage, 2),
        traceability_score=round(traceability, 2),
        neutrality_score=round(neutrality, 2),
        contradiction_coverage_score=round(contradiction_coverage, 2),
        warnings=warnings,
    )


def synthesize_extractions(
    records: list[PaperExtractionRecord],
    topic: str,
) -> CrossPaperSynthesisRecord:
    """Produce a design-neutral synthesis from Step 1 extraction records."""
    taxonomy = _build_taxonomy(records)
    patterns = _build_recurring_patterns(records)
    assumptions = _build_assumption_map(records)
    contradictions = _build_contradiction_map(records)
    evidence_strength = _build_evidence_strength_map(records, patterns)
    operational = _build_operational_implications(records)
    production = _build_production_readiness(records)
    mechanisms = _build_reusable_mechanisms(records)
    design = _build_design_implications(patterns, operational)
    risks = _build_risk_register(records, assumptions, contradictions)
    traceability = _build_traceability(
        [
            *taxonomy,
            *patterns,
            *evidence_strength,
            *operational,
            *production,
            *design,
            *risks,
        ],
        assumptions,
        contradictions,
        mechanisms,
    )

    synthesis = CrossPaperSynthesisRecord(
        topic=topic,
        corpus=_build_corpus(records),
        methodology=[
            (
                "Loaded Step 1 PaperExtractionRecord artifacts as the only "
                "evidence source."
            ),
            "Normalized statements by category, confidence, and evidence IDs.",
            (
                "Built matrices, taxonomy, assumptions, contradictions, and "
                "risks from structured fields."
            ),
            "Kept design implications conditional and architecture-neutral.",
        ],
        taxonomy=taxonomy,
        evidence_matrix=_build_evidence_matrix(records),
        recurring_patterns=patterns,
        assumption_map=assumptions,
        contradiction_map=contradictions,
        evidence_strength_map=evidence_strength,
        operational_implications=operational,
        production_readiness=production,
        reusable_mechanism_inventory=mechanisms,
        design_implications=design,
        unresolved_questions=_build_unresolved_questions(records, contradictions),
        risk_register=risks,
        traceability_appendix=traceability,
    )
    return synthesis.model_copy(
        update={"quality": _score_synthesis_quality(records, synthesis)}
    )


def project_structured_synthesis_to_report(
    synthesis: CrossPaperSynthesisRecord,
    summaries: list[PaperSummary],
) -> SynthesisReport:
    """Project rich Step 2 synthesis into the legacy SynthesisReport shape."""
    agreements = [
        SynthesisAgreement(
            claim=finding.finding,
            supporting_papers=list(finding.supporting_papers),
        )
        for finding in [*synthesis.recurring_patterns, *synthesis.evidence_strength_map]
    ]
    disagreements = [
        SynthesisDisagreement(
            topic=contradiction.topic,
            positions=dict(contradiction.positions),
        )
        for contradiction in synthesis.contradiction_map
    ]
    return SynthesisReport(
        topic=synthesis.topic,
        paper_count=len(summaries),
        agreements=agreements,
        disagreements=disagreements,
        open_questions=list(synthesis.unresolved_questions),
        paper_summaries=summaries,
    )


def _markdown_table(rows: Sequence[dict[str, str]], columns: Sequence[str]) -> str:
    """Render a simple Markdown table."""
    if not rows:
        return "_No entries._\n"
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    body = [
        "| "
        + " | ".join(str(row.get(column, "")).replace("|", "\\|") for column in columns)
        + " |"
        for row in rows
    ]
    return "\n".join([header, divider, *body]) + "\n"


def render_structured_synthesis_markdown(
    synthesis: CrossPaperSynthesisRecord,
) -> str:
    """Render the structured Step 2 synthesis as Markdown."""
    lines = [
        f"# Structured Research Synthesis: {synthesis.topic}",
        "",
        "## Executive Summary",
        "",
        (
            f"This synthesis covers {len(synthesis.corpus)} papers and is built "
            "from Step 1 extraction records. It preserves evidence, assumptions, "
            "contradictions, and conditional implications without selecting an "
            "architecture."
        ),
        "",
        "## Scope and Corpus",
        "",
        _markdown_table(
            synthesis.corpus, ("paper_id", "version", "title", "year", "venue")
        ),
        "## Methodology",
        "",
        *[f"- {item}" for item in synthesis.methodology],
        "",
        "## Taxonomy of Approaches",
        "",
        *[
            f"- **{f.confidence}** {f.finding} [{', '.join(f.supporting_papers)}]"
            for f in synthesis.taxonomy
        ],
        "",
        "## Evidence Matrix",
        "",
        _markdown_table(
            synthesis.evidence_matrix,
            (
                "paper_id",
                "methods",
                "datasets",
                "results",
                "assumptions",
                "limitations",
            ),
        ),
        "## Recurring Mechanisms and Patterns",
        "",
        *[
            f"- **{f.confidence}** {f.finding} [{', '.join(f.supporting_papers)}]"
            for f in synthesis.recurring_patterns
        ],
        "",
        "## Assumption Map",
        "",
        *[
            f"- **{a.assumption_id}** {a.assumption} "
            f"[{', '.join(a.source_papers)}] Risk if false: {a.risk_if_false}"
            for a in synthesis.assumption_map
        ],
        "",
        "## Contradiction Map",
        "",
        *[
            f"- **{c.contradiction_id}** {c.topic}: "
            + "; ".join(
                f"{paper}: {position}" for paper, position in c.positions.items()
            )
            for c in synthesis.contradiction_map
        ],
        *(
            ["- No contradictions identified."]
            if not synthesis.contradiction_map
            else []
        ),
        "",
        "## Evidence Strength Map",
        "",
        *[
            f"- **{f.confidence}** {f.finding} "
            f"[evidence: {', '.join(f.evidence_ids) or 'none'}]"
            for f in synthesis.evidence_strength_map
        ],
        "",
        "## Operational Implications",
        "",
        *[
            f"- **{f.confidence}** {f.finding}"
            for f in synthesis.operational_implications
        ],
        "",
        "## Production Readiness",
        "",
        *[f"- **{f.confidence}** {f.finding}" for f in synthesis.production_readiness],
        "",
        "## Reusable Mechanism Inventory",
        "",
        *[
            f"- **{m.mechanism_id}** {m.name}: {m.description} "
            f"[{', '.join(m.source_papers)}]"
            for m in synthesis.reusable_mechanism_inventory
        ],
        "",
        "## Design Implications",
        "",
        *[f"- **{f.confidence}** {f.finding}" for f in synthesis.design_implications],
        "",
        "## Unresolved Questions",
        "",
        *[f"- {question}" for question in synthesis.unresolved_questions],
        "",
        "## Risk Register",
        "",
        *[f"- **{f.confidence}** {f.finding}" for f in synthesis.risk_register],
        "",
        "## Traceability Appendix",
        "",
        _markdown_table(
            synthesis.traceability_appendix,
            ("item_id", "item_type", "papers", "evidence_ids", "confidence"),
        ),
    ]
    return "\n".join(lines).strip() + "\n"


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
