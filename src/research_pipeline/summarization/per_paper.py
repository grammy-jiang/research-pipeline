"""Per-paper evidence-driven summarization."""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from research_pipeline.extraction.chunking import chunk_markdown
from research_pipeline.extraction.retrieval import retrieve_relevant_chunks
from research_pipeline.llm.base import LLMProvider
from research_pipeline.models.extraction import ChunkMetadata
from research_pipeline.models.summary import (
    ConfidenceLevel,
    EvidenceSnippet,
    ExtractedStatement,
    ExtractionMetadata,
    ExtractionQuality,
    PaperExtractionRecord,
    PaperSummary,
    StatementType,
    SummaryEvidence,
)

logger = logging.getLogger(__name__)

_PAPER_SUMMARY_PROMPT = """\
You are an expert academic paper analyst. Summarize the following paper.

Paper title: {title}
Research topic: {topic}

Below are the most relevant excerpts from the paper:

{chunks_text}

Respond with a JSON object containing:
- "objective": a concise statement of the paper's main objective
- "methodology": a description of the key methodology or approach
- "findings": a list of key findings (strings)
- "limitations": a list of limitations (strings)
- "uncertainties": a list of uncertain or unresolved items (strings)
"""

_PAPER_EXTRACTION_PROMPT = """\
You are an expert academic paper analyst. Extract a schema-first record from
the paper excerpts below. Do not write a narrative summary.

Paper title: {title}
Research topic: {topic}

Available evidence IDs:
{evidence_catalog}

Paper excerpts:
{chunks_text}

Respond with valid JSON. Include these keys:
- context, problem, contributions, methods, datasets, evaluation, results
- operational_characteristics, scale_assumptions, hardware_requirements
- software_tools, cost_drivers, security_privacy, reliability
- observability_needs, assumptions, limitations, future_work
- reusable_mechanisms, generality, uncertainties

Each list item must be an object with:
- "statement": atomic statement text
- "statement_type": one of author_claim, empirical_result, interpretation,
  model_inference
- "confidence": HIGH, MEDIUM, or LOW
- "evidence_ids": list of available evidence IDs
- "notes": optional caveat text

Use "not_reported" only when the excerpts do not support a field. Do not invent
missing metrics, hardware, datasets, or limitations.
"""

_EXTRACTION_CATEGORIES: tuple[str, ...] = (
    "context",
    "problem",
    "contributions",
    "methods",
    "datasets",
    "evaluation",
    "results",
    "operational_characteristics",
    "scale_assumptions",
    "hardware_requirements",
    "software_tools",
    "cost_drivers",
    "security_privacy",
    "reliability",
    "observability_needs",
    "assumptions",
    "limitations",
    "future_work",
    "reusable_mechanisms",
    "generality",
)

_CRITICAL_EXTRACTION_FIELDS: tuple[str, ...] = (
    "problem",
    "contributions",
    "methods",
    "results",
    "limitations",
    "assumptions",
)

_GENERIC_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\b(as above|same as above|not applicable|n/?a)\b", re.IGNORECASE),
    re.compile(r"^\s*(none|unknown|unclear)\s*$", re.IGNORECASE),
)


def _build_paper_prompt(
    title: str,
    topic_terms: list[str],
    relevant: Sequence[tuple[object, str, float]],
) -> str:
    """Build the LLM prompt for per-paper summarization.

    Args:
        title: Paper title.
        topic_terms: Research topic terms.
        relevant: Retrieved relevant chunks (meta, text, score).

    Returns:
        Formatted prompt string.
    """
    chunks_text = "\n\n---\n\n".join(
        f"[Chunk {i + 1}]\n{chunk_text}"
        for i, (_, chunk_text, _) in enumerate(relevant)
    )
    return _PAPER_SUMMARY_PROMPT.format(
        title=title,
        topic=", ".join(topic_terms),
        chunks_text=chunks_text,
    )


def _parse_llm_paper_response(
    response: dict,  # type: ignore[type-arg]
    arxiv_id: str,
    version: str,
    title: str,
    evidence: list[SummaryEvidence],
) -> PaperSummary:
    """Parse LLM response dict into a PaperSummary.

    Args:
        response: Raw LLM response dict.
        arxiv_id: Paper arXiv ID.
        version: Paper version string.
        title: Paper title.
        evidence: Pre-built evidence references.

    Returns:
        PaperSummary constructed from LLM output.

    Raises:
        KeyError: If required fields are missing.
        TypeError: If field types are wrong.
    """
    objective = str(response["objective"])
    methodology = str(response["methodology"])
    findings = [str(f) for f in response["findings"]]
    limitations = [str(lim) for lim in response["limitations"]]
    uncertainties = [str(u) for u in response["uncertainties"]]

    return PaperSummary(
        arxiv_id=arxiv_id,
        version=version,
        title=title,
        objective=objective,
        methodology=f"[LLM] {methodology}",
        findings=findings,
        limitations=limitations,
        evidence=evidence,
        uncertainties=uncertainties,
    )


def _build_evidence_snippets(
    arxiv_id: str,
    relevant: Sequence[tuple[ChunkMetadata, str, float]],
) -> list[EvidenceSnippet]:
    """Build stable evidence snippets from retrieved chunks."""
    snippets: list[EvidenceSnippet] = []
    for i, (meta, chunk_text, score) in enumerate(relevant, start=1):
        quote = " ".join(chunk_text.split())[:400]
        snippets.append(
            EvidenceSnippet(
                evidence_id=f"E{i:03d}",
                paper_id=arxiv_id,
                chunk_id=meta.chunk_id,
                line_range=meta.source_span,
                section=meta.section_path,
                quote=quote,
                confidence=ConfidenceLevel.HIGH
                if score > 1.0
                else ConfidenceLevel.MEDIUM,
            )
        )
    return snippets


def _build_extraction_prompt(
    title: str,
    topic_terms: list[str],
    relevant: Sequence[tuple[ChunkMetadata, str, float]],
    evidence: list[EvidenceSnippet],
) -> str:
    """Build the structured extraction prompt."""
    chunks_text = "\n\n---\n\n".join(
        f"[{evidence[i].evidence_id}] [Chunk {i + 1}]\n{chunk_text}"
        for i, (_, chunk_text, _) in enumerate(relevant)
    )
    evidence_catalog = "\n".join(
        f"- {ev.evidence_id}: {ev.chunk_id}, {ev.line_range}, {ev.section}"
        for ev in evidence
    )
    return _PAPER_EXTRACTION_PROMPT.format(
        title=title,
        topic=", ".join(topic_terms),
        chunks_text=chunks_text,
        evidence_catalog=evidence_catalog,
    )


def _coerce_confidence(value: object) -> ConfidenceLevel:
    """Coerce arbitrary LLM confidence labels into the enum."""
    text = str(value or "MEDIUM").upper()
    if text in {"HIGH", "MEDIUM", "LOW"}:
        return ConfidenceLevel(text)
    if text.startswith("H"):
        return ConfidenceLevel.HIGH
    if text.startswith("L"):
        return ConfidenceLevel.LOW
    return ConfidenceLevel.MEDIUM


def _coerce_statement_type(value: object) -> StatementType:
    """Coerce arbitrary LLM statement type labels into the enum."""
    text = str(value or StatementType.AUTHOR_CLAIM.value).lower().replace("-", "_")
    aliases = {
        "author": StatementType.AUTHOR_CLAIM,
        "claim": StatementType.AUTHOR_CLAIM,
        "result": StatementType.EMPIRICAL_RESULT,
        "empirical": StatementType.EMPIRICAL_RESULT,
        "inference": StatementType.MODEL_INFERENCE,
        "inferred": StatementType.MODEL_INFERENCE,
    }
    if text in {item.value for item in StatementType}:
        return StatementType(text)
    return aliases.get(text, StatementType.INTERPRETATION)


def _as_statement_items(value: object) -> list[object]:
    """Normalize a field value into a list of statement-like items."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _parse_statement_items(
    raw_items: object,
    category: str,
    paper_id: str,
    default_evidence_id: str,
) -> list[ExtractedStatement]:
    """Parse LLM or fallback values into extracted statements."""
    statements: list[ExtractedStatement] = []
    for i, item in enumerate(_as_statement_items(raw_items), start=1):
        statement = ""
        statement_type = StatementType.AUTHOR_CLAIM
        confidence = ConfidenceLevel.MEDIUM
        evidence_ids: list[str] = []
        notes = ""

        if isinstance(item, dict):
            statement = str(
                item.get("statement")
                or item.get("text")
                or item.get("claim")
                or item.get("description")
                or ""
            )
            statement_type = _coerce_statement_type(item.get("statement_type"))
            confidence = _coerce_confidence(item.get("confidence"))
            evidence_value = item.get("evidence_ids") or item.get("evidence") or []
            evidence_ids = [str(ev) for ev in _as_statement_items(evidence_value) if ev]
            notes = str(item.get("notes") or "")
        else:
            statement = str(item)

        statement = statement.strip()
        if not statement:
            continue
        if not evidence_ids and statement != "not_reported" and default_evidence_id:
            evidence_ids = [default_evidence_id]
        if statement == "not_reported":
            statement_type = StatementType.INTERPRETATION
            confidence = ConfidenceLevel.LOW
            evidence_ids = []

        statements.append(
            ExtractedStatement(
                statement_id=f"{paper_id}:{category}:{i:03d}",
                statement=statement,
                category=category,
                statement_type=statement_type,
                confidence=confidence,
                evidence_ids=evidence_ids,
                notes=notes,
            )
        )
    return statements


def _all_extracted_statements(
    record: PaperExtractionRecord,
) -> list[ExtractedStatement]:
    """Return all statement fields from an extraction record."""
    statements: list[ExtractedStatement] = []
    for category in _EXTRACTION_CATEGORIES:
        statements.extend(getattr(record, category))
    return statements


def _with_evidence_support_links(
    evidence: list[EvidenceSnippet],
    statements: Sequence[ExtractedStatement],
) -> list[EvidenceSnippet]:
    """Populate evidence.supports from statement evidence IDs."""
    supports: dict[str, list[str]] = {ev.evidence_id: [] for ev in evidence}
    for statement in statements:
        for evidence_id in statement.evidence_ids:
            if evidence_id in supports:
                supports[evidence_id].append(statement.statement_id)
    return [
        ev.model_copy(update={"supports": supports.get(ev.evidence_id, [])})
        for ev in evidence
    ]


def score_extraction_quality(record: PaperExtractionRecord) -> ExtractionQuality:
    """Compute deterministic quality metadata for a paper extraction."""
    missing = [
        field
        for field in _CRITICAL_EXTRACTION_FIELDS
        if not getattr(record, field)
        or all(s.statement == "not_reported" for s in getattr(record, field))
    ]
    statements = _all_extracted_statements(record)
    substantive = [s for s in statements if s.statement != "not_reported"]
    unsupported = [s for s in substantive if not s.evidence_ids]
    generic = [
        s
        for s in substantive
        if any(pattern.search(s.statement) for pattern in _GENERIC_PATTERNS)
    ]

    completeness = 1.0 - (len(missing) / max(len(_CRITICAL_EXTRACTION_FIELDS), 1))
    provenance = 1.0
    if substantive:
        provenance = 1.0 - (len(unsupported) / len(substantive))
    specificity = 1.0
    if substantive:
        specificity = 1.0 - (len(generic) / len(substantive))

    warnings: list[str] = []
    if missing:
        warnings.append("Missing critical fields: " + ", ".join(missing))
    if unsupported:
        warnings.append(f"{len(unsupported)} substantive statement(s) lack evidence")
    if generic:
        warnings.append(f"{len(generic)} generic statement(s) detected")
    if not record.evidence:
        warnings.append("No evidence snippets available")

    return ExtractionQuality(
        completeness_score=round(max(completeness, 0.0), 2),
        provenance_score=round(max(provenance, 0.0), 2),
        specificity_score=round(max(specificity, 0.0), 2),
        unsupported_statement_count=len(unsupported),
        missing_critical_fields=missing,
        warnings=warnings,
    )


def _parse_extraction_response(
    response: dict[str, Any],
    arxiv_id: str,
    version: str,
    title: str,
    evidence: list[EvidenceSnippet],
    model_name: str = "",
) -> PaperExtractionRecord:
    """Parse a structured LLM response into a paper extraction record."""
    default_evidence_id = evidence[0].evidence_id if evidence else ""
    field_values: dict[str, list[ExtractedStatement]] = {}
    for category in _EXTRACTION_CATEGORIES:
        field_values[category] = _parse_statement_items(
            response.get(category, []),
            category,
            arxiv_id,
            default_evidence_id,
        )

    # Backward-compatible conversion from the older paper_summary schema.
    if not any(field_values.values()) and {"objective", "methodology"} <= set(response):
        field_values["problem"] = _parse_statement_items(
            response.get("objective", ""),
            "problem",
            arxiv_id,
            default_evidence_id,
        )
        field_values["methods"] = _parse_statement_items(
            response.get("methodology", ""),
            "methods",
            arxiv_id,
            default_evidence_id,
        )
        field_values["results"] = _parse_statement_items(
            response.get("findings", []),
            "results",
            arxiv_id,
            default_evidence_id,
        )
        field_values["limitations"] = _parse_statement_items(
            response.get("limitations", []),
            "limitations",
            arxiv_id,
            default_evidence_id,
        )

    record = PaperExtractionRecord.model_validate(
        {
            "paper_id": arxiv_id,
            "version": version,
            "title": title,
            "evidence": evidence,
            "uncertainties": [str(u) for u in response.get("uncertainties", [])],
            "extraction_metadata": ExtractionMetadata(
                mode="structured",
                model=model_name,
                generated_at=datetime.now(tz=UTC).isoformat(),
            ),
            **field_values,
        }
    )
    linked_evidence = _with_evidence_support_links(
        record.evidence, _all_extracted_statements(record)
    )
    record = record.model_copy(update={"evidence": linked_evidence})
    return record.model_copy(update={"quality": score_extraction_quality(record)})


def _section_matches(section: str, *terms: str) -> bool:
    """Return whether a section path contains any term."""
    lowered = section.lower()
    return any(term in lowered for term in terms)


def _fallback_statement(
    paper_id: str,
    category: str,
    text: str,
    evidence_id: str = "",
    confidence: ConfidenceLevel = ConfidenceLevel.LOW,
) -> ExtractedStatement:
    """Create a fallback statement."""
    evidence_ids = [evidence_id] if evidence_id and text != "not_reported" else []
    return ExtractedStatement(
        statement_id=f"{paper_id}:{category}:001",
        statement=text,
        category=category,
        statement_type=StatementType.INTERPRETATION,
        confidence=confidence,
        evidence_ids=evidence_ids,
    )


def _build_template_extraction(
    arxiv_id: str,
    version: str,
    title: str,
    relevant: Sequence[tuple[ChunkMetadata, str, float]],
    evidence: list[EvidenceSnippet],
) -> PaperExtractionRecord:
    """Build a valid low-confidence extraction without an LLM."""
    default_evidence_id = evidence[0].evidence_id if evidence else ""
    section_texts = [
        (meta.section_path, chunk_text) for meta, chunk_text, _ in relevant
    ]

    method_ev = default_evidence_id
    result_ev = default_evidence_id
    limitation_ev = default_evidence_id
    for i, (section, _text) in enumerate(section_texts):
        ev_id = evidence[i].evidence_id if i < len(evidence) else default_evidence_id
        if _section_matches(section, "method", "approach", "architecture"):
            method_ev = ev_id
        if _section_matches(section, "result", "evaluation", "experiment"):
            result_ev = ev_id
        if _section_matches(section, "limitation", "discussion", "future"):
            limitation_ev = ev_id

    problem = _fallback_statement(
        arxiv_id,
        "problem",
        f"Paper addresses the research topic through: {title}",
        default_evidence_id,
    )
    methods = _fallback_statement(
        arxiv_id,
        "methods",
        "Method details require structured LLM extraction; see linked evidence chunks.",
        method_ev,
    )
    results = _fallback_statement(
        arxiv_id,
        "results",
        "Results require structured LLM extraction; see linked evidence chunks.",
        result_ev,
    )
    limitations = _fallback_statement(
        arxiv_id,
        "limitations",
        "Limitations were not reliably extracted in template fallback mode.",
        limitation_ev,
    )
    assumptions = _fallback_statement(
        arxiv_id,
        "assumptions",
        "not_reported",
    )

    record = PaperExtractionRecord(
        paper_id=arxiv_id,
        version=version,
        title=title,
        problem=[problem],
        methods=[methods],
        results=[results],
        limitations=[limitations],
        assumptions=[assumptions],
        evidence=evidence,
        uncertainties=["Full structured LLM extraction not enabled."],
        extraction_metadata=ExtractionMetadata(
            mode="template_fallback",
            generated_at=datetime.now(tz=UTC).isoformat(),
        ),
    )
    linked_evidence = _with_evidence_support_links(
        record.evidence, _all_extracted_statements(record)
    )
    record = record.model_copy(update={"evidence": linked_evidence})
    return record.model_copy(update={"quality": score_extraction_quality(record)})


def project_extraction_to_summary(record: PaperExtractionRecord) -> PaperSummary:
    """Project a rich extraction record into the legacy PaperSummary shape."""

    def first_statement(*categories: str) -> str:
        for category in categories:
            statements = cast(list[ExtractedStatement], getattr(record, category))
            for statement in statements:
                if statement.statement and statement.statement != "not_reported":
                    return statement.statement
        return f"See paper: {record.title}"

    findings = [
        statement.statement
        for category in ("contributions", "results", "reusable_mechanisms")
        for statement in getattr(record, category)
        if statement.statement != "not_reported"
    ]
    limitations = [
        statement.statement
        for statement in record.limitations
        if statement.statement != "not_reported"
    ]
    evidence = [
        SummaryEvidence(
            chunk_id=ev.chunk_id,
            line_range=ev.line_range,
            quote=ev.quote,
        )
        for ev in record.evidence
    ]
    return PaperSummary(
        arxiv_id=record.paper_id,
        version=record.version,
        title=record.title,
        objective=first_statement("problem", "context"),
        methodology=first_statement("methods"),
        findings=findings or [first_statement("results", "contributions")],
        limitations=limitations or ["No limitations extracted."],
        evidence=evidence,
        uncertainties=list(record.uncertainties),
    )


def render_extraction_markdown(record: PaperExtractionRecord) -> str:
    """Render a human-readable Markdown view of a paper extraction."""
    lines = [
        f"# Paper Extraction: {record.title}",
        "",
        f"- Paper ID: `{record.paper_id}{record.version}`",
        f"- Extraction mode: `{record.extraction_metadata.mode}`",
        (
            "- Quality: "
            f"completeness={record.quality.completeness_score}, "
            f"provenance={record.quality.provenance_score}, "
            f"specificity={record.quality.specificity_score}"
        ),
        "",
    ]
    for category in _EXTRACTION_CATEGORIES:
        statements = getattr(record, category)
        title = category.replace("_", " ").title()
        lines.extend([f"## {title}", ""])
        if not statements:
            lines.append("- not_reported")
        for statement in statements:
            evidence = ", ".join(statement.evidence_ids) or "no evidence"
            lines.append(
                "- "
                f"[{statement.confidence}] "
                f"({statement.statement_type}) "
                f"{statement.statement} "
                f"[evidence: {evidence}]"
            )
        lines.append("")
    lines.extend(["## Evidence", ""])
    for ev in record.evidence:
        lines.append(
            f"- `{ev.evidence_id}` `{ev.chunk_id}` {ev.line_range}: {ev.quote}"
        )
    if record.uncertainties:
        lines.extend(["", "## Uncertainties", ""])
        lines.extend(f"- {item}" for item in record.uncertainties)
    if record.quality.warnings:
        lines.extend(["", "## Quality Warnings", ""])
        lines.extend(f"- {warning}" for warning in record.quality.warnings)
    return "\n".join(lines).strip() + "\n"


def extract_paper(
    markdown_path: Path,
    arxiv_id: str,
    version: str,
    title: str,
    topic_terms: list[str],
    max_chunk_tokens: int = 1500,
    top_k_chunks: int = 10,
    llm_provider: LLMProvider | None = None,
) -> PaperExtractionRecord:
    """Generate a schema-first extraction record from one paper."""
    text = markdown_path.read_text(encoding="utf-8")
    chunks = chunk_markdown(text, arxiv_id, max_tokens=max_chunk_tokens)
    query_terms = topic_terms or title.split()
    relevant = retrieve_relevant_chunks(chunks, query_terms, top_k=top_k_chunks)
    if not relevant:
        relevant = [
            (meta, chunk_text, 0.0) for meta, chunk_text in chunks[:top_k_chunks]
        ]
    evidence = _build_evidence_snippets(arxiv_id, relevant)

    if llm_provider is not None:
        try:
            prompt = _build_extraction_prompt(title, query_terms, relevant, evidence)
            response = llm_provider.call(
                prompt,
                schema_id="paper_extraction_record",
                temperature=0.0,
            )
            record = _parse_extraction_response(
                response,
                arxiv_id,
                version,
                title,
                evidence,
                model_name=llm_provider.model_name(),
            )
            logger.info(
                "Generated structured extraction for %s with quality %.2f",
                arxiv_id,
                record.quality.completeness_score,
            )
            return record
        except Exception as exc:
            logger.warning(
                "Structured extraction failed for %s, falling back: %s",
                arxiv_id,
                exc,
            )

    record = _build_template_extraction(arxiv_id, version, title, relevant, evidence)
    logger.info(
        "Generated fallback extraction for %s with %d evidence refs",
        arxiv_id,
        len(evidence),
    )
    return record


def summarize_paper(
    markdown_path: Path,
    arxiv_id: str,
    version: str,
    title: str,
    topic_terms: list[str],
    max_chunk_tokens: int = 1500,
    top_k_chunks: int = 10,
    llm_provider: LLMProvider | None = None,
) -> PaperSummary:
    """Generate a summary of a single paper from its Markdown.

    When *llm_provider* is given, the top relevant chunks are sent to the
    LLM for structured summarization.  Falls back to template mode if the
    LLM call fails or if no provider is supplied.

    Args:
        markdown_path: Path to the converted Markdown.
        arxiv_id: Paper arXiv ID.
        version: Paper version.
        title: Paper title.
        topic_terms: Terms for relevance-based chunk retrieval.
        max_chunk_tokens: Maximum tokens per chunk.
        top_k_chunks: Number of top chunks to use.
        llm_provider: Optional LLM provider for summarization.

    Returns:
        PaperSummary with evidence references.
    """
    text = markdown_path.read_text(encoding="utf-8")
    chunks = chunk_markdown(text, arxiv_id, max_tokens=max_chunk_tokens)
    relevant = retrieve_relevant_chunks(chunks, topic_terms, top_k=top_k_chunks)

    evidence = [
        SummaryEvidence(
            chunk_id=meta.chunk_id,
            line_range=meta.source_span,
            quote=chunk_text[:200],
        )
        for meta, chunk_text, _score in relevant
    ]

    # --- LLM mode ---
    if llm_provider is not None:
        try:
            prompt = _build_paper_prompt(title, topic_terms, relevant)
            response = llm_provider.call(
                prompt, schema_id="paper_summary", temperature=0.0
            )
            summary = _parse_llm_paper_response(
                response, arxiv_id, version, title, evidence
            )
            logger.info(
                "Generated LLM summary for %s with %d evidence refs",
                arxiv_id,
                len(evidence),
            )
            return summary
        except Exception as exc:
            logger.warning(
                "LLM summarization failed for %s, falling back to template: %s",
                arxiv_id,
                exc,
            )

    # --- Template mode (fallback) ---
    summary = PaperSummary(
        arxiv_id=arxiv_id,
        version=version,
        title=title,
        objective=f"See paper: {title}",
        methodology="(LLM summarization not enabled — template mode)",
        findings=[
            f"Relevant section: {meta.section_path}" for meta, _, _ in relevant[:5]
        ],
        limitations=["Template-based summary; enable LLM for detailed analysis."],
        evidence=evidence,
        uncertainties=["Full LLM-based extraction not enabled."],
    )

    logger.info(
        "Generated template summary for %s with %d evidence refs",
        arxiv_id,
        len(evidence),
    )
    return summary
